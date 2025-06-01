import json
import csv
import torch
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, login
import wandb

class TritonDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        instruction = example["instruction"]
        response = example["response"]
        
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        tokens = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze().clone()
        }


class SFTTrainingSetup:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model_and_tokenizer(self):
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            trust_remote_code=True
        )
        
        if torch.cuda.device_count() <= 1:
            self.model = self.model.to(self.device)
        
        return self.model, self.tokenizer
    
    
    def prepare_dataset(self, dataset_path_or_name, use_hf=True):
        """Prepare dataset for training"""

        if use_hf:
            print(f"Loading dataset from {dataset_path_or_name}...")
            hf_dataset = load_dataset(dataset_path_or_name, split="train")

            data = []
            for example in hf_dataset:
                data.append({
                    'instruction': example['prompt'],
                    'response': example['completion'],
                    'id': example.get('id', ''),
                    'type': example.get('type', ''),
                    'operation': example.get('operation', '')
                })
        else:
            with open(dataset_path_or_name, 'r') as f:
                data = [json.loads(line.strip()) for line in f]
            
        print(f"Loaded {len(data)} examples")

        train_size = int(0.9 * len(data))
        train_data = data[:train_size]
        eval_data = data[train_size:]
        
        train_dataset = TritonDataset(train_data, self.tokenizer)
        eval_dataset = TritonDataset(eval_data, self.tokenizer)
        
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset

    def upload_checkpoint_to_hf(self, local_path, repo_name, commit_message="Training checkpoint"):
        try:
            api = HfApi()
            api.upload_file(
                folder_path=local_path,
                repo_id=repo_name,
                repo_type="model",
                commit_message=commit_message,
            )
            print(f"Uploaded checkpoint to {repo_name}")
        except Exception as e:
            print(f"Error uploading checkpoint to {repo_name}: {e}")

    def setup_logging(self, output_dir):
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.train_log_file = os.path.join(self.log_dir, "training_log.jsonl")
        self.metrics_file = os.path.join(self.log_dir, "metrics.csv")

        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'train_loss', 'eval_loss', 'learning_rate'])

        print(f"Logging to {self.log_dir}")

    def log_metrics(self, epoch, step, train_loss, eval_loss=None, lr=None):
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "learning_rate": lr
        }

        with open(self.train_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, step, train_loss, eval_loss, lr])


    def train_epoch(self, model, dataloader, optimizer, epoch, gradient_accumulation_steps=4):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        accumulated_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += accumulated_loss
                current_lr = optimizer.param_groups[0]['lr']

                actual_step = (batch_idx + 1) // gradient_accumulation_steps
                global_step = (epoch - 1) * (len(dataloader) // gradient_accumulation_steps) + actual_step


                wandb.log({
                    "train/loss": accumulated_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch,
                    "train/step": actual_step
                }, step=global_step)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'eff_bs': gradient_accumulation_steps * dataloader.batch_size
                })
                
                if actual_step % 10 == 0:
                    self.log_metrics(epoch, batch_idx, loss.item(), lr=current_lr)
                    print(f"Epoch {epoch}, Step {actual_step}, Loss: {accumulated_loss:.4f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / len(dataloader) // gradient_accumulation_steps
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, model, dataloader):
        """Evaluate the model"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Eval Loss: {avg_loss:.4f}")

        wandb.log({"eval/loss": avg_loss})
        return avg_loss
    
    
    def train(self, dataset_path_or_name, output_dir="./sft_checkpoints", 
              num_epochs=3, batch_size=4, learning_rate=5e-5, use_hf=True, 
              hf_repo_name=None, wandb_project="triton-sft"):
        """Run SFT training"""
        
        if self.model is None:
            self.load_model_and_tokenizer()

        wandb.init(
            project=wandb_project,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_name": self.model_name,
                "dataset": dataset_path_or_name
            }
        )


        self.setup_logging(output_dir)

        if hf_repo_name:
            print("Logging into HF")
            login()
        
        train_dataset, eval_dataset = self.prepare_dataset(dataset_path_or_name, use_hf)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        num_training_steps = len(train_dataloader) * num_epochs
        warmup_steps = num_training_steps // 5
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        best_eval_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(self.model, train_dataloader, optimizer, epoch + 1, gradient_accumulation_steps=4)
            eval_loss = self.evaluate(self.model, eval_dataloader)

            wandb.log({
                "epoch/train_loss": train_loss,
                "epoch/eval_loss": eval_loss,
                "epoch/number": epoch + 1
            })

            current_lr = optimizer.param_groups[0]['lr']
            self.log_metrics(epoch + 1, len(train_dataloader), train_loss, eval_loss, current_lr)
            
            if epoch < warmup_steps // len(train_dataloader):
                scheduler.step()
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model(output_dir)
                print(f"Saved new best model with eval loss: {eval_loss:.4f}")

                if hf_repo_name:
                    self.upload_checkpoint_to_hf(
                        output_dir, 
                        hf_repo_name, 
                        f"Best model - epoch {epoch+1}, eval loss: {eval_loss:.4f}"
                    )
            
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            self.save_model(checkpoint_dir)

            if hf_repo_name:
                self.upload_checkpoint_to_hf(
                    checkpoint_dir,
                    hf_repo_name,
                    f"Epoch {epoch+1} checkpoint - train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}"
                )
            
            print(f"Epoch {epoch+1}/{num_epochs} completed")
            print("-" * 50)

        if hf_repo_name:
            self.upload_checkpoint_to_hf(
                output_dir,
                hf_repo_name,
                f"Final model - epoch {num_epochs}, eval loss: {eval_loss:.4f}"
            )
        
        print(f"Training completed! Best eval loss: {best_eval_loss:.4f}")
        print(f"Model saved to {output_dir}")

        wandb.finish()
        print(f"view training run at: {wandb.run.url}")
        
        return self.model
    
    def save_model(self, output_dir):
        """Save model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def test_model(self, checkpoint_path, test_prompt):
        """Test the trained model"""
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.device_count() > 1 else None
        )
        
        if torch.cuda.device_count() <= 1:
            model = model.to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        formatted_prompt = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):]
        
        return response


if __name__ == "__main__":
    trainer_setup = SFTTrainingSetup()
    
    model = trainer_setup.train(
        dataset_path_or_name="cdreetz/triton-sft-dataset",
        output_dir="./qwen_triton_sft",
        num_epochs=3,
        batch_size=2,  # Smaller batch size for 1.5B model
        learning_rate=5e-5,
        use_hf=True,
        hf_repo_name="cdreetz/triton-sft-dataset",
        wandb_project="triton-kernel-sft"
    )
    
    test_prompt = "Can you implement an elementwise addition triton kernel? Write both the kernel method and the corresponding launch method."
    
    response = trainer_setup.test_model(
        "./qwen_triton_sft", 
        test_prompt
    )
    
    print("Test Response:")
    print(response)

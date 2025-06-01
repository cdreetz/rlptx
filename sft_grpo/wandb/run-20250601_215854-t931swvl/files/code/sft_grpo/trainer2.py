import json
import csv
import torch
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, login
import wandb
from torch.amp import autocast, GradScaler
import numpy as np

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
        
        # Format input text
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        tokens = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels - mask instruction part
        labels = tokens['input_ids'].squeeze().clone()
        
        # Find where assistant response starts
        # Tokenize just the instruction part to find its length
        instruction_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        instruction_tokens = self.tokenizer(
            instruction_text,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        instruction_length = instruction_tokens['input_ids'].shape[1]
        
        # Mask the instruction (set to -100 so it's ignored in loss)
        labels[:instruction_length] = -100
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': labels
        }


class SFTTrainingSetup:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.scaler = GradScaler()  # For mixed precision
        
    def load_model_and_tokenizer(self):
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use float16 for mixed precision
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
            writer.writerow(['timestamp', 'epoch', 'step', 'train_loss', 'train_perplexity', 
                           'eval_loss', 'eval_perplexity', 'learning_rate', 'grad_norm'])

        print(f"Logging to {self.log_dir}")

    def log_metrics(self, epoch, step, global_step, train_loss, eval_loss=None, 
                   train_perplexity=None, eval_perplexity=None, lr=None, grad_norm=None):
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_perplexity": train_perplexity,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "learning_rate": lr,
            "grad_norm": grad_norm
        }

        with open(self.train_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, step, global_step, train_loss, train_perplexity,
                           eval_loss, eval_perplexity, lr, grad_norm])


    def train_epoch(self, model, dataloader, optimizer, scheduler, epoch, eval_dataloader, eval_steps, gradient_accumulation_steps=2):
        """Train for one epoch with mixed precision"""
        model.train()
        total_loss = 0
        accumulated_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed precision training
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            
            # Scale loss and backward
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step with scaler
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulated_loss
                current_lr = optimizer.param_groups[0]['lr']

                actual_step = (batch_idx + 1) // gradient_accumulation_steps
                
                # Calculate perplexity
                train_perplexity = np.exp(accumulated_loss)

                wandb.log({
                    "train/loss": accumulated_loss,
                    "train/perplexity": train_perplexity,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/epoch": epoch,
                    "train/step": actual_step,
                    "train/global_step": self.global_step
                }, step=self.global_step)
                
                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'ppl': f'{train_perplexity:.2f}',
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{grad_norm.item():.3f}',
                    'eff_bs': gradient_accumulation_steps * dataloader.batch_size
                })

                if self.global_step % eval_steps == 0:
                    eval_loss, eval_perplexity = self.evaluate(model, eval_dataloader)
                    model.train()

                    print(f"Epoch {epoch}, Step {actual_step}, Global Step {self.global_step}, "
                          f"Eval Loss: {eval_loss:.4f}, Eval Perplexity: {eval_perplexity:.2f}")
                
                if actual_step % 10 == 0:
                    self.log_metrics(epoch, actual_step, self.global_step, accumulated_loss, 
                                   train_perplexity=train_perplexity, lr=current_lr, grad_norm=grad_norm.item())

                accumulated_loss = 0
        
        avg_loss = total_loss / (len(dataloader) // gradient_accumulation_steps)
        avg_perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.2f}")
        return avg_loss
    
    def evaluate(self, model, dataloader):
        """Evaluate the model with perplexity calculation"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss)
        
        print(f"Eval Loss: {avg_loss:.4f}, Eval Perplexity: {perplexity:.2f}")

        wandb.log({
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity
        })
        
        return avg_loss, perplexity
    
    
    def train(self, dataset_path_or_name, output_dir="./sft_checkpoints", 
              num_epochs=3, batch_size=4, learning_rate=5e-5, use_hf=True, 
              hf_repo_name=None, wandb_project="triton-sft", eval_steps=50):
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
            weight_decay=0.1,
            eps=1e-8
        )
        
        gradient_accumulation_steps = 2
        steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        total_training_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.1 * total_training_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        best_eval_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(
                    self.model, 
                    train_dataloader, 
                    optimizer, 
                    scheduler,
                    epoch + 1, 
                    eval_dataloader,
                    eval_steps,
                    gradient_accumulation_steps=2
                )
            eval_loss, eval_perplexity = self.evaluate(self.model, eval_dataloader)

            wandb.log({
                "epoch/train_loss": train_loss,
                "epoch/eval_loss": eval_loss,
                "epoch/eval_perplexity": eval_perplexity,
                "epoch/number": epoch + 1,
                "epoch/global_step": self.global_step
            }, step=self.global_step)

            current_lr = optimizer.param_groups[0]['lr']
            self.log_metrics(epoch + 1, len(train_dataloader), self.global_step, train_loss, 
                           eval_loss, np.exp(train_loss), eval_perplexity, current_lr)
            
            
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
            torch_dtype=torch.float16,  # Use float16 for inference too
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
        learning_rate=2e-5,
        use_hf=True,
        hf_repo_name="cdreetz/triton-sft-dataset",
        wandb_project="triton-kernel-sft",
        eval_steps=50
    )
    
    test_prompt = "Can you implement an elementwise addition triton kernel? Write both the kernel method and the corresponding launch method."
    
    response = trainer_setup.test_model(
        "./qwen_triton_sft", 
        test_prompt
    )
    
    print("Test Response:")
    print(response)

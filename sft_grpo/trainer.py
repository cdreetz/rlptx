import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm


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
    
    
    def prepare_dataset(self, dataset_path):
        """Prepare dataset for training"""
        
        with open(dataset_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
        
        print(f"Loaded {len(data)} examples")
        
        train_size = int(0.9 * len(data))
        train_data = data[:train_size]
        eval_data = data[train_size:]
        
        train_dataset = TritonDataset(train_data, self.tokenizer)
        eval_dataset = TritonDataset(eval_data, self.tokenizer)
        
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def train_epoch(self, model, dataloader, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
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
        return avg_loss
    
    
    def train(self, dataset_path, output_dir="./sft_checkpoints", 
              num_epochs=3, batch_size=4, learning_rate=2e-4):
        """Run SFT training"""
        
        if self.model is None:
            self.load_model_and_tokenizer()
        
        train_dataset, eval_dataset = self.prepare_dataset(dataset_path)
        
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
            weight_decay=0.01
        )
        
        num_training_steps = len(train_dataloader) * num_epochs
        warmup_steps = num_training_steps // 10
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        best_eval_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(self.model, train_dataloader, optimizer, epoch + 1)
            eval_loss = self.evaluate(self.model, eval_dataloader)
            
            if epoch < warmup_steps // len(train_dataloader):
                scheduler.step()
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model(output_dir)
                print(f"Saved new best model with eval loss: {eval_loss:.4f}")
            
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            self.save_model(checkpoint_dir)
            
            print(f"Epoch {epoch+1}/{num_epochs} completed")
            print("-" * 50)
        
        print(f"Training completed! Best eval loss: {best_eval_loss:.4f}")
        print(f"Model saved to {output_dir}")
        
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
        dataset_path="sft_executable_data.jsonl",
        output_dir="./qwen_triton_sft",
        num_epochs=3,
        batch_size=2,  # Smaller batch size for 1.5B model
        learning_rate=1e-4
    )
    
    test_prompt = "Can you implement an elementwise addition triton kernel? Write both the kernel method and the corresponding launch method."
    
    response = trainer_setup.test_model(
        "./qwen_triton_sft", 
        test_prompt
    )
    
    print("Test Response:")
    print(response)
#!/usr/bin/env python3
"""
SFT trainer to finetune Qwen 1.5B on Triton kernel dataset
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import argparse
import os

class TritonKernelDataset(Dataset):
    """Dataset for SFT on Triton kernel generation"""
    
    def __init__(self, dataset_file, tokenizer, max_length=2048):
        print(f"Loading dataset from {dataset_file}")
        with open(dataset_file, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Create instruction-following format
        instruction = f"""Convert this PyTorch model to a Triton kernel implementation.

Your response must contain EXACTLY two functions:
1. A function named `triton_kernel` decorated with @triton.jit  
2. A function named `triton_wrapper` that calls the kernel

PyTorch code:
{example['torch_code']}"""
        
        response = example['triton_code']
        
        # Format as chat template
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # For instruction tuning, we typically mask the instruction part
        # and only compute loss on the response part
        
        # Find where assistant response starts
        assistant_start = formatted_text.find("assistant")
        if assistant_start != -1:
            # Tokenize just the instruction part to find where to start loss computation
            instruction_part = formatted_text[:assistant_start]
            instruction_tokens = self.tokenizer(
                instruction_part, 
                add_special_tokens=False,
                return_tensors=None
            )['input_ids']
            
            # Create labels: -100 for instruction tokens, actual tokens for response
            labels = input_ids.copy()
            if len(instruction_tokens) > 0:
                for i in range(min(len(instruction_tokens), len(labels))):
                    labels[i] = -100
        else:
            # Fallback: use all tokens for loss
            labels = input_ids.copy()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer"""
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
    )
    
    # Prepare for training
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def run_sft_training(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    dataset_file="sft_dataset.json",
    output_dir="./sft_model",
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_length=2048,
    device="cuda"
):
    """Run SFT training"""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Create dataset
    train_dataset = TritonKernelDataset(dataset_file, tokenizer, max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=False,
        bf16=True if device == "cuda" else False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
        gradient_checkpointing=True,  # Save memory
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting SFT training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("SFT training complete!")

def test_sft_model(model_path, test_prompt=None):
    """Test the SFT model with a simple example"""
    
    print(f"Testing SFT model from {model_path}")
    
    # Load trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Default test prompt
    if test_prompt is None:
        test_prompt = """Convert this PyTorch model to a Triton kernel implementation.

Your response must contain EXACTLY two functions:
1. A function named `triton_kernel` decorated with @triton.jit  
2. A function named `triton_wrapper` that calls the kernel

PyTorch code:
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0

def get_inputs():
    return [torch.randn(1024)]

def get_init_inputs():
    return []"""
    
    # Format as chat
    messages = [{"role": "user", "content": test_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    print("Generated Triton kernel:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # Try to verify it works
    try:
        from evaluator import extract_kernel_methods
        from evaluate_template import evaluate
        
        torch_code = """import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0

def get_inputs():
    return [torch.randn(1024)]

def get_init_inputs():
    return []"""
        
        kernel_code, wrapper_code = extract_kernel_methods(generated_text)
        if kernel_code and wrapper_code:
            result = evaluate(kernel_code, wrapper_code, torch_code)
            print(f"Verification result: {result}")
        else:
            print("Could not extract kernel and wrapper methods")
            
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training for Triton Kernels")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name")
    parser.add_argument("--dataset_file", type=str, default="sft_dataset.json", help="SFT dataset file")
    parser.add_argument("--output_dir", type=str, default="./sft_model", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    
    # Actions
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--model_path", type=str, help="Path to trained model for testing")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.train:
        run_sft_training(
            model_name=args.model_name,
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            device=device
        )
    
    if args.test:
        model_path = args.model_path or args.output_dir
        test_sft_model(model_path)
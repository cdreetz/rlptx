"""
Utility functions for GRPO training
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Any
from transformers import PreTrainedModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def get_per_token_logps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_start_idx: int
) -> torch.Tensor:
    """
    Get log probabilities for generated tokens.
    
    Args:
        model: The language model
        input_ids: Token IDs for the sequence
        attention_mask: Attention mask for the sequence
        logits_start_idx: Index to start computing logits from
        
    Returns:
        Tensor of log probabilities for each token
    """
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # Get the relevant logits (generated tokens only)
    shift_logits = outputs.logits[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    
    # Only compute for completion part
    start_pos = input_ids.size(1) - logits_start_idx
    shift_logits = shift_logits[:, -logits_start_idx:, :]
    shift_input_ids = shift_input_ids[:, -logits_start_idx:]
    
    # Compute log probs
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs


def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """
    Write generation log to a file.
    
    Args:
        log_data: Dictionary of log data
        log_file: Path to write log to
    """
    with open(log_file, 'w') as f:
        f.write(f"Prompt: {log_data['prompt']['text']}\n")
        f.write(f"Ground Truth: {log_data['prompt']['answer']}\n\n")
        
        for i, gen in enumerate(log_data['generations']):
            f.write(f"Generation {i+1}:\n")
            f.write(f"{gen['response']}\n\n")
            f.write("Scores:\n")
            for name, score in gen['scores'].items():
                f.write(f"  {name}: {score:.4f}\n")
            f.write("\n" + "-"*50 + "\n\n")



def get_client():
    client = OpenAI(
        api_key=os.getenv("LLAMA_API_KEY"),
        base_url="https://api.llama.com/compat/v1/",
    )
    return client

def llama_chat(prompt):
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response

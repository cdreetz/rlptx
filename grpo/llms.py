"""
Model loading and utility functions for LLMs
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_llm_tokenizer(model_name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a model and its tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model onto ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_name} on {device}...")
    
    # Load tokenizer with appropriate settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
    
    # Load model with BF16 precision when available
    model_kwargs = {}
    if device == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        **model_kwargs
    )
    
    return model, tokenizer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_llm_tokenizer(model_name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load and configure a language model and its tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer
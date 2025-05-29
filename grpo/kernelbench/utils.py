import random
import numpy as np
import torch
import torch.nn.functional as F
import json
def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def selective_log_softmax(logits, index):
    """Memory-efficient log_softmax -> gather operation."""
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        # More stable for bfloat16
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    """Get per-token log probabilities from model."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude last logit

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)


def write_generation_log(log_data: dict, log_file: str) -> None:
    """Write generation log to file."""
    with open(log_file, 'w') as f:
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")
        f.write("#### TORCH BASELINE ####\n\n")
        f.write(log_data['prompt']['answer'] + "\n\n")

        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
            f.write(gen['response'] + "\n\n")
            f.write(f"#### GENERATION {i} SCORES ####\n")

            for score_name, score_value in gen['scores'].items():
                f.write(f"{score_name}: {score_value}\n")
            f.write("\n")
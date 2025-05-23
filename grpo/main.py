"""
GRPO training loop for Triton kernel generation.

Implements Group Relative Policy Optimization for training language models
to generate better Triton kernels using the 3-part reward system.
"""

import os
import json
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
import torch.nn.functional as F

# Import our custom modules
from evaluator import TritonKernelEvaluator
from rldataset import get_dataloaders


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def selective_log_softmax(logits, index):
    """Memory-efficient log_softmax -> gather operation."""
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
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
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]  # Exclude last logit
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """Generate multiple completion sequences for a given prompt."""

    # Format prompt with system message
    chat_prompt = [
        {'role': 'system', 'content': 'You are an expert at writing high-performance Triton kernels. Write clean, efficient code with proper imports, BLOCK_SIZE constants, masking, and @triton.jit decorator. Provide only kernel code without explanation.'},
        {'role': 'user', 'content': prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(chat_prompt, tokenize=False)

    # Tokenize prompt
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt and repeat for multiple generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id
    )

    # Generate completions
    prompt_completion_ids = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=generation_config
    )

    # Extract completion portion
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Create completion mask (stop at EOS)
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text


def score_completions(
    completions_text: list[str],
    prompt: str,
    spec: dict,
    evaluator: TritonKernelEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """Score model completions and compute advantages for training."""

    # Build log data
    log_data = {
        'prompt': {
            'text': prompt,
            'spec': spec
        },
        'generations': []
    }

    # Format for evaluator
    mock_prompts = [[{'content': prompt}]] * len(completions_text)
    mock_completions = [[{'content': completion}] for completion in completions_text]

    # Get rewards from evaluator
    rewards_per_func, metrics = evaluator.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=spec,  # spec contains input_shapes, output_shape, operation, optimization_level
        device=device
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data for logging
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **evaluator.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages using group statistics
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary stats
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data


def compute_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the GRPO loss between current and base model."""

    logits_to_keep = completion_ids.size(1)

    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = get_per_token_logps(base_model, prompt_completion_ids, attention_mask, logits_to_keep)

    # Get training model logits
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # Compute GRPO loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    metrics["kl"] = mean_kl.item()

    return loss, metrics


def grpo_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    spec: dict,
    evaluator: TritonKernelEvaluator,
    device: str,
    round_num: int,
    training_log_dir: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss for a single training step."""

    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, prompt, device, args
    )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, prompt, spec, evaluator, device, args
    )

    # Write training log
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    write_generation_log(log_data, log_file)

    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics


def write_generation_log(log_data: dict, log_file: str) -> None:
    """Write generation log to file."""
    with open(log_file, 'w') as f:
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")
        f.write("#### KERNEL SPEC ####\n\n")
        f.write(json.dumps(log_data['prompt']['spec'], indent=2) + "\n\n")

        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
            f.write(gen['response'] + "\n\n")
            f.write(f"#### GENERATION {i} SCORES ####\n")

            f.write(f"Compilation: {gen['scores']['compilation']}\n")
            f.write(f"Correctness: {gen['scores']['correctness']}\n")
            f.write(f"Optimization: {gen['scores']['optimization']}\n")
            f.write(f"Total reward: {gen['scores']['total_reward']}\n\n")


def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader,
    evaluator: TritonKernelEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """Evaluate model on test set."""

    print("Running evaluation on test set...")

    total_scores = defaultdict(float)
    num_examples = 0

    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()

    with open(log_file, 'w') as f:
        for prompt, spec in tqdm(test_loader, desc="Evaluating"):
            # Generate completions
            _, _, _, _, completions_text, _ = generate_completions(
                model, tokenizer, prompt, device, args
            )

            # Score completions
            mock_prompts = [[{'content': prompt}]] * len(completions_text)
            mock_completions = [[{'content': completion}] for completion in completions_text]

            rewards_per_func, metrics = evaluator.compute_rewards(
                prompts=mock_prompts,
                completions=mock_completions,
                answer=spec,
                device=device
            )

            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Example {num_examples}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Spec: {json.dumps(spec, indent=2)}\n")
            f.write(f"Response: {completions_text[0]}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}

    # Use compilation rate as primary accuracy metric
    accuracy = avg_scores.get('compilation_rate', 0.0) * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    return avg_scores, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Triton kernels")

    # Model and dataset
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model to train")
    parser.add_argument("--dataset_path", type=str, default="data/triton_kernels_dataset_v5.json", help="Path to kernel dataset")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Gradient clipping")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Warmup percentage")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=8, help="Number of generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Max completion length")

    # Reference model updates
    parser.add_argument("--update_ref_model", action="store_true", help="Update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="Reference model update frequency")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, help="Reference model mixup alpha")

    # Logging and evaluation
    parser.add_argument("--output_dir", type=str, default="triton_grpo_output", help="Output directory")
    parser.add_argument("--eval_iterations", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=200, help="Model save frequency")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_llm_tokenizer(args.model_name, device)
    base_model, _ = get_llm_tokenizer(args.model_name, device)

    # Load dataset
    print("Loading dataset...")
    train_loader, test_loader = get_dataloaders(args.dataset_path)
    print(f"Train examples: {len(train_loader)}, Test examples: {len(test_loader)}")

    # Setup evaluator
    evaluator = TritonKernelEvaluator()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (step / warmup_steps) if step < warmup_steps else 1.0
    )

    # Training loop
    print("Starting training...")
    optimizer.zero_grad()
    train_metrics_total = {}

    for round_num in tqdm(range(args.num_train_iters), desc="Training"):

        # Evaluation
        if round_num % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model, tokenizer, test_loader, evaluator, device, args, round_num
            )
            print(f"Eval at step {round_num}: Compilation rate = {eval_accuracy:.1f}%")

        # Update reference model
        if args.update_ref_model and (round_num + 1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get training example
        prompt, spec = next(train_loader)

        # GRPO training step
        loss, train_metrics = grpo_loss(
            model, base_model, tokenizer, prompt, spec, evaluator,
            device, round_num, train_log_dir, args
        )

        # Backprop
        loss.backward()
        scheduler.step()

        # Optimizer step with gradient accumulation
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = loss.item()
        train_metrics_total[round_num] = train_metrics

        # Save training logs
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)

        # Save model checkpoint
        if (round_num + 1) % args.save_steps == 0:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{round_num + 1}"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{round_num + 1}"))

    # Final save
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print("Training complete!")


if __name__ == "__main__":
    main()

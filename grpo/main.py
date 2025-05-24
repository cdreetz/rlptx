"""
GRPO training for Triton kernel generation using Qwen3's thinking capabilities.
Based on proven GRPO implementation for reasoning models.
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    GenerationConfig
)

# Import our custom modules
import rldataset
import evaluator
import utils
import llms


def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldataset.TritonKernelLoader,
    eval_class: evaluator.TritonKernelEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """Evaluate model performance on test set."""
    print("Running evaluation on test set...")
    
    # track metrics
    total_scores = defaultdict(float)
    num_examples = 0
    total_accuracy = 0.0

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
            
            rewards_per_func, metrics = eval_class.compute_rewards(
                prompts=mock_prompts,
                completions=mock_completions,
                answer=spec,
                device=device
            )

            total_accuracy += metrics['compilation_rate']
            
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
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")

    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = total_accuracy / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    return avg_scores, accuracy


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """Generate multiple completion sequences for a given prompt."""

    # Format messages for Qwen3
    messages = [
        {'role': 'system', 'content': rldataset.SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt and repeat for multiple generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]

    # Repeat prompt for multiple generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=0.6,  # Qwen3 recommended for thinking
        top_p=0.95,
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


def extract_thinking_and_kernel(completion_text: str) -> tuple[str, str]:
    """Extract thinking portion and kernel portion from completion."""
    thinking_part = ""
    kernel_part = ""
    
    if '<think>' in completion_text and '</think>' in completion_text:
        # Extract thinking content
        think_start = completion_text.find('<think>') + len('<think>')
        think_end = completion_text.find('</think>')
        if think_end > think_start:
            thinking_part = completion_text[think_start:think_end].strip()
        
        # Extract everything after </think> as kernel part
        kernel_part = completion_text[think_end + len('</think>'):].strip()
    else:
        # No thinking tags, treat whole completion as kernel
        kernel_part = completion_text
    
    return thinking_part, kernel_part


def score_completions(
    completions_text: list[str],
    prompt: str,
    spec: dict,
    eval_class: evaluator.TritonKernelEvaluator,
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
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=spec,
        device=device
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data for logging
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Debug output every 50 steps
    if hasattr(args, '_debug_step_counter'):
        args._debug_step_counter += 1
    else:
        args._debug_step_counter = 0
        
    if args._debug_step_counter % 50 == 0:
        print(f"\n{'='*80}")
        print(f"DEBUG Step {args._debug_step_counter}")
        print(f"Prompt: {prompt}")
        print(f"Rewards: {rewards.tolist()}")
        
        # Show thinking and kernel portions for first completion
        thinking, kernel = extract_thinking_and_kernel(completions_text[0])
        
        if thinking:
            print(f"\nThinking (first 100 chars):")
            print(f"'{thinking[:100]}{'...' if len(thinking) > 100 else ''}'")
        else:
            print("\nNo thinking content found")
            
        print(f"\nKernel (first 100 chars):")
        print(f"'{kernel[:100]}{'...' if len(kernel) > 100 else ''}'")
        
        print(f"\nMetrics: compilation_rate={metrics.get('compilation_rate', 0)*100:.1f}%")
        print(f"{'='*80}")

    # Compute advantages using group statistics (GRPO core)
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
        ref_per_token_logps = utils.get_per_token_logps(base_model, prompt_completion_ids, attention_mask, logits_to_keep)

    # Get training model logits
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

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
    eval_class: evaluator.TritonKernelEvaluator,
    device: str,
    round_num: int,
    training_log_dir: str, 
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss for Triton kernel generation."""
    
    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, prompt, device, args
    )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, prompt, spec, eval_class, device, args
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Triton kernels")
    
    # Model and dataset
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model to train")
    parser.add_argument("--dataset_path", type=str, default="data/triton_kernels_dataset_v5.json", help="Path to kernel dataset")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="triton_grpo_output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--save_steps", type=int, default=200, help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=100, help="Evaluation frequency")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Gradient clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Warmup percentage")
    parser.add_argument("--update_ref_model", action="store_true", help="Update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="Reference model update frequency")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, help="Reference model mixup alpha")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=8, help="Number of generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Max completion length")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    # Get args and setup
    args = parse_args()
    utils.seed_everything(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)

    # Load dataset
    print("Loading dataset...")
    train_loader, test_loader = rldataset.get_dataloaders(args.dataset_path)
    print(f"Train examples: {len(train_loader)}, Test examples: {len(test_loader)}")

    # Setup evaluator
    eval_class = evaluator.TritonKernelEvaluator()

    # Setup logging directories
    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Training loop
    print("Starting GRPO training...")
    optimizer.zero_grad()
    train_metrics_total = {}
    
    for round_num in tqdm(range(args.num_train_iters), desc="Training"):
        
        # Evaluate on test set
        if round_num % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            print(f"\nEval at step {round_num}: Compilation rate = {eval_accuracy:.1f}%")
            
            # Save metrics
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

        # Update reference model
        if args.update_ref_model and (round_num + 1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get training example
        prompt, spec = next(train_loader)

        # GRPO loss computation
        total_loss, train_metrics = grpo_loss(
            model, base_model, tokenizer, prompt, spec, eval_class, 
            device, round_num, train_log_dir, args
        )
        
        # Backpropagation
        total_loss.backward()
        scheduler.step()

        # Optimizer step with gradient accumulation
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        
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

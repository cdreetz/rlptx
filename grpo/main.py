"""
Main script for GRPO training on Triton kernel generation
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import GenerationConfig

import llms
import utils
import evaluator
import rldatasets
from kernelbook_adapter import KernelBookAdapter


def eval_on_test_set(
    model,
    tokenizer,
    test_loader,
    eval_class,
    device,
    args,
    round_num
):
    """
    Evaluate model performance on test set.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_loader: DataLoader for test set
        eval_class: Evaluator for computing rewards
        device: Device to run on
        args: Training arguments
        round_num: Current training round number
        
    Returns:
        total_scores: Dictionary of average metrics
        compile_rate: Compilation success rate on test set
    """
    print("Running evaluation on test set...")
    
    # Track metrics across all test examples
    total_scores = defaultdict(float)
    num_examples = 0
    total_compile_rate = 0.0

    # Create log file for this evaluation round
    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        # Run through test set
        for question, answer in tqdm(test_loader, desc="Evaluating on test set"):
            # Generate completions using same function as training
            _, _, _, _, completions_text, _ = generate_completions(
                model, tokenizer, question, device, args
            )
            
            # Score completions using evaluator
            mock_prompts = [[{'content': question}]] * len(completions_text)
            mock_completions = [[{'content': completion}] for completion in completions_text]
            # Make answer array same length as completions
            answers = [answer] * len(completions_text)
            rewards_per_func, metrics = eval_class.compute_rewards(
                prompts=mock_prompts,
                completions=mock_completions, 
                answer=answers,
                device=device
            )
            
            # Track compile_rate and accumulate metrics
            total_compile_rate += metrics['compile_success_rate']
                
            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Response: {completions_text[0]}\n") # Log first completion
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")

    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    compile_rate = total_compile_rate / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'compile_success_rate': compile_rate}, f, indent=4)

    if args.verbose:
        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Compilation Success Rate: {compile_rate:.2f}%")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    return avg_scores, compile_rate


def generate_completions(
    model,
    tokenizer, 
    question,
    device,
    args
):
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        question: The input question/prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """
    # 1. Prepare prompting
    prompt = [
        {'role': 'system', 'content': args.system_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    
    # Repeat for number of chains/generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move tensors to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Set up generation config
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

    # Extract completion ids
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
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
    completions_text,
    question,
    answer,
    eval_class,
    device,
    args
):
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
            'answer': answer
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    mock_prompts = [[{'content': question}]] * len(completions_text)
    mock_completions = [[{'content': completion}] for completion in completions_text]
    answers = [answer] * len(completions_text)
    
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=answers,
        device=device
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data


def compute_loss(
    model,
    base_model, 
    prompt_completion_ids,
    prompt_ids,
    completion_ids,
    attention_mask,
    completion_mask,
    advantages,
    args
):
    """
    Compute the GRPO loss between current and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """
    # Only need the generated tokens' logits
    logits_to_keep = completion_ids.size(1)

    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = utils.get_per_token_logps(base_model, prompt_completion_ids, attention_mask, logits_to_keep)

    # Get training model logits
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # Compute loss with advantages
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
    model,
    base_model,
    tokenizer,
    question,
    answer,
    eval_class,
    device,
    round_num,
    training_log_dir, 
    args
):
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        tokenizer: Tokenizer for the models
        question: Input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """
    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, question, device, args
    )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, eval_class, device, args
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


def prepare_dataset(args):
    """
    Prepare the KernelBook dataset by converting Triton kernels to natural language queries.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the processed dataset
    """
    # If the dataset already exists, skip processing
    if os.path.exists(args.processed_dataset) and not args.reprocess_dataset:
        print(f"Using existing processed dataset: {args.processed_dataset}")
        return args.processed_dataset
    
    # Process the dataset
    print(f"Processing KernelBook dataset to create NL queries...")
    adapter = KernelBookAdapter(
        dataset_path=args.dataset_path,
        output_path=args.processed_dataset
    )
    adapter.process()
    print(f"Dataset processed and saved to {args.processed_dataset}")
    
    return args.processed_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Triton kernels")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="dataset_permissive.json", 
                        help="Path to original KernelBook dataset")
    parser.add_argument("--processed_dataset", type=str, default="kernelbook_nl_queries.json",
                        help="Path to save/load processed dataset")
    parser.add_argument("--reprocess_dataset", action="store_true",
                        help="Force reprocessing of dataset even if processed version exists")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="Name/path of base model")
    parser.add_argument("--evaluator", type=str, default="triton", 
                        help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=20, 
                        help="Number of iterations for evaluation")

    # Optimization hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
                        help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, 
                        help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, 
                        help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_percent", type=float, default=0.18, 
                        help="Percentage of total steps for warmup")
    parser.add_argument("--update_ref_model", action="store_true", 
                        help="Whether to update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, 
                        help="How often to update reference model")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, 
                        help="Alpha parameter for reference model mixup")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, 
                        help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=16, 
                        help="Number of parallel generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=256, 
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, 
                        help="Maximum completion length")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Custom system prompt (defaults to loader's prompt if None)")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, 
                        help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, 
                        help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, 
                        help="Random seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Seed everything 
    utils.seed_everything(args.seed)

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save arguments
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Prepare dataset (process if needed)
    dataset_path = prepare_dataset(args)

    # Load models
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)

    # Load dataset
    train_loader, test_loader = rldatasets.get_dataloaders("kernelbook", dataset_path)
    
    # If no custom system prompt provided, use the loader's
    if args.system_prompt is None:
        args.system_prompt = train_loader.system_prompt
    
    # Create evaluator
    eval_class = evaluator.get_evaluator(args.evaluator)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Setup learning rate scheduler with warmup
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Begin training
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    
    for round_num in tqdm(range(args.num_train_iters), desc="Training Progress"):
        # Evaluate on test set periodically
        if round_num % args.eval_iterations == 0:
            eval_metrics, compile_rate = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Save model checkpoint after evaluation
            if round_num > 0:
                checkpoint_path = os.path.join(model_dir, f'checkpoint_{round_num}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'round_num': round_num,
                    'compile_rate': compile_rate,
                    'args': args_dict
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Update reference model periodically
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data
            print(f"Updated reference model at iteration {round_num+1}")

        # Get next example
        question, answer = next(train_loader)

        # Compute GRPO loss
        total_loss, train_metrics = grpo_loss(
            model, base_model, tokenizer, question, answer, eval_class, 
            device, round_num, train_log_dir, args
        )
        
        # Apply gradient
        total_loss.backward()
        accumulated_loss += total_loss.item()

        # Step scheduler
        scheduler.step()

        # Perform parameter update
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    

        # Log metrics
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'round_num': args.num_train_iters,
        'args': args_dict
    }, final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")

import os
import torch
import argparse
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import SINGLE_AREA_PROMPT, SingleShapeAreaDataLoader
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from evaluator import SingleShapeEvaluator

def load_model(device: str):
    """
    Load the Qwen2.5-VL-7B-Instruct model and processor for image-text tasks.

    Args:
        device (str): Device to load the model onto (e.g., 'cuda', 'cpu').

    Returns:
        tuple: (model, processor)
            model: AutoModelForImageTextToText instance.
            processor: AutoProcessor instance with tokenizer and padding configured.
    """
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    processor.tokenizer.padding_side = "left"
    processor.padding_side = "left"

    model.config.use_cache = False

    return model, processor

def conversation_template(prompt: str, image_path: str):
    """
    Create a conversation template for the Qwen model with an image and prompt.

    Args:
        prompt (str): The user prompt or question.
        image_path (str): Path to the image file to include in the conversation.

    Returns:
        list: Conversation formatted as a list of message dicts for the model.
    """
    return [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. You are an expert image analyst.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ],
        },
    ] 

# Implementation from:
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py 
def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def generate_completions(model, tokenizer, image_path: str, prompt: str, num_completions, temperature: float, max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str, int, torch.Tensor]:
    """
    Generate multiple completions from the model for a given image and prompt.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer/processor for the model.
        image_path (str): Path to the input image.
        prompt (str): The user prompt or question.
        num_completions (int): Number of completions to generate.
        temperature (float): Sampling temperature for generation.
        max_new_tokens (int): Maximum number of new tokens to generate per completion.

    Returns:
        tuple:
            - prompt_completion_ids (torch.Tensor): Generated token IDs (prompt + completion).
            - prompt_ids (torch.Tensor): Token IDs for the prompt.
            - completion_ids (torch.Tensor): Token IDs for the generated completions.
            - attention_mask_full (torch.Tensor): Attention mask for prompt + completion.
            - completions_text (list[str]): Decoded completion strings.
            - prompt (str): The original prompt string.
            - prompt_length (int): Length of the prompt in tokens.
            - completion_tokens_mask (torch.Tensor): Mask for valid completion tokens.
    """
    conversation = conversation_template(prompt, image_path)

    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    images, videos, video_kwarg = process_vision_info(conversation, return_video_kwargs=True)
    inputs = tokenizer(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwarg).to(model.device).to(model.dtype)

    # Repeat the prompt for N completions
    batched_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            batched_inputs[key] = value.repeat(num_completions, *([1] * (value.dim() - 1)))
        else:
            batched_inputs[key] = value
    
    # Save the prompt IDs for filtering later
    input_ids = inputs["input_ids"]
             
    # Generate completions
    prompt_completion_ids = model.generate(
        **batched_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,    
        temperature=temperature,
        pad_token_id=tokenizer.tokenizer.pad_token_id
    )
    
    # Extract the prompt and completion IDs
    prompt_length = input_ids.size(1) 
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:] 
   
    # Identify where end-of-sequence (EOS) tokens appear in each completion
    is_eos = completion_ids == tokenizer.tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=model.device).expand(is_eos.size(0), -1)
    completion_tokens_mask = (sequence_indices <  eos_idx.unsqueeze(1)).int()
  
    # Create full attention mask (ignoring padding tokens) 
    prompt_mask = batched_inputs["attention_mask"]
    attention_mask_full = torch.cat([prompt_mask, completion_tokens_mask], dim=1)
    
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask_full, completions_text, prompt, prompt_length, completion_tokens_mask

def sequence_log_probs(model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt):
    """
    Compute per-token log probabilities for a sequence given a model and input.

    Args:
        model: The model to use for computing log probabilities.
        input_ids (torch.Tensor): Input token IDs (batch, seq_len).
        attention_mask (torch.Tensor): Attention mask for the input (batch, seq_len).
        image_path (str): Path to the input image.
        tokenizer: The tokenizer/processor for the model.
        logits_to_keep (int): Number of logits/tokens to keep from the end of the sequence.
        prompt (str): The user prompt or question.

    Returns:
        torch.Tensor: Log probabilities for each token in the sequence (batch, seq_len_kept).
    """
    # This process is repeated as per `generate_completions`. This is needed to include the pixels of the image in the input context.
    conversation = conversation_template(prompt, image_path)
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, padding_side="left")
    images, videos, video_kwarg = process_vision_info(conversation, return_video_kwargs=True)
    inputs = tokenizer(text=text, images=images, videos=videos, padding=True, return_tensors="pt", padding_side="left", **video_kwarg).to(model.device).to(model.dtype)

    batched_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            batched_inputs[key] = value.repeat(input_ids.size(0), *([1] * (value.dim() - 1)))
        else:
            batched_inputs[key] = value
            
    batched_inputs["input_ids"] = input_ids
    batched_inputs["attention_mask"] = attention_mask
    
    logits = model(**batched_inputs).logits[:, :-1, :]           

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)


def rollout(policy, tokenizer, evaluator, image_path: str, prompt: str, area: float, num_rollouts, temperature: float):
    """
    Generate multiple completions and compute rewards for a given image, prompt, and target area.

    Args:
        policy: The policy model used for generation.
        tokenizer: The tokenizer/processor for the model.
        evaluator: Evaluator object to compute rewards and metrics.
        image_path (str): Path to the input image.
        prompt (str): The user prompt or question.
        area (float): Ground truth area value for reward computation.
        num_rollouts (int): Number of completions to generate.
        temperature (float): Sampling temperature for generation.

    Returns:
        tuple:
            - prompt_completion_ids (torch.Tensor): Generated token IDs (prompt + completion).
            - attention_mask (torch.Tensor): Attention mask for prompt + completion.
            - scalar_prompt_length (int): Length of the prompt in tokens.
            - completion_tokens_mask (torch.Tensor): Mask for valid completion tokens.
            - rewards_all (torch.Tensor): Total reward for each completion.
            - rewards_per_func_tensor (torch.Tensor): Per-function rewards for each completion.
            - completions_text (list[str]): Decoded completion strings.
            - metrics (dict): Aggregated reward metrics.
    """
    prompt_completion_ids, _, _, attention_mask, completions_text, _, scalar_prompt_length, completion_tokens_mask = \
        generate_completions(policy, tokenizer, image_path, prompt, num_rollouts, temperature, args.max_new_tokens)
    
    # Compute rewards for the completions
    rewards_per_func, metrics = evaluator.compute_rewards(completions_text, area)
    # Convert rewards to tensors for easier manipulation
    rewards_tensor = torch.tensor(rewards_per_func, device=policy.device, dtype=torch.float32)
    rewards_per_func_tensor = torch.tensor(rewards_per_func, device=policy.device, dtype=torch.float32)
    
    # Compute the total reward for each completion by summing rewards from all functions
    rewards_all = rewards_tensor.sum(dim=1)
    
    return (
        prompt_completion_ids, 
        attention_mask, 
        scalar_prompt_length, 
        completion_tokens_mask, 
        rewards_all, 
        rewards_per_func_tensor,  # Return tensor instead of list
        completions_text, 
        metrics
    )

def grpo_loss(policy, reference, tokenizer, evaluator, image_path: str, prompt: str, area: float, num_rollouts, args):
    """
    Compute the GRPO (Generalized Reward Policy Optimization) loss for a batch of completions.

    Args:
        policy: The policy model being optimized.
        reference: The reference model for KL regularization.
        tokenizer: The tokenizer/processor for the model.
        evaluator: Evaluator object to compute rewards and metrics.
        image_path (str): Path to the input image.
        prompt (str): The user prompt or question.
        area (float): Ground truth area value for reward computation.
        num_rollouts (int): Number of completions to generate per batch.
        args: Namespace of hyperparameters (must include beta_kl, temperature, etc.).

    Returns:
        tuple:
            - loss (torch.Tensor): The computed GRPO loss (scalar).
            - metrics (dict): Aggregated reward and evaluation metrics for the batch.
    """
    (
        prompt_completion_ids, 
        attention_mask, 
        scalar_prompt_length, 
        completion_tokens_mask, 
        rewards_all, 
        rewards_per_func,
        completions_text, 
        metrics           
    ) = rollout(policy, tokenizer, evaluator, image_path, prompt, area, num_rollouts, args.temperature)
    
    # Log completions with zero rewards
    zero_reward_indices = (rewards_all == 0).nonzero(as_tuple=True)[0]
    # if len(zero_reward_indices) > 0:
    #     print("\n=== Completions with Zero Rewards ===")
    #     for idx in zero_reward_indices:
    #         print(f"Completion {idx}:")
    #         print(completions_text[idx])
    #         print("---")
    #     print("=====================================\n")

    # print(f"Mean relative error: {metrics['mean_rel_error']}")
    # print(f"Mean XML format: {metrics['mean_xml_format']}")
    # print(f"Mean area format: {metrics['mean_area_format']}")
    # print(f"Mean area correctness: {metrics['mean_area_correctness']}")

    logits_to_keep = completion_tokens_mask.size(1)

    # Get per-token log probabilites of the completions for the policy model and reference model
    policy_log_probs = sequence_log_probs(policy, prompt_completion_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt)
    with torch.inference_mode():
        reference_log_probs = sequence_log_probs(reference,  prompt_completion_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt)

    # print(f"rewards_all: {rewards_all}")
    # print(f"rewards_all mean: {rewards_all.mean().item()}, std: {rewards_all.std().item()}")
    
    # Detailed log probabilities for completions
    masked_policy_log_probs = policy_log_probs * completion_tokens_mask
    masked_reference_log_probs = reference_log_probs * completion_tokens_mask
    
    mean_policy_completion_log_probs = (masked_policy_log_probs.sum(dim=1) / completion_tokens_mask.sum(dim=1).clamp(min=1)).mean().item()
    mean_reference_completion_log_probs = (masked_reference_log_probs.sum(dim=1) / completion_tokens_mask.sum(dim=1).clamp(min=1)).mean().item()

    # print(f"policy_log_probs (completions) mean: {mean_policy_completion_log_probs}, policy_log_probs[0] (completions) sum: {(masked_policy_log_probs[0]).sum().item()}")
    # print(f"reference_log_probs (completions) mean: {mean_reference_completion_log_probs}, reference_log_probs[0] (completions) sum: {(masked_reference_log_probs[0]).sum().item()}")

    beta_kl = args.beta_kl     

    kl_divergence_per_token = torch.exp(reference_log_probs - policy_log_probs) - (reference_log_probs - policy_log_probs) - 1
    # print(f"kl_divergence_per_token sum mean: {kl_divergence_per_token.sum(dim=1).mean().item()}")
    # print(f"kl_divergence_per_token mean: {kl_divergence_per_token.mean().item()}")
    
    # Calculate the advantage 
    advantages_per_token = (rewards_all - rewards_all.mean()) / (rewards_all.std() + 1e-8)
    adv_expanded = advantages_per_token.unsqueeze(1) 

    # Only keep the gradients of the policy, and scale them by the advantage
    loss_per_token = torch.exp(policy_log_probs - policy_log_probs.detach()) * adv_expanded
    loss_per_token = -(loss_per_token - beta_kl * kl_divergence_per_token) 
    loss = ((loss_per_token * completion_tokens_mask).sum(dim=1) / completion_tokens_mask.sum(dim=1)).mean()
    
    # print(f"loss: {loss.item()}")
    # Calculate mean KL divergence across all batches
    mean_kl = ((kl_divergence_per_token * completion_tokens_mask).sum(dim=1) / completion_tokens_mask.sum(dim=1).to(torch.float32)).mean()
    # print(f"mean_kl: {mean_kl.item()}")

    return loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Area")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--num_train_iters", type=int, default=3000)
    parser.add_argument("--eval_iterations", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--beta_kl", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--update_ref_model_iter", type=int, default=200)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--rotate", type=bool, default=True, help="Enable random rotation of shapes generated")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set a fixed random seed
    seed = 42
    torch.manual_seed(seed)
    policy, tokenizer = load_model(device)
    torch.manual_seed(seed) 
    reference, _ = load_model(device)
    reference.eval() 
    
    # Create datasets
    train_dataset = SingleShapeAreaDataLoader(dataset_size=args.num_samples, is_train=True, rotate=args.rotate)
    
    # Evaluator
    evaluator = SingleShapeEvaluator()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=args.learning_rate, 
        betas=(args.beta1, args.beta2), 
        weight_decay=args.weight_decay, 
        eps=1e-8
    )
    
    # Linear warmup scheduler
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)

    train_metrics = {}
    pdf_log_round_data = {}

    start_round = 0
    accumlated_loss = 0
    optimizer.zero_grad()
    for round in tqdm(range(start_round, args.num_train_iters), initial=start_round, total=args.num_train_iters, desc="Training Loop"):
        # Update the refernce model periodically:
        # https://github.com/willccbb/verifiers/tree/main?tab=readme-ov-file#grpo-rules-of-thumb
        if round > 0 and round % args.update_ref_model_iter == 0:
            with torch.no_grad():
                for param, ref_param in zip(policy.parameters(), reference.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data
            reference.eval()

        # Training step
        batch = next(train_dataset)
        img_path, area = batch

        loss, metrics = grpo_loss(policy, reference, tokenizer, evaluator, img_path, train_dataset.prompt, area, args.num_rollouts, args)

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        accumlated_loss += loss.item()

        if (round + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Log mean_rel_error to a file for plotting
        # with open('rewards.log', 'a') as f:
        #     f.write(f"Mean relative error: {metrics['mean_rel_error']}\n")
        #     f.write(f"Mean XML format: {metrics['mean_xml_format']}\n")
        #     f.write(f"Mean area format: {metrics['mean_area_format']}\n")
        #     f.write(f"Mean area correctness: {metrics['mean_area_correctness']}\n")

        # Print the rewards for each function
        print(f"Mean XML format: {metrics['mean_xml_format']}\n")
        print(f"Mean area format: {metrics['mean_area_format']}\n")
        print(f"Mean area correctness: {metrics['mean_area_correctness']}\n") 
        
        
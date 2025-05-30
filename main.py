import os
import torch
import argparse
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import SINGLE_AREA_PROMPT, SingleShapeAreaDataLoader
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from evaluator import SingleShapeEvaluator

def load_model(device: str):
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


def generate_completions(model, tokenizer, image_path: str, prompt: str, num_completions, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str, int, torch.Tensor]:
    # System prompt and user prompt
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
        max_new_tokens=512,
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
    conversation = conversation_template(prompt, image_path)
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, padding_side="left")

    images, videos, video_kwarg = process_vision_info(conversation, return_video_kwargs=True)

    inputs = tokenizer(text=text, images=images, videos=videos, padding=True, return_tensors="pt", padding_side="left", **video_kwarg).to(model.device).to(model.dtype)

    batched = {k: (v.repeat(input_ids.size(0), *([1]*(v.dim()-1))) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    batched["input_ids"] = input_ids
    batched["attention_mask"] = attention_mask

    logits = model(**batched).logits[:, :-1, :]           
    logits = logits[:, -logits_to_keep:, :]               

    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, -logits_to_keep:].unsqueeze(-1)
    return log_probs.gather(-1, targets).squeeze(-1)


def rollout(policy, tokenizer, evaluator, image_path: str, prompt: str, area: float, num_rollouts, temperature: float):
    prompt_completion_ids, _, _, attention_mask, completions_text, _, scalar_prompt_length, completion_tokens_mask = \
        generate_completions(policy, tokenizer, image_path, prompt, num_rollouts, temperature)
    
    # Compute rewards for the completions
    rewards_per_func, metrics = evaluator.compute_rewards(completions_text, area)
    # Convert rewards to a tensor for easier manipulation
    rewards_tensor = torch.tensor(rewards_per_func, device=policy.device, dtype=torch.float32)
    
    # Compute the total reward for each completion by summing rewards from all functions
    rewards_all = rewards_tensor.sum(dim=1)
    
    return (
        prompt_completion_ids, 
        attention_mask, 
        scalar_prompt_length, 
        completion_tokens_mask, 
        rewards_all, 
        rewards_per_func,
        completions_text, 
        metrics
    )

def grpo_loss(policy, reference, tokenizer, evaluator, image_path: str, prompt: str, area: float, num_rollouts, args):
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
    
    # print(f"Completions text: {completions_text}")

    logits_to_keep = completion_tokens_mask.size(1)
    # Get per-token log probabilites of the completions for the policy model and reference model
    policy_log_probs = sequence_log_probs(policy, prompt_completion_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt)
    with torch.inference_mode(): # Ensure no gradients for reference model operations
        reference_log_probs = sequence_log_probs(reference,  prompt_completion_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt)

    # print(f"\n--- GRPO Loss Debug ---")
    print(f"rewards_all: {rewards_all}")
    print(f"rewards_all mean: {rewards_all.mean().item()}, std: {rewards_all.std().item()}")
    
    print(f"policy_log_probs mean: {policy_log_probs.mean().item()}, policy_log_probs[0] sum: {policy_log_probs[0].sum().item()}")
    print(f"reference_log_probs mean: {reference_log_probs.mean().item()}, reference_log_probs[0] sum: {reference_log_probs[0].sum().item()}")
    # # --- GRPO Loss Calculation ---
    ppo_clip_param = args.ppo_clip_param  # Epsilon for PPO clipping
    beta_kl = args.beta_kl       # Coefficient for KL penalty (Increased from 0.01)


    kl_divergence_per_token = torch.exp(reference_log_probs - policy_log_probs) - (reference_log_probs - policy_log_probs) - 1
    print(f"kl_divergence_per_token sum mean: {kl_divergence_per_token.sum(dim=1).mean().item()}")
    print(f"kl_divergence_per_token mean: {kl_divergence_per_token.mean().item()}")
    
    # # Calculate the advantage 
    advantages_per_token = (rewards_all - rewards_all.mean()) / (rewards_all.std() + 1e-8)
    adv_expanded = advantages_per_token.unsqueeze(1) 

    ### NEW
    loss_per_token = torch.exp(reference_log_probs - reference_log_probs.detach()) * adv_expanded
    loss_per_token = -(loss_per_token - beta_kl * kl_divergence_per_token) 
    loss = ((loss_per_token * completion_tokens_mask).sum(dim=1) / completion_tokens_mask.sum(dim=1)).mean()
    
    print(f"loss: {loss.item()}")
    return loss
    ### OLD
    # print(f"advantages_per_token: {advantages_per_token}")
    # print(f"advantages_per_token mean: {advantages_per_token.mean().item()}, std: {advantages_per_token.std().item()}")

    # # # Policy ratio 
    # log_ratio = policy_log_probs - reference_log_probs 
    # ratio = torch.exp(log_ratio)
    # print(f"log_ratio mean: {log_ratio.mean().item()}")
    # print(f"ratio mean: {ratio.mean().item()}, ratio max: {ratio.max().item()}, ratio min: {ratio.min().item()}")

    # # # PPO Clipped Surrogate Objective per token 
    # surr1 = ratio * adv_expanded
    # surr2 = torch.clamp(ratio, 1.0 - ppo_clip_param, 1.0 + ppo_clip_param) * adv_expanded
    # masked_L_clip = torch.min(surr1, surr2) * completion_tokens_mask
    # print(f"surr1 mean: {surr1.mean().item()}")
    # print(f"surr2 mean: {surr2.mean().item()}")
    # print(f"masked_L_clip sum mean: {masked_L_clip.sum(dim=1).mean().item()}")
    
    # # # Policy Gradient Loss 
    # pg_loss = -masked_L_clip.sum(dim=1).mean()
    # print(f"pg_loss: {pg_loss.item()}")
    
    # # # KL Loss 
    # # # Ensure kl_divergence is summed only over valid tokens using completion_tokens_mask
    # masked_kl_divergence = kl_divergence_per_token * completion_tokens_mask
    # mean_kl_div_sum_per_seq = masked_kl_divergence.sum(dim=1).mean()
    # kl_penalty = beta_kl * mean_kl_div_sum_per_seq
    # print(f"mean_kl_div_sum_per_seq (masked): {mean_kl_div_sum_per_seq.item()}")
    # print(f"kl_penalty: {kl_penalty.item()}")
    
    # final_loss = pg_loss + kl_penalty
    # print(f"Final loss (pg_loss + kl_penalty): {final_loss.item()}")
    # print(f"--- End GRPO Loss Debug ---\n")

    # return final_loss, completions_text, rewards_all, rewards_per_func, metrics, advantages_per_token 


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Area")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_iters", type=int, default=1000)
    parser.add_argument("--eval_iterations", type=int, default=100)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_clip_param", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
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
    
    
    
    # # TODO:
    # # Initialise 
    # # - output dir
    # # - checkpoint dir
    # # - eval dir (logs, pdf, json)
    # # - train dir (logs, pdf, json)
    
    # Create datasets
    train_dataset = SingleShapeAreaDataLoader(dataset_size=args.num_samples, is_train=True)
    test_dataset = SingleShapeAreaDataLoader(dataset_size=args.num_samples, is_train=False)
    
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
    

    train_metrics = {}
    pdf_log_round_data = {}

    start_round = 0
    accumlated_loss = 0
    optimizer.zero_grad()
    for round in tqdm(range(start_round, args.num_train_iters), initial=start_round, total=args.num_train_iters, desc="Training Loop"):
        # Run step periodically
        # if round % args.eval_iterations == 0:
        #    # Do eval on test set

        # Reference model update why?

        # Training step
        batch = next(train_dataset)
        img_path, area = batch

        # GRPO loss
        # loss, completions_text, rewards_all, rewards_per_func, metrics, advantages_per_token = \
        #     grpo_loss(policy, reference, tokenizer, evaluator, img_path, train_dataset.prompt, area, args.num_rollouts, args)

        loss = grpo_loss(policy, reference, tokenizer, evaluator, img_path, train_dataset.prompt, area, args.num_rollouts, args)

        # # Backprop
        loss.backward()
        accumlated_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # # Print metrics
        # print(f"Mean reward: {metrics['mean_reward']}")
        # print(f"Mean area correctness: {metrics['mean_area_correctness']}")
        # print(f"Mean area format: {metrics['mean_area_format']}")
        # print(f"Mean XML format: {metrics['mean_xml_format']}")
        # print(f"Loss: {loss.item()}")
        

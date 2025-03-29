# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#u
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import traceback
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration
import PIL
import numpy as np
from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from sklearn.metrics import f1_score

import wandb

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def calculate_reward(gt_shifts, pred_shifts, extra_penalty=0.5):
    """
    Calculate a positive reward (0 to K, K = number of GT shifts).
    - Penalizes extra predictions (not matching any GT shift).
    - Reward is clamped between 0 and K.
    
    Args:
        gt_shifts (list): Ground truth shifts, e.g., [('left', 5), ('right', 3)]
        pred_shifts (list): Predicted shifts, same format.
        extra_penalty (float): Penalty per extra prediction (default: 0.5).
    
    Returns:
        float: Reward in [0, K].
    """
    K = len(gt_shifts)
    total_reward = 0
    matched_pred_indices = set()

    # Step 1: Reward for correct matches (direction + magnitude)
    for gt_side, gt_pixels in gt_shifts:
        best_match_score = 0

        for j, (pred_side, pred_pixels) in enumerate(pred_shifts):
            if j in matched_pred_indices:
                continue  # Skip already matched predictions

            if gt_side == pred_side:
                # Normalized magnitude error (0=perfect, 1=worst)
                error = abs(gt_pixels - pred_pixels) / (gt_pixels + 1e-6)  # Avoid division by zero
                score = max(0, 1 - error)  # Clamped [0, 1]
                if score > best_match_score:
                    best_match_score = score
                    best_match_index = j

        total_reward += best_match_score
        if best_match_score > 0:
            matched_pred_indices.add(best_match_index)  # Mark as matched

    # Step 2: Penalize extra predictions (not matched to any GT)
    extra_predictions = len(pred_shifts) - len(matched_pred_indices)
    total_reward -= extra_penalty * extra_predictions

    # Clamp reward between 0 and K
    total_reward = max(0, min(K, total_reward))
    
    return total_reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                gt_shifts = re.findall(r'\s*[{\[]\s*(?:[<\(]\s*([^,]+?)\s*,\s*(\d+)\s*[>\)]\s*(?:,\s*)?)+\s*[}\]]\s*', ground_truth)
                pred_shifts = re.findall(r'\s*[{\[]\s*(?:[<\(]\s*([^,]+?)\s*,\s*(\d+)\s*[>\)]\s*(?:,\s*)?)+\s*[}\]]\s*', student_answer)
                if len(gt_shifts)>=1 and len(pred_shifts)>=1:
                    reward = calculate_reward(gt_shifts, pred_shifts)
                else:
                    # print('No shift operations found in the solution or GT answer.')
                    print(f'GT shifts: {ground_truth}')
                    print(f'Pred shifts: {student_answer}')
                    reward = 0.0
                    
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    Flexible Brackets

    Works with {} or [] for the outer list.
    Works with <> or () for individual shifts.
    Whitespace-Agnostic
    Handles {<left,5>} or [ ( up , 10 ) ].
    Captures All Shifts
    Extracts each (side, pixels) pair regardless of bracket style.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<answer>\s*[{\[]\s*(?:[<\(]\s*([^,]+?)\s*,\s*(\d+)\s*[>\)]\s*(?:,\s*)?)+\s*[}\]]\s*</answer>"
    
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    data_root = '/home/lix0i/Xiang/RS/R1-V/data/RL_DOTA_train/images/'
    train_dataset = load_from_disk('/home/lix0i/Xiang/RS/R1-V/data/RL_DOTA_train/DOTA_train')
    test_dataset = load_from_disk('/home/lix0i/Xiang/RS/R1-V/data/RL_DOTA_val/DOTA_val')
    dataset = DatasetDict({
        'train': train_dataset,
        # 'test': test_dataset,
    })

    QUESTION_TEMPLATE = "In the image, there's an marked red bounding box indicates the rough location of an object. Help me shift the left/right/bottom/right side of the bounding box to make it tightly enclose the object. Output the thinking process in <think> </think>, and a list of side shifts in <answer> </answer>. \
        The answer follows the format of {<shift side, shift pixels>}, \
        where {<shift side, shift pixels>} is a list of side shifts to move the marked red bounding box to enclose the object. An example answer could be {<bottom, 2>, <right, 10>}. \
        Each side should be chosen from [left, right, top, bottom], shift pixels ranges from -50 to 50 pixels."
    
    def make_conversation_image(example):
        img_path = os.path.basename(example["crop_filename"]).replace("clean", "shifted")
        image_path = os.path.join(data_root, img_path)
        img = PIL.Image.open(image_path)
        shift_operations = example["shift_operations"]
        solution = "<answer> {"
        for ii,shift_op in enumerate(shift_operations):
            side, shift_px = shift_op
            shift_px = int(np.round(float(shift_px)))
            if ii == len(shift_operations) - 1:
                solution += f"<{side}, {-shift_px}>"
            else:
                solution += f"<{side}, {-shift_px}>,"
        solution += "}</answer>"
        
        return {
            "image": img, 
            "solution": solution,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE}, # .format(Question=example["problem"])
                    ],
                },
            ],
        }
    # import pudb; pudb.set_trace()
    dataset = dataset.map(make_conversation_image)

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    
    # Log in as a new user
    wandb.login(key="631c24185e91c6025231a245e160eb569d6e630c")
    
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    run_name = f"{training_args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="RFT", name=run_name)
    
    main(script_args, training_args, model_args)

    wandb.finish()

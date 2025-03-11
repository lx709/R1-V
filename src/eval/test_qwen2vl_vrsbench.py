import os  # Added missing import
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import tqdm
from math_verify import parse, verify
import argparse
import pandas as pd
import numpy as np
from torch.multiprocessing import Process, set_start_method, Manager
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()
import re

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 1. get evaluation configuration <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_eval_config():
    parser = argparse.ArgumentParser(description="Inference script for GeoQA evaluation.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the model checkpoint (e.g., qwen2vl model or a fine-tuned model).")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for inference. Reduce if GPU OOM (default: 50).")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save inference result (e.g., JSON file).")
    parser.add_argument("--prompt_path", required=True, type=str, help="Path to the prompts JSONL file for GeoQA evaluation.")
    all_gpu = ",".join(map(str, range(torch.cuda.device_count())))
    parser.add_argument("--gpu_ids", default=all_gpu, help="comma-separated list of GPU IDs to use")
    args = parser.parse_args()
    return args

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 2. load testset <<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def prepare_test_messages(testset_path):
    testset_data = json.load(open(testset_path, "r"))
    if 'vqa' in args.prompt_path:
        QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    elif 'ref' in args.prompt_path:
        QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format. x1,y1,x2,y2 are in the range of 0 to 100."
    else:
        raise ValueError("prompt_path should contain either 'vqa' or 'ref' for VQA or Referring Expression task.")
    
    tested_messages = []
    for item in testset_data[:500]:
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"/ibex/project/c2106/Xiang/VRSBench/Images_val/{item['image_id']}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=item['question'])
                }
            ]
        }]
        tested_messages.append(message)
    return testset_data, tested_messages

def extract_answer(output_str):
    try:
        pattern = r"<answer>\s*(.*?)\s*</answer>"
        match = re.search(pattern, output_str)
        return match.group(1) if match else None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def extract_bbox(output_str):
    """
    Extracts bounding box coordinates from a string in the format `(x1, y1), (x2, y2)`.
    Returns a tuple of tuples ((x1, y1), (x2, y2)) if successful, otherwise None.
    """
    try:
        # Pattern to match coordinates in the format (x1, y1), (x2, y2)
        pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        match = re.search(pattern, output_str)
        
        if match:
            # Extract and convert coordinates to integers
            x1, y1, x2, y2 = map(int, match.groups())
            return ((x1, y1), (x2, y2))  # Return in the desired format
        else:
            return None
    except Exception as e:
        print(f"Error extracting bounding box: {e}")
        return None

import re

def extract_bbox_gt(output_str):
    """
    Extracts bounding box coordinates from a string in the format `{<x1><y1><x2><y2>}`.
    Returns a tuple of integers (x1, y1, x2, y2) if successful, otherwise None.
    """
    try:
        # Pattern to match coordinates in the format {<x1><y1><x2><y2>}
        pattern = r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}"
        match = re.search(pattern, output_str)
        
        if match:
            # Extract and convert coordinates to integers
            x1, y1, x2, y2 = map(int, match.groups())
            return ((x1, y1), (x2, y2))
        else:
            return None
    except Exception as e:
        print(f"Error extracting bounding box: {e}")
        return None
    
def compute_giou(gt_bbox, student_bbox):
    x1_gt, y1_gt = gt_bbox[0]
    x2_gt, y2_gt = gt_bbox[1]
    
    x1_st, y1_st = student_bbox[0]
    x2_st, y2_st = student_bbox[1]

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    x1_c = min(x1_gt, x1_st)
    y1_c = min(y1_gt, y1_st)
    x2_c = max(x2_gt, x2_st)
    y2_c = max(y2_gt, y2_st)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou

    giou_scaled = giou + 1
    return giou_scaled, iou

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 3. use several GPUs to accelerate inference at testset <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def init_model(model_path, gpu_id):
    if 'Qwen2.5' in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{gpu_id}",
            )
    elif 'Qwen2' in model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{gpu_id}",
        )
    else:
        raise ValueError("Model path should contain 'Qwen2VL' or 'Qwen2.5VL' for Qwen2VL or Qwen2.5VL")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    return model, processor

def answer_a_batch_question_qwen(batch_messages, model, processor):
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]        
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text

def infer_on_single_gpu(model_path, device_id, chunk_of_tested_messages, batch_size, results=None):
    model, processor = init_model(model_path, device_id)
    
    responses = []
    batch_messages_list = [chunk_of_tested_messages[start: start + batch_size] 
               for start in range(0, len(chunk_of_tested_messages), batch_size)]

    for batch_messages in tqdm.auto.tqdm(batch_messages_list, desc=f"GPU {device_id} progress", position=device_id, leave=False):
        batch_output_text = answer_a_batch_question_qwen(batch_messages, model, processor)
        responses.extend(batch_output_text)
    
    results[device_id] = responses
    return
        
def multi_gpu_inference(prompts, gpu_ids, model_path, batch_size):
    set_start_method("spawn", force=True)
    manager = Manager()
    gpu_id2result = manager.dict()

    gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    chunk_size = len(prompts) // num_gpus
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(prompts)
        chunk = prompts[start_idx: end_idx]
        process = Process(target=infer_on_single_gpu, args=(model_path, gpu_id, chunk, batch_size, gpu_id2result))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    all_predicts = []
    for gpu_id in gpu_ids:
        all_predicts.extend(gpu_id2result[gpu_id])

    return all_predicts

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 4. compute metrics <<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def compute_metrics(testset_data, all_predicts, tested_messages):  # Added tested_messages as argument
    final_output = []
    correct_number = 0

    for input_example, model_output in zip(testset_data, all_predicts):
        original_output = model_output
        ground_truth = input_example['ground_truth'].lower()
        model_answer = extract_answer(original_output) 

        if model_answer is not None and ground_truth in model_answer:
            correct_number += 1
            is_correct = True
        else:
            is_correct = False
        
        result = input_example.copy()
        try:
            result.update({
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer if model_answer is not None else None,
                'is_correct': is_correct
            })
        except Exception as e:
            print("no answer parsed", e, model_answer)
            result.update({
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': None,
                'is_correct': is_correct
            })

        final_output.append(result)

    accuracy = correct_number / len(tested_messages) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    with open(args.output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_path}")

def compute_metrics_ref(testset_data, all_predicts):
    final_output = []
    
    total_count = len(testset_data)
    thres_list = [0.5, 0.7]
    count = np.zeros(len(thres_list))

    total_iou = 0.0
    
    for input_example, model_output in zip(testset_data, all_predicts):
        gt_bbox = extract_bbox_gt(input_example['ground_truth'])
        pred_bbox = extract_bbox(model_output)
        print('gt_bbox, pred_bbox', gt_bbox, pred_bbox)
        # import pudb; pudb.set_trace()
        
        if gt_bbox and pred_bbox:
            giou, iou = compute_giou(gt_bbox, pred_bbox)
            total_iou += iou
            
            for ii, thres in enumerate(thres_list):
                if iou >= thres:  # Fixed: Replaced iou_score with iou
                    count[ii] += 1
        
        result = input_example.copy()
        result.update({
            'pred_bbox': pred_bbox,
            'iou': iou if gt_bbox and pred_bbox else None,
        })
        
        final_output.append(result)
    
    avg_iou = total_iou / total_count * 100
    print(f"\nAverage IoU Score: {avg_iou:.2f}%")
    
    for ii, thres in enumerate(thres_list):
        print(f'Acc at iou_{thres}:', count[ii] / total_count * 100, flush=True)
    
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    with open(args.output_path, "w") as f:
        json.dump({'average_iou': avg_iou, 'results': final_output}, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    args = get_eval_config()
    testset_data, tested_messages = prepare_test_messages(testset_path=args.prompt_path)
    all_predicts = multi_gpu_inference(tested_messages, args.gpu_ids, args.model_path, args.batch_size)
    if 'vqa' in args.prompt_path:
        compute_metrics(testset_data, all_predicts, tested_messages)  # Pass tested_messages as argument
    elif 'ref' in args.prompt_path:
        compute_metrics_ref(testset_data, all_predicts)
    else:
        raise ValueError("prompt_path should contain either 'vqa' or 'ref' for VQA or Referring Expression task.")
#!/bin/bash
#SBATCH --mem=100G # memory pool for all cores`
#SBATCH --time 4:00:00 # time, specify max time allocation`
#SBATCH --mail-type=ALL # notifications for job done & fail`
#SBATCH --mail-user=xiang.li.1@kaust.edu.sa
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-gpu=10
#SBATCH --job-name=rft
#SBATCH --output=%x-%j.out

source ~/.bashrc
conda init bash
conda activate r1-v

module load cuda/11.7

export PATH=$PATH:/home/lix0i/Xiang/RS/GeoChat/HIP/bin

r1_v_path=/home/lix0i/Xiang/RS/R1-V/
cd ${r1_v_path}

batch_size=4
gpu_ids=0

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench/checkpoint-500
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench/eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_ref/checkpoint-200
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_ref/eval/pred_s200_ref.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_referring.json

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_ref/checkpoint-200
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_ref/eval/pred_s200_ref.json


model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B_vllm/checkpoint-500
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B_vllm//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B/checkpoint-500
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B/checkpoint-200
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B//eval/pred_s200.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json


model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-v2/checkpoint-500
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-v2//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-v3/checkpoint-500
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-v3//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json


model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-SFT/
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-SFT//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json

model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B-SFT/
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B-SFT//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json


model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-SFT+RL/
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2-VL-2B-SFT+RL/eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json


model_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B_vllm-v2/checkpoint-250
output_path=${r1_v_path}/src/r1-v/outputs/VRSBench_Qwen2.5-VL-3B_vllm-v2//eval/pred_s500.json
prompt_path=/ibex/project/c2106/Xiang/VRSBench/VRSBench_EVAL_vqa.json

python src/eval/test_qwen2vl_vrsbench.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}


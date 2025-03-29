#!/bin/bash
#SBATCH --mem=200G # memory pool for all cores`
#SBATCH --time 4:00:00 # time, specify max time allocation`
#SBATCH --mail-type=ALL # notifications for job done & fail`
#SBATCH --mail-user=xiang.li.1@kaust.edu.sa
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-gpu=10
#SBATCH --job-name=rft
#SBATCH --output=%x-%j.out

source ~/.bashrc
conda init bash
conda activate r1-v

module load cuda/11.7

export PATH=$PATH:/home/lix0i/Xiang/RS/GeoChat/HIP/bin

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_3b_vrsbench_vllm.txt"

cd /home/lix0i/Xiang/RS/R1-V/src/r1-v

QWEN_PATH=/ibex/project/c2106/Xiang/Qwen2.5-VL-3B-Instruct/
HF_DATASET=/ibex/project/c2106/Xiang/VRSBench/RL_data/VRSBench_RL_train_vqa_2k/
OUTPUT_DIR=outputs/VRSBench_Qwen2.5-VL-3B_vllm-v2
RUN_NAME=VRSBench_Qwen2.5-VL-3B_vllm-v2
DS_CONFIG=local_scripts/zero1_no_optimizer.json  # Note that other zero setting would meet bugs related to vllm at current stage.

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2" torchrun \
    --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 4096 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model true \
    --report_to wandb \
    --temperature 1.0 \
    --num_generations 4 \
    --vllm_device "cuda:2" \
    --vllm_gpu_memory_utilization 0.8 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
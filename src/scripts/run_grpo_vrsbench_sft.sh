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

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_vrsbench-v2.txt"

cd /home/lix0i/Xiang/RS/R1-V/src/r1-v


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/zero3.yaml \
src/open_r1/sft.py --config configs/qwen2vl_sft_config.yaml 


# QWEN_PATH=/ibex/project/c2106/Xiang/Qwen2-VL-2B-Instruct/
# HF_DATASET=/ibex/project/c2106/Xiang/VRSBench/RL_data/VRSBench_SFT_train_vqa_2k/
# OUTPUT_DIR=outputs/VRSBench_Qwen2-VL-2B-SFT
# RUN_NAME=VRSBench_Qwen2-VL-2B-SFT

# torchrun --nproc_per_node="2" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12346" \
#     src/open_r1/sft.py \
#     --output_dir ${OUTPUT_DIR} \
#     --model_name_or_path ${QWEN_PATH} \
#     --dataset_name ${HF_DATASET} \
#     --deepspeed local_scripts/zero2.json \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --logging_steps 1 \
#     --bf16 \
#     --report_to wandb \
#     --gradient_checkpointing false \
#     --attn_implementation flash_attention_2 \
#     --num_train_epochs 2 \
#     --run_name ${RUN_NAME} \
#     --save_steps 100 \
#     --save_total_limit 1 \
#     --save_only_model true
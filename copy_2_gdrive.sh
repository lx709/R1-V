#!/bin/bash
#SBATCH --mem=100G # memory pool for all cores`
#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=ALL # notifications for job done & fail`
#SBATCH --mail-user=xiang.li.1@kaust.edu.sa
#SBATCH --job-name=copy
#SBATCH --output=%x-%j.out

# cd /home/lix0i/Xiang/
cd /ibex/project/c2106/Xiang/

git clone https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy env_create/ gdrive:/Codes_Backup20250311/env_create --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy Uni3DL/ gdrive:/Codes_Backup20250311/Uni3DL --progress


# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone sync RS/R1-V gdrive:/Codes_Backup20250311/RS/R1-V


# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy X-Decoder/ gdrive:/Codes_Backup20250311/X-Decoder --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy 3DCoMPaT-v2/ gdrive:/Codes_Backup20250311/3DCoMPaT-v2 --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy 3DCoMPaT/ gdrive:/Codes_Backup20250311/3DCoMPaT --progress


# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy MiniGPT-4/ gdrive:/Codes_Backup20250311/MiniGPT-4 --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy MiniGPT-4_finetune/ gdrive:/Codes_Backup20250311/MiniGPT-4_finetune --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone sync minigpt4_spatial/ gdrive:/Codes_Backup20250311/minigpt4_spatial --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy minigpt4_spatial_new/ gdrive:/Codes_Backup20250311/minigpt4_spatial_new --progress

# /home/lix0i/Xiang/rclone-v1.69.1-linux-amd64/rclone copy data/ gdrive:/Codes_Backup20250311/data --progress --exclude "/ShapeNet*/**"

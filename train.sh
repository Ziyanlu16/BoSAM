export CUDA_VISIBLE_DEVICES=0,1,2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12362

WORK_DIR="work_dir"
TASK_NAME="union_train_turbo_lora"
mkdir -p ${WORK_DIR}/${TASK_NAME}

python train_lora.py \
    --task_name ${TASK_NAME} \
    --checkpoint /mnt/risk2/SAM-Med3D/work_dir/union_train_turbo_best_continue3/sam_model_dice_best.pth \
    --work_dir ${WORK_DIR} \
    --num_workers 24 \
    --gpu_ids 0 1 2 \
    --multi_gpu \
    --lr_scheduler coswarm \
    # --pcc \
    --num_epochs 50 \
    --batch_size 3 \
    --accumulation_steps 30 \
    --lr 4e-5 \
# ===== 可配置参数 =====
VAL_RATIO=0.05         # 验证集比例（0=不使用验证集，0.05=5%）
MAX_STEPS=10000        # 最大训练步数
EVAL_STEPS=500         # 每 N 步评估一次
SAVE_STEPS=500         # 每 N 步保存一次
BATCH_SIZE=1           # 每卡 batch size
GRAD_ACCUM=8           # 梯度累积步数
# ========================

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=20213 sft_minicpm_v2.py  \
        --model_name_or_path /data/zyx/model/MiniCPM-2B-sft-bf16-llama-format \
        --bf16 True \
        --output_dir ../output_v2/fetaqa_2d \
        --model_max_length 4096 \
        --use_flash_attn True \
        --data_path /data/zyx/dataset/TableInstruct/data_v3/fetaqa_train_7325.json \
        --val_ratio $VAL_RATIO \
        --low_rank_training False \
        --max_steps $MAX_STEPS  \
        --per_device_train_batch_size $BATCH_SIZE     \
        --gradient_accumulation_steps $GRAD_ACCUM     \
        --evaluation_strategy "steps"     \
        --eval_steps $EVAL_STEPS     \
        --save_strategy "steps"     \
        --save_steps $SAVE_STEPS     \
        --save_total_limit 3     \
        --load_best_model_at_end True     \
        --metric_for_best_model "loss"     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_ratio 0.03     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 10     \
        --deepspeed ../ds_configs/stage2_offload.json \
        --tf32 True

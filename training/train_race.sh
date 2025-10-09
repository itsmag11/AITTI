SEED=666
export NOTE='race-sd21'
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATA_DIR='./PATH_TO_DATA'

echo ${NOTE}
echo ${DATA_DIR}

export WANDB_MODE=offline
export HF_HUB_DISABLE_HEAD_REQUESTS=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

accelerate launch train_aitti.py \
        --report_to="wandb" \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="adjective" \
        --placeholder_token="<race-diverse>" --initializer_token="individual" \
        --resolution=512 \
        --train_batch_size=1 \
        --repeats=15 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=1 \
        --learning_rate=5.0e-04 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --output_dir=$NOTE \
        --validation_prompt="a photo of a <race-diverse> doctor" \
        --validation_steps=1000 \
        --anchor_loss 1000000.0 \
        --train_adaptive_token_mapping \
        --num_transformer_head 8 \
        --num_transformer_block 6 \
        --num_vectors 1 \
        --is_run

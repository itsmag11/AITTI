# Training

Train an inclusive token (the learned token embedding + adaptive token mapping) on the balanced data from [data_generation](../data_generation). Run the commands below from this folder.

```bash
bash train_gender.sh   # or train_race.sh / train_age.sh
```

**Custom training:**
```bash
accelerate launch train_aitti.py \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --train_data_dir "./data/gender_balanced_data.txt" \
    --placeholder_token "<gender-diverse>" \
    --resolution 512 \
    --train_batch_size 1 \
    --repeats 15 \
    --num_train_epochs 1 \
    --learning_rate 5.0e-04 \
    --output_dir "gender-inclusive" \
    --train_adaptive_token_mapping \
    --anchor_loss 1000000.0 \
    --is_run
```

`--train_data_dir` is the comma-separated list described in [DATA_FORMAT.md](../data_generation/DATA_FORMAT.md). The output folder contains `learned_embeds.safetensors` and `adaptive_mapping.safetensors`, which can be loaded for [inference](../inference) with:

```python
pipe.load_aitti(pretrained_model_name_or_path="training/gender-inclusive", token="<gender-diverse>")
```

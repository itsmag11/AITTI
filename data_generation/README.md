# Data Generation

Generate attribute-balanced training images with Stable Diffusion v1.5. Run the commands below from this folder.

**Gender** (48 prompts: 24 professions × {male, female}, 100 images each):
```bash
bash generate_gender_data.sh
```

**Custom prompt:**
```bash
python generate_data.py \
    --prompts "a photo of a male doctor" \
    --seed 666 \
    --run_times 100 \
    --output_dir ./data/images/doctor_male
```

Each kept image passes two checks:
- RetinaFace detects exactly one face (confidence threshold 0.97).
- Zero-shot CLIP on the face crop agrees with the attribute in the prompt.

For race and age data, change the prompts and the CLIP classes in `generate_data.py`.

See [DATA_FORMAT.md](DATA_FORMAT.md) for how to turn the generated images into the training list used by [training](../training).

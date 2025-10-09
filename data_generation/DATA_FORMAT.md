# Data Format Specification

## Training Data Format

The training data should be in CSV format with the following columns:

### Required Columns

1. **profession**: The occupation/profession name (e.g., "doctor", "engineer", "teacher")
2. **attribute**: The bias attribute type (e.g., "gender", "race", "age")
3. **class**: The specific class value (e.g., "male", "female" for gender; "young", "old" for age)
4. **filename**: Full path to the image file

### Example CSV Structure

```csv
profession,attribute,class,filename
doctor,gender,female,/path/to/images/doctor_female_001.jpg
doctor,gender,male,/path/to/images/doctor_male_001.jpg
engineer,gender,female,/path/to/images/engineer_female_001.jpg
engineer,gender,male,/path/to/images/engineer_male_001.jpg
nurse,gender,female,/path/to/images/nurse_female_001.jpg
nurse,gender,male,/path/to/images/nurse_male_001.jpg
```

## Balanced Dataset Requirements

For effective bias mitigation, ensure:

1. **Equal representation**: Same number of samples for each class within a profession
2. **Diverse professions**: Include both stereotypically associated and neutral professions
3. **Quality images**: All images should pass face detection (if applicable)
4. **Consistent format**: All images at same resolution (typically 512x512 for SD 1.5)

## Creating Your Own Dataset

### Step 1: Generate Initial Images

Use the data generation scripts to create images:

```bash
cd data_generation
python generate_data.py \
    --prompts "a photo of a male doctor" \
    --seed 666 \
    --run_times 100 \
    --output_dir ./data/images/doctor_male \
    --checkface
```

### Step 2: Create CSV File

Create a CSV file listing all generated images with their metadata:

```python
import pandas as pd
import os

data = []
for profession in professions:
    for gender in ['male', 'female']:
        image_dir = f'./data/images/{profession}_{gender}'
        for img_file in os.listdir(image_dir):
            data.append({
                'profession': profession,
                'attribute': 'gender',
                'class': gender,
                'filename': os.path.join(image_dir, img_file)
            })

df = pd.DataFrame(data)
df.to_csv('training_data.txt', index=False)
```

### Step 3: Ensure Data Quality

The `generate_data.py` script automatically ensures quality through:

**Built-in Quality Control:**
- **Face Detection**: RetinaFace ensures exactly one face per image (0.97 confidence threshold)
- **CLIP Classification**: Verifies the demographic attribute matches the prompt

This eliminates the need for post-processing and ensures balanced, high-quality training data.

## Supported Attributes

### Gender
- Classes: `male`, `female`
- Token: `<gender-diverse>`

### Race
- Classes: `Caucasian`, `Black`, `Asian`, `Latino`, `Indian`, `Middle Eastern`
- Token: `<race-diverse>`

### Age
- Classes: `young`, `old`
- Token: `<age-diverse>`
`

## Tips

1. **Start small**: Begin with 5-10 professions and 10 samples per class
2. **Verify quality**: Manually inspect a subset of generated images
3. **Monitor balance**: Ensure equal representation to avoid bias amplification
4. **Document metadata**: Keep track of generation parameters for reproducibility


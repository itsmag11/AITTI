# Inference & Evaluation

The released tokens are downloaded automatically from [itsmag11/AITTI](https://huggingface.co/itsmag11/AITTI) (`gender/`, `race/`, `age/`). To keep a local copy:

```bash
hf download itsmag11/AITTI --local-dir checkpoints/AITTI                    # all tokens
hf download itsmag11/AITTI --include "gender/*" --local-dir checkpoints/AITTI  # a single token
```

## Benchmark on 24 professions

Generates 100 images for each of 24 professions and evaluates them; results are saved to `inference/results/`:

```bash
bash inference_gender.sh   # or inference_race.sh / inference_age.sh
```

## Custom inference

Run from this folder:

```bash
python inference_aitti.py \
    --prompt "A photo of a <gender-inclusive> doctor" \
    --profession_name "doctor" \
    --attribute gender \
    --token_name "<gender-inclusive>" \
    --seed 666 \
    --run_times 100 \
    --output_dir "./results/gender/A_photo_of_a_<gender-inclusive>_doctor" \
    --checkface  # only keep generations with exactly one detected face
```

`--textual_inversion_dir` defaults to the Hub repo `itsmag11/AITTI`. Point it to a local directory (e.g. `checkpoints/AITTI` or your own training output) to load weights from disk; for your own tokens, also pass the `--token_name` used during training.

Use prompts of the form `A photo of a <token> <profession>`.

## Evaluation

`evaluation.py` reports three metrics for each result folder (a folder containing `images/`):

- **KL divergence** between the predicted attribute distribution and the uniform distribution (lower = more balanced). The attribute is predicted by zero-shot CLIP on the detected face crop.
- **FID** against FFHQ.
- **CLIP score** between each image and the prompt without the inclusive token.

```bash
python evaluation.py --attribute gender --root_dir ./results/gender
```

`--root_dir` can be a single result folder or a folder of them (one per profession). The prompt for the CLIP score is derived from the folder name (e.g. `A_photo_of_a_<gender-inclusive>_doctor` becomes `A photo of a doctor`), or set it with `--gt_prompt`. Per-folder results are written to `<folder>/evaluation_<attribute>.txt`, and the per-profession metrics and their averages to `<root_dir>/summary_<attribute>.txt`.

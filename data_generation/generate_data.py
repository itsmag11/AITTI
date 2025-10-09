import os
import torch
import argparse
import random
from pytorch_lightning import seed_everything

from diffusers import StableDiffusionPipeline

from facexlib.detection import init_detection_model

import clip
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def crop_face(img, left, top, right, bottom, expansion_factor=0.5):
    # Calculate the expanded bounding box
    width = right - left
    height = bottom - top

    expanded_left = max(0, left - expansion_factor * width)
    expanded_top = max(0, top - expansion_factor * height)
    expanded_right = min(img.width, right + expansion_factor * width)
    expanded_bottom = min(img.height, bottom + expansion_factor * height)

    # Crop the face with the expanded bounding box
    face_crop = img.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))
    return face_crop

def parse_args():
    parser = argparse.ArgumentParser()
    ### model inputs
    parser.add_argument("--prompts", default='high quality photo of a real person male administrative assistant', 
                        type=str, help='input prompt')
    parser.add_argument("--num_inference_steps", type=int, default=25, help="num_inference_steps")

    ### experiment settings
    parser.add_argument("--seed", type=int, default=-1, help="the seed (for reproducible sampling)")
    parser.add_argument("--run_times", type=int, default=1, help="times to run the pipeline = no. rows in output grid")
    parser.add_argument("--num_col", type=int, default=0, help="number of columns in final grid. if 0, len(prompts)")
    parser.add_argument("--checkface", action='store_true', default=True, help="check if face exists in results")

    ### model inputs
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # -----------------------------------------------------------------------------------------------
    # Setting running parameters and output dir
    args = parse_args()
    SEED = args.seed if args.seed > 0 else random.randint(0, 10000)
    seed_everything(SEED)
    PROMPT = args.prompts
    RUN_TIMES = args.run_times
    NUM_ROWS = len(PROMPT) if args.num_col == 0 else args.num_col
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NUM = {
        'male': 0,
        'female': 1
    }
    target_class = CLASS_NUM[PROMPT.split(' ')[5]]
    dir_name = args.output_dir.split('/')[-1]

    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)

    # Check existing images and adjust valid_generation starting point
    images_dir = os.path.join(OUT_DIR, 'images')
    existing_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    existing_count = len(existing_images)
    
    if existing_count >= RUN_TIMES:
        print(f"Found {existing_count} existing images, which is >= required {RUN_TIMES} images. Skipping generation process.")
        exit(0)
    elif existing_count > 0:
        print(f"Found {existing_count} existing images. Will generate remaining images.")

    logger = open(os.path.join(OUT_DIR, 'log.txt'), 'a')
    logger.write(str(args))

    CLASSES_prompts = ['a photo of a male', 
                        'a photo of a female']
                        
    ## PREPARE CLIP CLASSIFIER
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    CLASSES_text = clip.tokenize(CLASSES_prompts).to(device)

    ## PREAPRE CLIP TRANSFORMS
    clip_transforms = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # -----------------------------------------------------------------------------------------------
    # Establish running pipeline
    pipeline_text2image = pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)

    # -----------------------------------------------------------------------------------------------
    # Run text2image
    valid_generation = existing_count
    attempt = 0
    if args.checkface:
        det_net = init_detection_model('retinaface_resnet50', half=True, device=device)

    while valid_generation < RUN_TIMES:
        attempt += 1
        print(f'{PROMPT}: attempt {attempt} | valid: {valid_generation}')
        image = pipeline_text2image(prompt=PROMPT, num_inference_steps=args.num_inference_steps).images[0] ## PIL.Image.Image

        # print(image)
        # raise NotImplementedError
        if args.checkface:
            with torch.no_grad():
                ## x0, y0, x1, y1, confidence_score, five points (x, y)
                face_locations = det_net.detect_faces(image, 0.97)

            if len(face_locations) != 1:
                continue
            else:
                # (top, right, bottom, left) = face_locations[0]
                left, top, right, bottom, conf = face_locations[0][:5]
                ## CROP FACE WITH 0.5 TIMES EXPAND
                cropped_img = crop_face(image, left, top, right, bottom, 0.5)

                cropped_img_transformed = clip_transforms(cropped_img).to(device)

                with torch.no_grad():
                    logits_per_image, _ = clip_model(cropped_img_transformed.unsqueeze(0), CLASSES_text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # (bs, n_text_query)
                    class_num = int(np.argmax(probs, axis=1))

                    if class_num != target_class:
                        continue

        image.save(os.path.join(OUT_DIR, 'images', f'{str(valid_generation).zfill(3)}.jpg'))
        valid_generation += 1

    assert len(os.listdir(os.path.join(OUT_DIR, 'images'))) == args.run_times

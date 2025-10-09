import os
import sys
import torch
import argparse
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

sys.path.append('../pipelines')
from aitti_pipeline import StableDiffusionAdaptiveTokenPipeline, AdaptiveTokenMapping_v1

from facexlib.detection import init_detection_model


def crop_face(img, left, top, right, bottom, expansion_factor=0.5):
    """Crop face region with expanded bounding box"""
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
    parser.add_argument("--prompt", default="A photo of a <gender-diverse> doctor", type=str, help='input prompts')
    parser.add_argument("--profession_name", default="doctor", type=str, help='profession')
    parser.add_argument("--textual_inversion_dir", type=str, default="log200")
    parser.add_argument("--inference_step", type=str, default=None)
    parser.add_argument("--token_name", type=str, default="<gender-diverse>")
    parser.add_argument("--all_token_name", type=str, default="<gender-diverse>")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="num_inference_steps")
    parser.add_argument("--change_step", type=int, default=0, help="change to inclusive prompt")
    parser.add_argument("--num_transformer_head", type=int, default=6, help="number of transformer attention head in adaptive mapping")
    parser.add_argument("--num_transformer_block", type=int, default=4, help="number of transformer block in adaptive mapping")

    ### experiment settings
    parser.add_argument("--seed", type=int, default=666, help="the seed (for reproducible sampling)")
    parser.add_argument("--run_times", type=int, default=1, help="times to run the pipeline = no. rows in output grid")
    parser.add_argument("--num_col", type=int, default=10, help="number of columns in final grid. if 0, len(prompts)")
    parser.add_argument("--checkface", action='store_true', default=False, help="check if face exists in results")

    ### model inputs
    parser.add_argument("--output_dir", type=str, default='./results/debug')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # -----------------------------------------------------------------------------------------------
    # Setting running parameters and output dir
    args = parse_args()
    SEED = args.seed
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    
    PROMPT = args.prompt
    RUN_TIMES = args.run_times
    NUM_ROWS = len(PROMPT) if args.num_col == 0 else args.num_col

    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'cropped'), exist_ok=True)

    if not os.path.exists(os.path.join(OUT_DIR, 'grid.jpg')):

        embed_dim = 1024 if args.sd_model == "stabilityai/stable-diffusion-2-1" else 768
        adaptive_mapping = AdaptiveTokenMapping_v1(embed_dim, embed_dim, embed_dim, 
                                                   num_heads=args.num_transformer_head, 
                                                   num_layers=args.num_transformer_block).to(dtype=torch.float16)
        am_path = 'adaptive_mapping.safetensors' if args.inference_step is None else f'adaptive_mapping-steps-{args.inference_step}.safetensors'
        adaptive_mapping.load_state_dict(torch.load(os.path.join(args.textual_inversion_dir, am_path),
                                                    map_location=torch.device('cpu')), strict=True)
        adaptive_mapping.requires_grad = False

        pipe = StableDiffusionAdaptiveTokenPipeline.from_pretrained(args.sd_model, adaptive_mapping=adaptive_mapping, torch_dtype=torch.float16).to(device)

        print('===== load textual inversion weights =====', flush=True)
        le_path = 'learned_embeds.safetensors' if args.inference_step is None else f'learned_embeds-steps-{args.inference_step}.safetensors'
        pipe.load_textual_inversion(args.textual_inversion_dir, 
                                    weight_name=le_path, 
                                    token=args.token_name)

        imgs = list()
        transform = transforms.Compose([transforms.ToTensor()])
        attempt = 0
        valid_generation = 0
        if args.checkface:
            det_net = init_detection_model('retinaface_resnet50', half=True)

        while valid_generation < RUN_TIMES:
            attempt += 1
            print(f'{PROMPT}: attempt {attempt} | valid: {valid_generation}')
            out = pipe(PROMPT, num_inference_steps=args.num_inference_steps, guidance_scale=7.5, 
                    profession_name=args.profession_name.replace('_', ' '),
                    token_name=args.all_token_name.replace('|', ' '),
                    change_step=args.change_step)
            image = out.images[0]

            if args.checkface:
                with torch.no_grad():
                    ## x0, y0, x1, y1, confidence_score, five points (x, y)
                    face_locations = det_net.detect_faces(image, 0.97)

            if len(face_locations) != 1:
                continue
            
                # Detected exactly one face, crop and save
                left, top, right, bottom, conf = face_locations[0][:5]
                cropped_face = crop_face(image, left, top, right, bottom, 0.5)
                cropped_face.save(os.path.join(OUT_DIR, 'cropped', f'{str(valid_generation).zfill(3)}.jpg'))

            image.save(os.path.join(OUT_DIR, 'images', f'{str(valid_generation).zfill(3)}.jpg'))
            imgs.append(transform(image))
            valid_generation += 1

        # -----------------------------------------------------------------------------------------------
        # Visualize results
        grid = make_grid(imgs, nrow=NUM_ROWS)
        transform_back = transforms.ToPILImage()
        grid_out = transform_back(grid)
        grid_out.save(os.path.join(OUT_DIR, 'grid.jpg'))
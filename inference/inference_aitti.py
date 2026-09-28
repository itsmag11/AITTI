import os
import sys
import torch
import argparse
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipelines'))
from aitti_pipeline import StableDiffusionAdaptiveTokenPipeline, AITTI_REPO_ID

from facexlib.detection import init_detection_model


def parse_args():
    parser = argparse.ArgumentParser()
    ### model inputs
    parser.add_argument("--prompt", default="A photo of a <gender-inclusive> doctor", type=str, help='input prompts')
    parser.add_argument("--profession_name", default="doctor", type=str, help='profession')
    parser.add_argument("--textual_inversion_dir", type=str, default=AITTI_REPO_ID,
                        help="Hugging Face repo id or local directory holding the AITTI weights")
    parser.add_argument("--attribute", type=str, default=None,
                        help="sub-folder of --textual_inversion_dir to load, e.g. gender / race / age")
    parser.add_argument("--inference_step", type=str, default=None)
    parser.add_argument("--token_name", type=str, default="<gender-inclusive>")
    parser.add_argument("--sd_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="num_inference_steps")
    parser.add_argument("--change_step", type=int, default=None,
                        help="change to inclusive prompt after this many steps (defaults to the token's config.json, otherwise 0)")

    ### experiment settings
    parser.add_argument("--seed", type=int, default=666, help="the seed (for reproducible sampling)")
    parser.add_argument("--run_times", type=int, default=1, help="times to run the pipeline = no. rows in output grid")
    parser.add_argument("--num_col", type=int, default=10, help="number of columns in final grid. if 0, len(prompts)")
    parser.add_argument("--checkface", action='store_true', default=False,
                        help="only keep generations with exactly one detected face")

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

    if not os.path.exists(os.path.join(OUT_DIR, 'grid.jpg')):

        pipe = StableDiffusionAdaptiveTokenPipeline.from_pretrained(args.sd_model, torch_dtype=torch.float16).to(device)

        print('===== load AITTI weights =====', flush=True)
        am_path = 'adaptive_mapping.safetensors' if args.inference_step is None else f'adaptive_mapping-steps-{args.inference_step}.safetensors'
        le_path = 'learned_embeds.safetensors' if args.inference_step is None else f'learned_embeds-steps-{args.inference_step}.safetensors'
        pipe.load_aitti(args.attribute,
                        pretrained_model_name_or_path=args.textual_inversion_dir,
                        token=args.token_name,
                        adaptive_mapping_weight_name=am_path,
                        learned_embeds_weight_name=le_path)

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
                    token_name=args.token_name,
                    change_step=args.change_step)
            image = out.images[0]

            if args.checkface:
                with torch.no_grad():
                    ## x0, y0, x1, y1, confidence_score, five points (x, y)
                    face_locations = det_net.detect_faces(image, 0.97)
                if len(face_locations) != 1:
                    continue

            image.save(os.path.join(OUT_DIR, 'images', f'{str(valid_generation).zfill(3)}.jpg'))
            imgs.append(transform(image))
            valid_generation += 1

        # -----------------------------------------------------------------------------------------------
        # Visualize results
        grid = make_grid(imgs, nrow=NUM_ROWS)
        transform_back = transforms.ToPILImage()
        grid_out = transform_back(grid)
        grid_out.save(os.path.join(OUT_DIR, 'grid.jpg'))

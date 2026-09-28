import os
import re
import glob
import argparse

import cv2
import clip
import torch
import numpy as np
import scipy.stats
from PIL import Image
from cleanfid import fid
from facexlib.detection import init_detection_model


ATTRIBUTE_CLASSES = {
    'gender': ['a photo of a female',
               'a photo of a male'],
    'race': ['a photo of a Caucasian person',
             'a photo of a Black person',
             'a photo of an Asian person',
             'a photo of a Middle Eastern person',
             'a photo of a Latino person',
             'a photo of an Indian person'],
    'age': ['a photo of a young person',
            'a photo of an old person'],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate attribute balance (KL), FID and CLIP score of generated images.")
    parser.add_argument("--attribute", type=str, default="gender", choices=list(ATTRIBUTE_CLASSES))
    parser.add_argument("--root_dir", type=str, required=True,
                        help="a result dir with an 'images' folder, or a dir whose sub-dirs each have one")
    parser.add_argument("--gt_prompt", type=str, default=None,
                        help="prompt for CLIP score; defaults to the sub-dir name without the inclusive token, "
                             "e.g. 'A_photo_of_a_<gender-inclusive>_doctor' -> 'A photo of a doctor'")
    return parser.parse_args()


def crop_face(img, left, top, right, bottom, expansion_factor=0.5):
    width, height = right - left, bottom - top
    return img.crop((max(0, left - expansion_factor * width),
                     max(0, top - expansion_factor * height),
                     min(img.width, right + expansion_factor * width),
                     min(img.height, bottom + expansion_factor * height)))


def gt_prompt_from_dirname(dirname):
    return re.sub(r'<[^>]+>\s*', '', dirname.replace('_', ' ')).strip()


@torch.no_grad()
def evaluate_dir(path, classes, gt_prompt, clip_model, preprocess, det_net, device, fout):
    image_paths = sorted(glob.glob(os.path.join(path, 'images', '*.jpg')) + glob.glob(os.path.join(path, 'images', '*.png')))
    class_text = clip.tokenize(classes).to(device)
    gt_text = clip.tokenize([gt_prompt]).to(device)

    fout.write(f'classes: {classes}\nCLIP score prompt: {gt_prompt}\n')
    preds, clip_scores = [], []
    for img_path in image_paths:
        bgr_img = cv2.imread(img_path)
        image = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

        # classify the attribute on the face crop if exactly one face is detected, otherwise on the whole image
        faces = det_net.detect_faces(bgr_img, 0.97)
        face = crop_face(image, *faces[0][:4]) if len(faces) == 1 else image

        logits, _ = clip_model(preprocess(face).unsqueeze(0).to(device), class_text)
        pred = int(logits.argmax(dim=-1))
        gt_logits, _ = clip_model(preprocess(image).unsqueeze(0).to(device), gt_text)
        clip_score = float(gt_logits)

        preds.append(pred)
        clip_scores.append(clip_score)
        fout.write(f'{img_path}: {classes[pred]} | clip_score = {clip_score}\n')

    counts = np.bincount(preds, minlength=len(classes))
    ratios = counts / counts.sum()
    for cls, count, ratio in zip(classes, counts, ratios):
        fout.write(f'{cls}: {count} | ratio: {ratio:.4f}\n')

    metrics = {
        'KL': scipy.stats.entropy(ratios, np.ones(len(classes)) / len(classes)),
        'FID': fid.compute_fid(os.path.join(path, 'images'), dataset_name="FFHQ", dataset_res=1024,
                               dataset_split="trainval70k", device=device),
        'CLIP': float(np.mean(clip_scores)),
    }
    fout.write(' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items()) + '\n')
    return metrics


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ATTRIBUTE_CLASSES[args.attribute]

    if os.path.isdir(os.path.join(args.root_dir, 'images')):
        result_dirs = [args.root_dir]
    else:
        result_dirs = sorted(os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                             if os.path.isdir(os.path.join(args.root_dir, d, 'images')))
    if not result_dirs:
        raise FileNotFoundError(f"No 'images' folder found in {args.root_dir} or its sub-dirs.")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    det_net = init_detection_model('retinaface_resnet50', half=device.type == 'cuda', device=device)

    all_metrics = {}
    for result_dir in result_dirs:
        name = os.path.basename(os.path.normpath(result_dir))
        gt_prompt = args.gt_prompt or gt_prompt_from_dirname(name)
        print(f'Evaluating {name} ...', flush=True)
        with open(os.path.join(result_dir, f'evaluation_{args.attribute}.txt'), 'w') as fout:
            all_metrics[name] = evaluate_dir(result_dir, classes, gt_prompt, clip_model, preprocess, det_net, device, fout)
        print(' | '.join(f'{k}: {v:.4f}' for k, v in all_metrics[name].items()), flush=True)

    summary_path = os.path.join(args.root_dir, f'summary_{args.attribute}.txt')
    with open(summary_path, 'w') as fout:
        fout.write(f'attribute: {args.attribute}\n')
        for name, metrics in all_metrics.items():
            fout.write(f'{name}: ' + ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items()) + '\n')
        averages = {k: np.mean([m[k] for m in all_metrics.values()]) for k in ('KL', 'FID', 'CLIP')}
        summary = 'average ' + ' | '.join(f'{k}: {v:.4f}' for k, v in averages.items())
        fout.write(summary + '\n')
    print(summary)
    print(f'Saved to {summary_path}')

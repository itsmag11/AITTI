import os
import glob
import clip
import torch
import argparse
import scipy.stats
import numpy as np
from PIL import Image
from cleanfid import fid
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from PIL import Image, ImageDraw, ImageFont

import cv2
from facexlib.detection import init_detection_model


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
    desc = "Evaluation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--attribute_to_eval", type=str, default="gender")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--gt_prompt", type=str, default="a photo of a doctor")
    # parser.add_argument('--device_num', type=str, default='0', help='visible device')

    return parser.parse_args()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_img_list(root_dir):
    file_list = glob.glob(os.path.join(root_dir, '*.png'))
    file_list += glob.glob(os.path.join(root_dir, '*.jpg'))
    print('Found {} generated images.'.format(len(file_list)))

    return file_list

def get_text_position(attribute_to_eval, image_size):
    """Determine text position in image based on attribute type"""
    width, height = image_size
    
    if attribute_to_eval == 'gender':
        # Gender: top-left corner
        return (10, 10)
    elif attribute_to_eval == 'race':
        # Race: top-right corner
        return (width - 200, 10)
    elif attribute_to_eval == 'age-5cls':
        # Age: bottom-left corner
        return (10, height - 60)
    elif attribute_to_eval == 'age-2cls':
        # Age: bottom-right corner
        return (width - 200, height - 60)
    else:
        # Default: top-left corner
        return (10, 10)

def crop_faces(path, device):
    """Check if cropped face images exist, if not perform face detection and crop"""
    ## GET IMAGE LIST TO EVALUATE
    eval_list = get_img_list(os.path.join(path, 'images'))
    
    # Check if cropped directory already exists and is not empty
    cropped_dir = os.path.join(path, 'cropped')
    if os.path.exists(cropped_dir):
        cropped_files = os.listdir(cropped_dir)
        if len(cropped_files) > 0:
            print(f"Found {len(cropped_files)} existing cropped images, skipping face detection and cropping")
            return eval_list
    
    print("No existing cropped images found. Starting face detection and cropping...")
    os.makedirs(cropped_dir, exist_ok=True)
    
    ## PREPARE FACE DETECTOR TO FILTER NO-FACE OR MORE-THAN-ONE FACE IMAGES
    det_net = init_detection_model('retinaface_resnet50', half=True, device=device)
    
    for img_path in eval_list:
        ori_img = cv2.imread(img_path)

        ## DETECT FACE
        with torch.no_grad():
            ## x0, y0, x1, y1, confidence_score, five points (x, y)
            face_locations = det_net.detect_faces(ori_img, 0.97)

        rgb_ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        pil_ori_img = Image.fromarray(rgb_ori_img)

        if len(face_locations) != 1:
            ## NOT DOING CROPPING
            cropped_img = pil_ori_img
        else:
            # (top, right, bottom, left) = face_locations[0]
            left, top, right, bottom, conf = face_locations[0][:5]
            ## CROP FACE WITH 0.5 TIMES EXPAND
            cropped_img = crop_face(pil_ori_img, left, top, right, bottom, 0.5)

        name = img_path.split('/')[-1]
        cropped_img.save(os.path.join(cropped_dir, f'{name}'))
    
    print("Face detection and cropping completed")
    return eval_list

def run_classification(path, CLASSES, GT, f, device, attribute_to_eval):
    """Run CLIP classification"""
    ## GET IMAGE LIST TO EVALUATE
    eval_list = get_img_list(os.path.join(path, 'images'))

    ## PREPARE CLIP CLASSIFIER
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    CLASSES_text = clip.tokenize(CLASSES).to(device)
    GT_text = clip.tokenize(GT).to(device)

    os.makedirs(os.path.join(path, 'labeled'), exist_ok=True)

    f.write('----------------------------------------------------------------\n')
    f.write(str(CLASSES) + '\n')

    ## PREAPRE CLIP TRANSFORMS
    transforms = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    img_list = []
    img_pred_cls_list = []
    img_clip_score_list = []
    
    for img_path in eval_list:
        ori_img = cv2.imread(img_path)
        rgb_ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        pil_ori_img = Image.fromarray(rgb_ori_img)

        # Load corresponding cropped image
        name = img_path.split('/')[-1]
        cropped_path = os.path.join(path, 'cropped', f'{name}')
        cropped_img = Image.open(cropped_path)

        ## RUN CLIP CLASSIFIER ON CROPPED IMAGE
        pil_ori_img_transformed = transforms(pil_ori_img).to(device)
        cropped_img_transformed = transforms(cropped_img).to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(cropped_img_transformed.unsqueeze(0), CLASSES_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # (bs, n_text_query)
            class_num = np.argmax(probs, axis=1)
            gt_logits_per_image, _ = clip_model(pil_ori_img_transformed.unsqueeze(0), GT_text)
            clip_score = float(gt_logits_per_image)

            img_list.append(img_path)
            img_pred_cls_list.append(class_num)
            img_clip_score_list.append(clip_score)

        ## SAVE LABELED IMAGE
        label_name = CLASSES[int(class_num)].split(' ')[4]
        labeled_path = os.path.join(path, 'labeled', f'{name}')
        
        # Check if labeled image already exists
        if os.path.exists(labeled_path):
            source_img = Image.open(labeled_path)
            # print(f"Loading existing labeled image: {labeled_path}")
        else:
            source_img = pil_ori_img.copy()
        
        draw = ImageDraw.Draw(source_img)
        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=50)
        
        # Place text in different positions based on attribute_to_eval
        text_position = get_text_position(attribute_to_eval, source_img.size)
        draw.text(text_position, label_name, (255,255,255), font=font)

        source_img.save(labeled_path)
        f.write(f'{img_path}: {class_num} | clip_score = {clip_score}\n')

    if len(img_list) == 0:
        f.write(f'no invalid generation.\n')
        return None
    else:
        num_each_cls_list = []
        for k in range(len(CLASSES)):
            num_each_cls = len(np.where(np.array(img_pred_cls_list) == k)[0])
            num_each_cls_list.append(num_each_cls)
            print("{}: total pred: {} | ratio: {}".format(CLASSES[k], num_each_cls, num_each_cls / len(img_list)))
            f.write("{}: total pred: {} | ratio: {} \n".format(CLASSES[k], num_each_cls, num_each_cls / len(img_list)))
        print(f'invalid generation: {len(eval_list)-len(img_list)}')
        f.write(f'invalid generation: {len(eval_list)-len(img_list)}\n')

        return num_each_cls_list, img_clip_score_list

def eval(path, CLASSES, GT, f, device, attribute_to_eval):
    """Main evaluation function"""
    # Step 1: Face detection and crop
    eval_list = crop_faces(path, device)
    
    # Step 2: Run classification
    return run_classification(path, CLASSES, GT, f, device, attribute_to_eval)


if __name__ == '__main__':

    args = parse_args()

    if args.attribute_to_eval == 'gender':
        CLASSES_prompts = ['a photo of a female', 
                           'a photo of a male']
    elif args.attribute_to_eval == 'race':
        CLASSES_prompts = ['a photo of a Caucasian person', 
                           'a photo of a Black person',
                           'a photo of an Asian person',
                           'a photo of a Middle Eastern person',
                           'a photo of a Latino person',
                           'a photo of an Indian person']
    elif args.attribute_to_eval == 'age':
        CLASSES_prompts = ['a photo of a young person', 
                           'a photo of an old person']
    else:
        raise NotImplementedError('attribute type to evaluate is not supported.')
    length = len(CLASSES_prompts)
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device_}', flush=True)

    GT_prompt = args.gt_prompt
    
    fout = open(os.path.join(args.root_dir, f'evaluation_{args.attribute_to_eval}.txt'), 'a')
    
    # if not os.path.exists(fout):
    # evaluate
    num_each_cls_list, img_clip_score_list = eval(args.root_dir, CLASSES_prompts, GT_prompt, fout, device_, args.attribute_to_eval)

    if num_each_cls_list is not None:
        # get the ratio
        each_cls_ratio = num_each_cls_list/np.sum(num_each_cls_list)

        # compute KL
        uniform_distribution = np.ones(length)/length

        KL1 = np.sum(scipy.special.kl_div(each_cls_ratio, uniform_distribution))
        KL2 = scipy.stats.entropy(each_cls_ratio, uniform_distribution)
        assert round(KL1, 4) == round(KL2, 4)

        print("For Class {}, KL Divergence is {:4f}".format(CLASSES_prompts, KL1))
        fout.write("For Class {}, KL Divergence is {:4f}\n".format(CLASSES_prompts, KL1))

        score = fid.compute_fid(os.path.join(args.root_dir, 'images'), dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")
        print("FID Score is {}".format(score))
        fout.write("FID Score is {}\n".format(score))

        mean_clip = sum(img_clip_score_list) / len(img_clip_score_list)
        print("CLIP Score is {}".format(mean_clip))
        fout.write("CLIP Score is {}".format(mean_clip))

        fout.close()

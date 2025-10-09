import os
import argparse

def parse_args():
    desc = "Evaluation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--attribute_to_eval", type=str, default="gender")
    parser.add_argument("--root_dir", type=str, default=None, 
                        help="root dir with 'image' folder inside for images to be evaluated")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    EVAL_BIAS = args.attribute_to_eval
    DIR = args.root_dir
    DIR_NAME = os.path.basename(DIR)
    PARENT_DIR = os.path.dirname(DIR)

    with open(os.path.join(PARENT_DIR, f'{DIR_NAME}_evaluation_{EVAL_BIAS}.txt'), 'w') as fout:
        fout.write(f'Evaluating attribute: {EVAL_BIAS}\n')
        total_kl, total_fid, total_clip = 0.0, 0.0, 0.0
        total_num = 0
        for subdir in os.listdir(DIR):
            eval_file = os.path.join(DIR, subdir, f'evaluation_{EVAL_BIAS}.txt')
            f = open(eval_file, 'r')
            lines = f.readlines()
            kl = float(lines[-3].strip().split(' ')[-1])
            fid = float(lines[-2].strip().split(' ')[-1])
            clip_t = float(lines[-1].strip().split(' ')[-1])
            fout.write(f'{subdir}: {str(kl)} | {str(fid)} | {str(clip_t)}\n')
            total_kl += kl
            total_fid += fid
            total_clip += clip_t
            total_num += 1
        avg_kl = total_kl / total_num
        avg_fid = total_fid / total_num
        avg_clip = total_clip / total_num
        fout.write(f'average KL Divergency: {str(avg_kl)}\n')
        fout.write(f'average FID: {str(avg_fid)}\n')
        fout.write(f'average CLIP Score: {str(avg_clip)}\n')

        fout.close()
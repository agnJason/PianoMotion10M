import argparse
import os
import torch
from tqdm import tqdm

from datasets.PianoPose import PianoPose
from datasets.show import render_result


DEBUG = 0

def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--data_root", type=str, default='/mnt/ssd/PianoMotion10M')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--train_sec", type=int, default=-1)
    parser.add_argument("--preload", action='store_true')
    parser.add_argument("--is_random", action='store_true')
    parser.add_argument("--adjust", action='store_true')
    parser.add_argument('--up_list', nargs='+', default=[])
    parser.add_argument("--return_beta", action='store_true')
    args = parser.parse_args()
    return args

def draw():
    args = get_args()
    args.preload = False
    args.is_random = False
    args.return_beta = False

    # get dataloader
    valid_dataset = PianoPose(args=args, phase=args.mode)
    os.makedirs('draw_sample', exist_ok=True)

    for test_id in tqdm(range(10)):
        batch, para = valid_dataset.__getitem__(test_id, True)
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key]).cuda().unsqueeze(0)

        audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']
        frame_num = left_pose.shape[1]

        render_result(f'draw_sample/gt_{para["bv"]}_{para["start_frame"]}.mp4',
                      audio[0].cpu().numpy(),
                      right_pose[0, :, 1:52].cpu().numpy(),
                      left_pose[0, :, 1:52].cpu().numpy())

if __name__ == '__main__':
    draw()
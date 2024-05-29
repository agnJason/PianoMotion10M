import argparse
import copy
import json
import os
import numpy as np
import torch
from scipy.signal import savgol_filter

from datasets.PianoPose import PianoPose
from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D
from datasets.show import render_result


DEBUG = 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_path", type=str, default="logs/diffusion_posiguide")
    # dataset
    parser.add_argument("--data_root", type=str, default='/hdd/data3/piano-bilibili')
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default='test')
    args = parser.parse_args()
    return args

def main():
    # get args
    args_exp = get_args()
    args = copy.copy(args_exp)
    with open(args.exp_path + '/args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    args.preload = False
    args.data_root = args_exp.data_root
    args.up_list = ['1467634', '66685747']

    # get dataloader
    valid_dataset = PianoPose(args=args, phase=args_exp.mode)

    args_piano2posi = copy.copy(args)
    with open(args_exp.exp_path + '/args_posi.txt', 'r') as f:
        dic = json.load(f)
        dic['hidden_type'] = args.hidden_type if 'hidden_type' in args.__dict__.keys() else 'audio_f'
        args_piano2posi.__dict__ = dic
    piano2posi = Piano2Posi(args_piano2posi)
    if 'hidden_type' in args.__dict__.keys():
        if args.hidden_type == 'audio_f':
            cond_dim = 768 if 'base' in args_piano2posi.wav2vec_path else 1024
        elif args.hidden_type == 'hidden_f':
            cond_dim = args_piano2posi.feature_dim
        elif args.hidden_type == 'both':
            cond_dim = args_piano2posi.feature_dim + (768 if 'base' in args_piano2posi.wav2vec_path else 1024)
    else:
        cond_dim = 768 if 'base' in args_piano2posi.wav2vec_path else 1024

    unet = Unet1D(
        dim=args.unet_dim,
        dim_mults=(1, 2, 4, 8),
        channels=args.bs_dim,
        remap_noise=args.remap_noise if 'remap_noise' in args.__dict__ else True,
        condition=True,
        guide=args.xyz_guide,
        guide_dim=6 if args.xyz_guide else 0,
        condition_dim=cond_dim,
        encoder_type=args.encoder_type if 'encoder_type' in args.__dict__ else 'none',
        num_layer=args.num_layer if 'num_layer' in args.__dict__ else None
    )

    timesteps = args.timesteps
    model = GaussianDiffusion1D_piano2pose(
        unet,
        piano2posi,
        seq_length=args.train_sec * 30,
        timesteps=timesteps,
        objective='pred_v',
    )

    model_path = sorted([i for i in os.listdir(args_exp.exp_path) if 'ckpt' in i], key=lambda x: int(x.split('-')[1].split('=')[1]))[-1]
    print('load from:', model_path)
    state_dict = torch.load(args_exp.exp_path + '/' + model_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)

    model.to('cuda')

    scale = torch.tensor([1.5, 1.5, 25]).cuda()
    # test_ids = [1, 10, 20, 30, 40, 50] # PLEASE MODIFY TO WITCH IDS TO INFER
    os.makedirs(f'results/{args.experiment_name}', exist_ok=True)
    with torch.no_grad():
        for test_id in range(len(valid_dataset)):
            batch, para = valid_dataset.__getitem__(test_id, True)
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key]).cuda().unsqueeze(0)

            audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']
            frame_num = left_pose.shape[1]

            pose_hat, guide = model.sample(audio, frame_num, right_pose.shape[0])
            pose_hat = pose_hat.permute(0, 2, 1)
            guide = guide.permute(0, 2, 1)
            prediction = pose_hat[0].detach().cpu().numpy() * np.pi
            guide = (guide * scale.repeat(2))[0].cpu().numpy()


            for i in range(prediction.shape[1]):
                prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
            for i in range(guide.shape[1]):
                guide[:, i] = savgol_filter(guide[:, i], 5, 2)
            os.makedirs(f'results/{args.experiment_name}/pred_{para["bv"]}_{para["start_frame"]}', exist_ok=True)
            os.makedirs(f'results/{args.experiment_name}/gt_{para["bv"]}_{para["start_frame"]}', exist_ok=True)
            render_result(f'results/{args.experiment_name}/pred_{para["bv"]}_{para["start_frame"]}',
                          audio[0].cpu().numpy(),
                          np.concatenate([guide[:, :3], prediction[:, :48]], 1),
                          np.concatenate([guide[:, 3:], prediction[:, 48:]], 1), False) # save images
            render_result(f'results/{args.experiment_name}/gt_{para["bv"]}_{para["start_frame"]}',
                          audio[0].cpu().numpy(),
                          right_pose[0, :, 1:52].cpu().numpy(),
                          left_pose[0, :, 1:52].cpu().numpy(), False)

if __name__ == "__main__":
    main()

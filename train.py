import argparse
import json
import os
import numpy as np
import torch
import torch.utils.data as data
from scipy.signal import savgol_filter
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.PianoPose import PianoPose
from models.piano2posi import Piano2Posi
from models.utils import velocity_loss
from datasets.show import render_result
from models.BalancedDataParallel import BalancedDataParallel
DEBUG = 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bs_dim", type=int, default=122, help='number of blendshapes:122')
    parser.add_argument("--feature_dim", type=int, default=832, help='number of feature dim')
    parser.add_argument("--period", type=int, default=30, help='number of period')
    parser.add_argument("--max_seq_len", type=int, default=900, help='max sequence length')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu0_bs", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    # additional
    parser.add_argument("--experiment_name", type=str, default="test")

    # dataset
    parser.add_argument("--data_root", type=str, default='/hdd/data3/piano-bilibili')
    parser.add_argument("--preload", action='store_true')
    parser.add_argument("--tiny", action='store_true')
    parser.add_argument("--adjust", action='store_true')
    parser.add_argument("--is_random", action='store_true')
    parser.add_argument("--return_beta", action='store_true')
    parser.add_argument("--classes_num", type=int, default=88)
    parser.add_argument("--begin_note", type=int, default=21)
    parser.add_argument('--up_list', nargs='+', default=[])

    # model
    parser.add_argument("--continue_train", action='store_true')
    parser.add_argument("--wav2vec_path", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--encoder_type", choices=['transformer', 'mamba'], default='transformer')
    parser.add_argument("--num_layer", type=int, default=6)
    parser.add_argument("--latest_layer", choices=['sigmoid', 'tanh', 'none'], default='tanh')

    # train
    parser.add_argument("--loss_mode", choices=['naive_l2', 'naive_l1'], default='naive_l1')
    parser.add_argument("--weight_rec", type=float, default=1.)
    parser.add_argument("--weight_vel", type=float, default=1.)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--train_sec", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--check_val_every_n_iteration", type=int, default=500)
    parser.add_argument("--save_every_n_iteration", type=int, default=500)
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()
    return args

def main():
    # get args
    args = get_args()
    os.makedirs(args.logdir + '/' + args.experiment_name, exist_ok=True)
    out_dire = args.logdir + '/' + args.experiment_name # + '/' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    os.makedirs(out_dire, exist_ok=True)
    with open(out_dire + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # get dataloader
    train_loader = data.DataLoader(
        PianoPose(args=args, phase='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=8,
        drop_last=True)
    valid_loader = data.DataLoader(
        PianoPose(args=args, phase='valid', tiny=True),
        batch_size=args.valid_batch_size,
        drop_last=True)

    # get model
    model = Piano2Posi(args)

    if args.continue_train:
        model_path = sorted([i for i in os.listdir(out_dire) if 'ckpt' in i], key=lambda x: int(x.split('-')[1].split('=')[1]))[-1]
        print('load from:', model_path)
        state_dict = torch.load(out_dire + '/' + model_path)['state_dict']
        model.load_state_dict(state_dict)
        start_iter = int(model_path.split('-')[1].split('=')[1]) + 1
    else:
        start_iter = 0

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print('GPU number: {}'.format(torch.cuda.device_count()))
    model.to('cuda')
    if torch.cuda.device_count() == 1:
        model = torch.nn.DataParallel(model)
    else:
        model = BalancedDataParallel(args.gpu0_bs, model, dim=0)

    tb_writer = SummaryWriter(out_dire)

    # scale for position
    scale = torch.tensor([1.5, 1.5, 25]).cuda()
    best_score = 999999
    pbar = tqdm(range(start_iter, args.iterations))
    iter_train = iter(train_loader)
    last_checkpoint_path = None

    for iteration in pbar:
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)

        for key in batch.keys():
            batch[key] = batch[key].cuda()

        audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']

        frame_num = left_pose.shape[1]
        right_pose[:, :, 1:4] = right_pose[:, :, 1:4] / scale
        left_pose[:, :, 1:4] = left_pose[:, :, 1:4] / scale
        right_pose[:, :, 4:52] = right_pose[:, :, 4:52] / np.pi
        left_pose[:, :, 4:52] = left_pose[:, :, 4:52] / np.pi
        # forward
        rec_loss, vel_loss = model(audio, frame_num, gt=(right_pose, left_pose))

        loss = args.weight_rec * rec_loss.mean() + args.weight_vel * vel_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        des = 'loss:%.4f' % loss.item()
        tb_writer.add_scalar("train_loss_patches/loss", loss.item(), iteration)
        pbar.set_description(des)
        if (iteration % args.check_val_every_n_iteration == 0 and iteration != 0)  or iteration == args.iterations - 1:
            rec_losss = []
            vel_losss = []
            losss = []
            # eval
            with torch.no_grad():
                for val_idx, batch in tqdm(enumerate(valid_loader)):
                    for key in batch.keys():
                        batch[key] = batch[key].cuda()

                    audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']

                    frame_num = left_pose.shape[1]

                    pose_hat = model(audio, frame_num)

                    if val_idx == 0:
                        prediction = pose_hat[0].detach().cpu().numpy()
                        for i in range(prediction.shape[1]):
                            prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
                        if args.bs_dim == 6:
                            render_result(out_dire + '/result-%d.mp4' % iteration, audio[0].cpu().numpy(),
                                          np.concatenate([prediction[:, :3] * scale.cpu().numpy(), right_pose[0, :, 4:].cpu().numpy()], 1),
                                          np.concatenate([prediction[:, 3:6] * scale.cpu().numpy(), left_pose[0, :, 4:].cpu().numpy()], 1))
                        elif args.bs_dim == 96:
                            render_result(out_dire + '/result-%d.mp4' % iteration, audio[0].cpu().numpy(),
                                          np.concatenate([right_pose[0, :, 1:4].cpu().numpy(), prediction[:, :48] * np.pi], 1),
                                          np.concatenate([left_pose[0, :, 1:4].cpu().numpy(), prediction[:, 48:] * np.pi], 1))
                        elif args.bs_dim == 102:
                            render_result(out_dire + '/result-%d.mp4' % iteration, audio[0].cpu().numpy(),
                                          np.concatenate([prediction[:, :3] * scale.cpu().numpy(), prediction[:, 3:51] * np.pi], 1),
                                          np.concatenate([prediction[:, 51:54] * scale.cpu().numpy(), prediction[:, 54:102] * np.pi], 1))
                        render_result(out_dire + '/gt-%d.mp4' % iteration, audio[0].cpu().numpy(),
                                      # prediction[:, :51], prediction[:, 51:])
                                      right_pose[0, :, 1:52].cpu().numpy(),
                                      left_pose[0, :, 1:52].cpu().numpy())
                    right_hat = pose_hat[:, :, :args.bs_dim // 2]
                    left_hat = pose_hat[:, :, args.bs_dim // 2:]
                    pose_hat_sel = torch.cat([right_hat[right_pose[:, :, 0] == 1], left_hat[left_pose[:, :, 0] == 1]], 0)
                    if args.bs_dim == 6:
                        pose_gt_sel = torch.cat(
                            [right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)[:,
                                      1:4] / scale  # torch.cat([right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)[:, 4:] / np.pi
                    elif args.bs_dim == 96:
                        pose_gt_sel = torch.cat(
                            [right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)[:,
                                      4:52] / np.pi
                    elif args.bs_dim == 102:
                        pose_gt_sel = torch.cat(
                            [right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)
                        pose_gt_sel = torch.cat([pose_gt_sel[:, 1:4] / scale, pose_gt_sel[:, 4:52] / np.pi], 1)

                    # get loss
                    if args.loss_mode.startswith('naive'):
                        if args.loss_mode == 'naive_l1':
                            func = torch.nn.functional.l1_loss
                        elif args.loss_mode == 'naive_l2':
                            func = torch.nn.functional.mse_loss
                        rec_loss = func(pose_gt_sel, pose_hat_sel)
                        if args.bs_dim == 6:
                            vel_loss = velocity_loss(
                                torch.cat([right_pose[:, :, 1:4] / scale, left_pose[:, :, 1:4] / scale], 2), pose_hat,
                                func)
                        elif args.bs_dim == 96:
                            vel_loss = velocity_loss(
                                torch.cat([right_pose[:, :, 4:52] / np.pi, left_pose[:, :, 4:52] / np.pi], 2), pose_hat,
                                func)
                        elif args.bs_dim == 102:
                            vel_loss = velocity_loss(torch.cat([right_pose[:, :, 1:4] / scale,
                                                                right_pose[:, :, 4:52] / np.pi,
                                                                left_pose[:, :, 1:4] / scale,
                                                                left_pose[:, :, 4:52] / np.pi], 2), pose_hat, func)
                    loss = args.weight_rec * rec_loss + args.weight_vel * vel_loss

                    rec_losss.append(rec_loss.item())
                    vel_losss.append(vel_loss.item())
                    losss.append(loss.item())
            print('Val Iter %d: loss: %.4f, rec_loss: %.4f, vel_loss: %.4f' % (iteration, np.mean(losss), np.mean(rec_losss), np.mean(vel_losss)))
            tb_writer.add_scalar("eval/rec", np.mean(rec_losss), iteration)
            tb_writer.add_scalar("eval/vel", np.mean(vel_losss), iteration)
            if iteration % args.save_every_n_iteration == 0  or iteration == args.iterations - 1:
                if np.mean(losss) < best_score:
                    best_score = np.mean(losss)
                    if last_checkpoint_path is not None:  ## To save storage
                        os.system(f'rm {last_checkpoint_path}')
                    last_checkpoint_path = out_dire + '/piano2posi-iter=%d-val_loss=%.16f.ckpt' % (iteration, best_score)
                    torch.save({'state_dict': model.module.state_dict(), 'hyper_parameters': args},
                            last_checkpoint_path)


if __name__ == "__main__":
    main()

import math
import random
import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import imageio
import sys
import librosa
from scipy.signal import find_peaks
from tqdm import tqdm
from datasets.utils import read_midi, read_frame_roll

class PianoPose(Dataset):
    def __init__(self, args, phase='train', tiny=False):
        super().__init__()
        data_root = args.data_root
        self.mano_dir = f'{data_root}/annotation'
        self.video_dir = data_root
        self.audio_dir = f'{data_root}/audio'
        self.midi_dir = f'{data_root}/midi'
        self.mode = phase
        if phase == 'test':
            with open(data_root + '/test.txt') as f:
                data_list = f.readlines()
        elif phase == 'valid':
            with open(data_root + '/valid.txt') as f:
                data_list = f.readlines()
        elif phase == 'train':
            with open(data_root + '/train.txt') as f:
                data_list = f.readlines()
        else:
            raise AttributeError
        random.seed(2014)
        # random shuffle
        random.shuffle(data_list)

        self.file_list = []
        self.up_list = args.up_list
        for up_video_name in data_list:
            up, video_name = up_video_name.strip().split(' ')
            if len(self.up_list) == 0 or (len(self.up_list) > 0 and up in self.up_list):
                json_names = os.listdir(os.path.join(self.mano_dir, up, video_name))
                for json_name in json_names:
                    self.file_list.append([up, video_name, os.path.splitext(json_name)[0]])
        if tiny:
            self.file_list = self.file_list[:50]
        print('data Nums:', len(self.file_list))
        self.train_sec = args.train_sec
        self.fps = 30
        self.preload = args.preload
        self.is_random = args.is_random
        self.return_beta = args.return_beta
        self.adjust = args.adjust

        if self.preload:
            # preload dataset to memory to spped up
            self.audio_array_list = []
            self.left_pose_array_list = []
            self.right_pose_array_list = []
            self.adjust_list = []
            self.fps_list = []
            self.midi_dicts = []
            for file in tqdm(self.file_list):
                up, video_name, seq_name = file
                mano_json = os.path.join(self.mano_dir, up, video_name, seq_name + '.json')
                audio_path = os.path.join(self.audio_dir, up, video_name, seq_name + '.mp3')
                audio_array, sampling_rate = librosa.load(os.path.join(audio_path), sr=16000)
                with open(mano_json, 'r') as f:
                    mano_para = json.load(f)
                self.audio_array_list.append(audio_array)
                self.left_pose_array_list.append(np.array(mano_para['left']).astype(np.float32))
                self.right_pose_array_list.append(np.array(mano_para['right']).astype(np.float32))
                self.fps_list.append(mano_para['fps'])
                self.adjust_list.append(np.array(mano_para['adjust']).astype(np.float32))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item, return_param=False):
        if not self.preload:
            up, video_name, seq_name = self.file_list[item]
            mano_json = os.path.join(self.mano_dir, up, video_name, seq_name + '.json')
            audio_path = os.path.join(self.audio_dir, up, video_name, seq_name + '.mp3')
            audio_array, sampling_rate = librosa.load(os.path.join(audio_path), sr=16000)
            with open(mano_json, 'r') as f:
                mano_para = json.load(f)
            fps = mano_para['fps']

            frame_num = len(mano_para['left'])

            if self.is_random:
                # select random begin frame
                train_frame = self.train_sec * fps
                random_end = min(frame_num, math.floor((audio_array.shape[0] / 16000) * fps)) - train_frame  #
                count = 0
                # Keep same dim
                while True:
                    count += 1
                    frame_begin = random.randint(0, random_end)
                    frame_end = frame_begin + train_frame
                    if frame_end <= frame_num:
                        break
                    if count > 100:
                        print(up, video_name, seq_name, 'maybe error!', frame_end, frame_num)
                        break

                audio_begin = math.ceil(audio_array.shape[0] * (frame_begin / frame_num))
                audio_end = audio_begin + math.floor(self.train_sec * 16000)
            else:
                # use all frame
                random_end = 0
                train_frame = frame_num
                frame_begin = 0
                frame_end = frame_num
                self.train_sec = frame_num / fps
                audio_begin = 0
                audio_end = audio_array.shape[0]

            if audio_begin < 0 or audio_end > audio_array.shape[0] or frame_end > frame_num or random_end < 0:
                print(up, video_name, seq_name, 'maybe error!', frame_end, frame_num)

            # audio
            audio = audio_array[audio_begin:audio_end]

            # select poses
            data_left = torch.from_numpy(np.array(mano_para['left'][frame_begin:frame_end]).astype(np.float32))
            data_right = torch.from_numpy(np.array(mano_para['right'][frame_begin:frame_end]).astype(np.float32))

            if self.adjust:
                data_left[:, 2:4] = data_left[:, 2:4] + torch.tensor(mano_para['adjust'][1:3])
                data_right[:, 2:4] = data_right[:, 2:4] + torch.tensor(mano_para['adjust'][1:3])
            if fps != self.fps:
                data_left = F.interpolate(data_left.transpose(1,0).unsqueeze(0), int(self.train_sec * self.fps), mode='linear')[0].transpose(1, 0)
                data_right = F.interpolate(data_right.transpose(1, 0).unsqueeze(0), int(self.train_sec * self.fps), mode='linear')[0].transpose(1, 0)

        else:
            audio_array = np.copy(self.audio_array_list[item])
            fps = self.fps_list[item]
            data_right = np.copy(self.right_pose_array_list[item])
            data_left = np.copy(self.left_pose_array_list[item])
            frame_num = data_left.shape[0]

            train_frame = self.train_sec * fps
            random_end = min(frame_num, math.floor((audio_array.shape[0] / 16000) * fps)) - train_frame  #
            count = 0
            while True:
                count += 1
                frame_begin = random.randint(0, random_end)
                frame_end = frame_begin + train_frame
                if frame_end <= frame_num:
                    break
                if count > 100:
                    print(item, 'maybe error!', frame_end, frame_num)
                    break
            audio_begin = math.floor(audio_array.shape[0] * (frame_begin / frame_num))
            audio_end = audio_begin + int(self.train_sec * 16000)
            if audio_begin < 0 or audio_end > audio_array.shape[0] or frame_end > frame_num or random_end < 0:
                print(item, 'maybe error!', frame_end, frame_num)
            audio = audio_array[audio_begin:audio_end]

            data_left = torch.from_numpy(data_left[frame_begin:frame_end])
            data_right = torch.from_numpy(data_right[frame_begin:frame_end])
            if self.adjust:
                data_left[:, 2:4] = data_left[:, 2:4] + torch.tensor([self.adjust_list[item][1:3]])
                data_right[:, 2:4] = data_right[:, 2:4] + torch.tensor([self.adjust_list[item][1:3]])
            if fps != self.fps:
                data_left = F.interpolate(data_left.transpose(1, 0).unsqueeze(0), self.train_sec * self.fps, mode='linear')[0].transpose(1, 0)
                data_right = F.interpolate(data_right.transpose(1, 0).unsqueeze(0), self.train_sec * self.fps, mode='linear')[0].transpose(1, 0)

        if not self.return_beta:
            data_right, data_left = data_right[:, :-10], data_left[:, :-10]

        outs = {'audio': audio, 'right': data_right, 'left': data_left}

        if return_param:
            return outs, mano_para
        return outs

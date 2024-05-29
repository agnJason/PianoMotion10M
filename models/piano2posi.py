import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Processor
from models.wav2vec import Wav2Vec2Model, HubertModel
from models.utils import init_biased_mask, enc_dec_mask, velocity_loss
from mamba_ssm import Mamba

def load_without_some_keys(model, pretrained_dict, dropout_key):
    model_dict = model.state_dict()
    not_match = set()
    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']
    for k, v in model_dict.items():
        if k.split('.')[0] not in dropout_key:
            if 'hubert' in k:
                kk = k.replace('hubert', 'wav2vec2')
            else:
                kk = k
            if list(pretrained_dict.keys())[0].startswith('model'):
                kk = 'model.' + kk
            if v.size() == pretrained_dict[kk].size():
                model_dict[k] = pretrained_dict[kk]
            else:
                not_match.add(k.split('.')[0])
                print(k, ' ', kk)
                print(model_dict[k].shape)
                print(pretrained_dict[kk].shape)
    assert len(not_match) == 0, f"load error at keys: {not_match}"
    model.load_state_dict(model_dict)

        
class Piano2Posi(nn.Module):
    def __init__(self, args):
        super(Piano2Posi, self).__init__()
        self.feature_dim = args.feature_dim
        self.bs_dim = args.bs_dim
        self.device = 'cuda'
        self.loss_mode = args.loss_mode
        self.encoder_type = args.encoder_type
        try:
            self.hidden_type = args.hidden_type # audio_f, hidden_f, both
        except:
            self.hidden_type = 'audio_f'

        self.processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_path)
        if 'hubert' in args.wav2vec_path:
            self.audio_encoder_cont = HubertModel.from_pretrained(args.wav2vec_path)

        else:
            self.audio_encoder_cont = Wav2Vec2Model.from_pretrained(args.wav2vec_path)
        if 'large' in args.wav2vec_path:
            self.audio_feature_map_cont = nn.Linear(1024, self.feature_dim)
        if 'base' in args.wav2vec_path:
            self.audio_feature_map_cont = nn.Linear(768, self.feature_dim)

        self.audio_encoder_cont.feature_extractor._freeze_parameters() # only freeze the feature_extractor layer

        self.max_seq_len = args.max_seq_len


        self.pos_feature_map = nn.Linear(6, 512)
        self.relu = nn.ReLU()
        if self.encoder_type == 'transformer':
            print('!!! Use transformer in position predictor !!!')
            self.biased_mask1 = init_biased_mask(n_head=4, max_seq_len=args.max_seq_len, period=args.period)

            decoder_layer = nn.TransformerDecoderLayer(d_model=self.feature_dim, nhead=4, dim_feedforward=self.feature_dim,
                                                       batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_layer)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=4, batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.num_layer)

        elif self.encoder_type == 'mamba':
            print('!!! Use mamba in position predictor !!!')
            self.mamba_encoder = Mamba(
                                    d_model=self.feature_dim, # Model dimension d_model
                                    d_state=args.num_layer,  # SSM state expansion factor
                                    d_conv=4,    # Local convolution width
                                    expand=2,    # Block expansion factor
                                ).to("cuda")

        if args.latest_layer == "sigmoid": 
            self.bs_map_r = nn.Sequential(*[nn.Linear(self.feature_dim, 128),
                                            nn.Linear(128, self.bs_dim),
                                            nn.Sigmoid()])
        elif args.latest_layer == "tanh":
            self.bs_map_r = nn.Sequential(*[nn.Linear(self.feature_dim, 128),
                                            nn.Linear(128, self.bs_dim),
                                            nn.Tanh()])
        
    def forward(self, audio, frame_num, return_hidden=False, gt=None):
        bs = audio.shape[0]
        # get conent feature
        countent = self.processor(audio, sampling_rate=16000, return_tensors="pt",
                                padding="longest").input_values.to(audio.device)
        hidden_states_cont = self.audio_encoder_cont(countent.squeeze(0), frame_num=frame_num).last_hidden_state

        hidden_states = self.audio_feature_map_cont(hidden_states_cont) # B, N, 512
        if self.hidden_type == 'audio_f':
            return_hid_tensor = hidden_states_cont.clone().detach()
        elif self.hidden_type == 'hidden_f':
            return_hid_tensor = hidden_states.clone().detach()
        elif self.hidden_type == 'both':
            return_hid_tensor = torch.cat([hidden_states_cont, hidden_states], 2).clone().detach()
        else:
            print('Error type of hidden_type')
            raise InterruptedError

        if self.encoder_type == 'transformer':
            ## get mask
            tgt_mask = None
            if audio.shape[0] == 1:
                tgt_mask = self.biased_mask1[:, :hidden_states.shape[1], :hidden_states.shape[1]].clone().detach().to(device=audio.device)
            memory_mask = enc_dec_mask(audio.device, hidden_states.shape[1], hidden_states.shape[1])

            time_queries = hidden_states.clone()

            bs_out = self.transformer_decoder(time_queries, self.transformer_encoder(hidden_states), tgt_mask=tgt_mask, memory_mask=memory_mask)
        elif self.encoder_type == 'mamba':
            bs_out = self.mamba_encoder(hidden_states)
        bs_output = self.bs_map_r(bs_out)

        if gt is not None:
            right_pose, left_pose = gt
            right_hat = bs_output[:, :, :self.bs_dim // 2]
            left_hat = bs_output[:, :, self.bs_dim // 2:]
            pose_hat_sel = torch.cat([right_hat[right_pose[:, :, 0] == 1], left_hat[left_pose[:, :, 0] == 1]], 0)
            if self.bs_dim == 6:
                pose_gt_sel = torch.cat([right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)[
                              :,
                              1:4]  # torch.cat([right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)[:, 4:] / np.pi
            elif self.bs_dim == 96:
                pose_gt_sel = torch.cat([right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)[
                              :, 4:52]
            elif self.bs_dim == 102:
                pose_gt_sel = torch.cat([right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]], 0)
                pose_gt_sel = torch.cat([pose_gt_sel[:, 1:4], pose_gt_sel[:, 4:52]], 1)
                # get loss
            if self.loss_mode.startswith('naive'):
                if self.loss_mode == 'naive_l1':
                    func = torch.nn.functional.l1_loss
                elif self.loss_mode == 'naive_l2':
                    func = torch.nn.functional.mse_loss
                rec_loss = func(pose_gt_sel, pose_hat_sel)
                if self.bs_dim == 6:
                    vel_loss = velocity_loss(
                        torch.cat([right_pose[:, :, 1:4], left_pose[:, :, 1:4]], 2), bs_output, func)
                elif self.bs_dim == 96:
                    vel_loss = velocity_loss(
                        torch.cat([right_pose[:, :, 4:52], left_pose[:, :, 4:52]], 2), bs_output, func)
                elif self.bs_dim == 102:
                    vel_loss = velocity_loss(torch.cat([right_pose[:, :, 1:4] ,
                                                        right_pose[:, :, 4:52],
                                                        left_pose[:, :, 1:4],
                                                        left_pose[:, :, 4:52]], 2), bs_output, func)
            return rec_loss, vel_loss
        if return_hidden:
            return bs_output, return_hid_tensor
        else:
            return bs_output

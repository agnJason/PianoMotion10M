import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import ot   # pip install POT if missing
from sklearn.decomposition import IncrementalPCA
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

import vedo
import time

# https://github.com/zhenglinpan/FI3D/blob/master/eval.ipynb
def fid(real, fake):
    """
    Frechet Inception Distance
    """
    n = real.shape[0]

    mean_real = np.mean(real, axis=0)[None, ...]
    mean_fake = np.mean(fake, axis=0)[None, ...]

    cov_fake_sum = fake.T @ fake - n * mean_fake.T @ mean_fake
    cov_real_sum = real.T @ real - n * mean_real.T @ mean_real

    cov_fake = cov_fake_sum / (n - 1)
    cov_real = cov_real_sum / (n - 1)

    mu1, mu2, sig1, sig2 = mean_fake, mean_real, cov_fake, cov_real

    mu1_tensor = torch.tensor(mu1)
    mu2_tensor = torch.tensor(mu2)
    sig1_tensor = torch.tensor(sig1)
    sig2_tensor = torch.tensor(sig2)

    a = (mu1_tensor - mu2_tensor).square().sum(dim=-1)
    b = sig1_tensor.trace() + sig2_tensor.trace()
    c = torch.linalg.eigvals(sig1_tensor @ sig2_tensor).sqrt().real.sum(dim=-1)

    score = a + b - 2 * c

    return score.item()

def loc_distance(real, fake):
    """
    Compute Euclidean distance between locations
    """
    # n, 3/2
    return np.mean(np.sqrt(((real - fake) ** 2).sum(1)))

# https://github.com/zhenglinpan/FI3D/blob/master/eval.ipynb
def MW2(dist1, dist2, K=13):
    """
    Compute the Wasserstein distance between two distributions, each represented as
    a multivariate Gaussian Mixture Model (GMM).

    Args:
        dist1: the first distribution, shape (N, D)
        dist2: the second distribution, shape (N, D)
        K: int, number of components in the GMM

    Returns:
        float: The Wasserstein-2 distance between the two GMMs.
    """

    # Apply PCA to reduce the dimensionality of the data
    dist1_pca = apply_pca(dist1)
    dist2_pca = apply_pca(dist2)

    # Fit GMMs to the input distributions
    gmm1 = GaussianMixture(n_components=K, random_state=0, max_iter=200).fit(dist1_pca)
    gmm2 = GaussianMixture(n_components=K, random_state=0, max_iter=200).fit(dist2_pca)

    # Compute the pairwise Euclidean distances between the means of the GMM components
    C = ot.dist(gmm1.means_, gmm2.means_, metric='euclidean')

    # print(f'gmm1 weights: {gmm1.weights_}, gmm2 weights: {gmm2.weights_}, gmm1 means: {gmm1.means_}, gmm2 means: {gmm2.means_}')

    # Normalize the cost matrix to prevent numerical instability
    C /= C.max()

    # Compute the optimal transport plan using the Earth Mover's Distance (EMD) algorithm
    gamma = ot.emd(gmm1.weights_, gmm2.weights_, C)

    # Calculate the Wasserstein distance using the transport plan and the cost matrix
    W2 = np.sum(gamma * C)

    return W2

def acceleration(joint):
    """
    the mean per-joint acceleration (per-frame) is calculated to assess the smoothness of the generated motion;
    Args:
        joint1: the first distribution, shape (N, J, 3)

    Returns:
         the mean per-joint acceleration
    """
    movement = joint[:-1] - joint[-1:]
    speed = np.sqrt((movement ** 2).sum(1)) # N-1, J
    acc = speed[:-1] - speed[-1:] # N-2, J
    return acc

def apply_pca(data, n_components=10):
    """
    Apply PCA to reduce the dimensionality of the data.

    Args:
        data: numpy.ndarray, dataset to be transformed.
        n_components: int, number of principal components to retain.

    Returns:
        numpy.ndarray: Transformed dataset with reduced dimensions.
    """

    # pca = PCA(n_components=n_components)
    pca = IncrementalPCA(n_components=n_components, batch_size=16)

    data_reduced = pca.fit_transform(data)

    return data_reduced


def visualize(pose, trans, smpl_model):
    keypoints3d = smpl_model(
        global_orient=torch.from_numpy(pose[:, :3]).float().cuda(),
        hand_pose=torch.from_numpy(pose[:, 3:]).float().cuda(),
        betas=torch.zeros(pose.shape[0], 10).cuda(),
        transl=torch.from_numpy(trans).float().cuda(),
    ).joints.cpu().numpy()  # (seq_len, 24, 3)

    bbox_center = (
                          keypoints3d.reshape(-1, 3).max(axis=0)
                          + keypoints3d.reshape(-1, 3).min(axis=0)
                  ) / 2.0
    bbox_size = (
            keypoints3d.reshape(-1, 3).max(axis=0)
            - keypoints3d.reshape(-1, 3).min(axis=0)
    )
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    vedo.show(world, axes=True, viewup="y", interactive=0)
    for kpts in keypoints3d:
        pts = vedo.Points(kpts).c("red")
        plotter = vedo.show(world, pts)
        if plotter.escaped: break  # if ESC
        time.sleep(0.01)
    vedo.interactive().close()





def mae(target, output, mask):
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore warnings

device = torch.device("cuda:0")


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net

class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        feat_size = 256
        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )

        self.out_net = nn.Sequential(
            # nn.Linear(864, 256),  # for 64 frames
            nn.Linear(3680, 512),  # for 34 frames
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, feat_size),
        )

        self.fc_mu = nn.Linear(feat_size, feat_size)
        self.fc_logvar = nn.Linear(feat_size, feat_size)

    def forward(self, poses, variational_encoding):
        # encode
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderFC(nn.Module):
    def __init__(self, gen_length, pose_dim, use_pre_poses=False):
        super().__init__()
        self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.use_pre_poses = use_pre_poses

        in_size = 32
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(pose_dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            in_size += 32

        self.net = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, gen_length * pose_dim),
        )

    def forward(self, latent_code, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, latent_code), dim=1)
        else:
            feat = latent_code
        output = self.net(feat)
        output = output.view(-1, self.gen_length, self.pose_dim)

        return output

class PoseDecoderGRU(nn.Module):
    def __init__(self, gen_length, pose_dim):
        super().__init__()
        self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.in_size = 32 + 32
        self.hidden_size = 300

        self.pre_pose_net = nn.Sequential(
            nn.Linear(pose_dim * 4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, pose_dim)
        )

    def forward(self, latent_code, pre_poses):
        pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
        feat = torch.cat((pre_pose_feat, latent_code), dim=1)
        feat = feat.unsqueeze(1).repeat(1, self.gen_length, 1)

        output, decoder_hidden = self.gru(feat)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        output = output.view(pre_poses.shape[0], self.gen_length, -1)

        return output


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False):
        super().__init__()
        self.use_pre_poses = use_pre_poses

        feat_size = 256
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 256),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(True),
                nn.Linear(64, 136),
            )
        elif length == 240:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 960),
            )
        else:
            assert False

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)

        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 4, -1)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out


class ContextEncoder(nn.Module):
    def __init__(self, args, n_frames, n_words, word_embed_size, word_embeddings):
        super().__init__()

        # encoders
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings)
        self.audio_encoder = WavEncoder()
        self.gru = nn.GRU(32+32, hidden_size=256, num_layers=2,
                          bidirectional=False, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32)
        )

        self.fc_mu = nn.Linear(32, 32)
        self.fc_logvar = nn.Linear(32, 32)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, in_text, in_spec):
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        text_feat_seq, _ = self.text_encoder(in_text)
        audio_feat_seq = self.audio_encoder(in_spec)

        input = torch.cat((audio_feat_seq, text_feat_seq), dim=2)
        output, _ = self.gru(input)

        last_output = output[:, -1]
        out = self.out(last_output)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        z = reparameterize(mu, logvar)
        return z, mu, logvar

class EmbeddingNet(nn.Module):
    def __init__(self, pose_dim, n_frames):
        super().__init__()
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)

    def forward(self, poses, variational_encoding=False):
        poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
        return poses_feat, pose_mu, pose_logvar

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False


class EmbeddingSpaceEvaluator:
    def __init__(self, embed_net_path = "./checkpoints/gesture_autoencoder_checkpoint_best.bin", device=device):
        # init embed net
        ckpt = torch.load(embed_net_path, map_location=device)
        self.device = device
        n_frames = 240
        self.pose_dim = ckpt['pose_dim']
        self.net = EmbeddingNet(self.pose_dim, n_frames).to(device)
        self.net.load_state_dict(ckpt['gen_dict'])
        self.net.train(False)
        self.net.eval()
        self.net.freeze_pose_nets()
        # storage
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

    def reset(self):
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []


    def push_samples(self, generated_poses, real_poses):
        # convert poses to latent features
        generated_poses = torch.tensor(generated_poses).to(self.device)
        real_poses = torch.tensor(real_poses).to(self.device)
        real_feat, _, _ = self.net(real_poses, variational_encoding=False)
        generated_feat, _, _ = self.net(generated_poses, variational_encoding=False)

        self.real_feat_list.append(real_feat.data.cpu().numpy())
        self.generated_feat_list.append(generated_feat.data.cpu().numpy())

        # reconstruction error
        # recon_err_real = F.l1_loss(real_poses, real_recon).item()
        # recon_err_fake = F.l1_loss(generated_poses, generated_recon).item()
        # self.recon_err_diff.append(recon_err_fake - recon_err_real)

    def get_features_for_viz(self):
        import umap
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        transformed_feats = umap.UMAP().fit_transform(np.vstack((generated_feats, real_feats)))
        n = int(transformed_feats.shape[0] / 2)
        generated_feats = transformed_feats[0:n, :]
        real_feats = transformed_feats[n:, :]

        return real_feats, generated_feats
    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)
        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)

            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10000000000000
            return frechet_dist
        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            #     m = np.max(np.abs(covmean.imag))
            #     raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        # tr_covmean =  np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2))).real.sum(-1)
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def get_diversity_scores(self):
        feat1 = np.vstack(self.generated_feat_list[:500])
        random_idx = torch.randperm(len(self.generated_feat_list))[:500]
        shuffle_list = [self.generated_feat_list[x] for x in random_idx]
        feat2 = np.vstack(shuffle_list)

        feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
        return feat_dist
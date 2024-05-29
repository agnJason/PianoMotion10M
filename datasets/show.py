import argparse
import os

import cv2
from tqdm import tqdm
import imageio
from moviepy.editor import VideoFileClip, AudioFileClip
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
glctx = dr.RasterizeGLContext()

import sys
from datasets.utils import get_normals
from models.mano import build_mano

FX = 37500
RESOLUTION = [1080, 1920]
mano_layer = build_mano()
for key in mano_layer.keys():
    mano_layer[key] = mano_layer[key].cuda()

def render_mesh(glctx, vertices, faces, proj, w2c, resolution, is_right):
    # ertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0)
    faces = torch.from_numpy(faces.astype(np.int32)).int().cuda()
    vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2)
    rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
    proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)

    normals = get_normals(rot_verts[:, :, :3], faces.long()) * (2 * is_right -1)
    rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
    feat, _ = dr.interpolate(normals, rast_out, faces)
    gt_normals = dr.antialias(feat, rast_out, proj_verts, faces)
    gt_normals = F.normalize(gt_normals, p=2, dim=3)
    valid_idx = torch.where(rast_out[:, :, :, 3] > 0)
    gt_normals = gt_normals[valid_idx]

    light_direction = torch.zeros_like(gt_normals)
    light_direction[:, 2] = -1
    reflect = (-light_direction) - 2 * gt_normals * torch.sum(gt_normals * (-light_direction), dim=1, keepdim=True)
    dot = torch.sum(reflect * light_direction, dim=1, keepdim=True)  # n 1
    specular = 0.2 * torch.pow(torch.maximum(dot, torch.zeros_like(dot)), 16)
    color = torch.sum(gt_normals * light_direction, dim=1, keepdim=True) + specular
    color = torch.clamp(color, 0, 1)
    # color = color.squeeze().detach().cpu().numpy()
    mesh_img = torch.zeros_like(rast_out[:, :, :, :3])
    mesh_img[valid_idx] = color
    mesh_img = torch.cat([mesh_img, rast_out[:, :, :, 3:4]>0], 3)[0].cpu().numpy()
    return mesh_img

def render_result(out_path, audio,  pose_right, pose_left, video=True):
    assert pose_right.shape[0] == pose_left.shape[0]
    if video:
        videowriter = imageio.get_writer(os.path.dirname(out_path) + '/tmp.mp4', fps=30)

    px = RESOLUTION[1] / 2.
    py = RESOLUTION[0] / 2.

    proj = np.array([[FX, 0, px, 0],
                     [0, FX, py, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    R = np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    scale_mats = np.eye(4)
    scale_mats[:3, :3] = R
    cam_t = [0, 0, 0]
    scale_mats[:3, 3] = cam_t

    proj[0, 0] = proj[0, 0] / (RESOLUTION[1] / 2.)
    proj[0, 2] = proj[0, 2] / (RESOLUTION[1] / 2.) - 1
    proj[1, 1] = proj[1, 1] / (RESOLUTION[0] / 2.)
    proj[1, 2] = proj[1, 2] / (RESOLUTION[0] / 2.) - 1
    proj[2, 2] = 0.
    proj[2, 3] = -0.1
    proj[3, 2] = 1
    proj[3, 3] = 0.

    w2c = torch.from_numpy(scale_mats).unsqueeze(0).permute(0, 2, 1).float().cuda()
    proj = torch.from_numpy(proj).unsqueeze(0).permute(0, 2, 1).float().cuda()

    for i in tqdm(range(pose_right.shape[0])):
        if video:
            img = np.zeros((RESOLUTION[0], RESOLUTION[1], 3), dtype=np.uint8)
        else:
            img = np.zeros((RESOLUTION[0], RESOLUTION[1], 4), dtype=np.uint8)

        camera_t = pose_right[i, :3]
        global_orient = pose_right[i, 3:6]  # 1, 3
        hand_pose = pose_right[i, 6:51]  # 15, 3
        betas = np.zeros(10, dtype=np.float32)

        output = mano_layer['right'](global_orient=torch.tensor(global_orient).float().unsqueeze(0).cuda(),
                                     hand_pose=torch.tensor(hand_pose).float().unsqueeze(0).cuda(),
                                     betas=torch.tensor(betas).float().unsqueeze(0).cuda(),
                                     transl=torch.tensor(camera_t).float().unsqueeze(0).cuda())
        vertices = output.vertices

        with torch.no_grad():
            render = render_mesh(glctx, vertices, mano_layer['right'].faces, proj, w2c, RESOLUTION, 1)
        if video:
            img[render[:, :, 3] == 1] = (render[:, :, :3][render[:, :, 3] == 1] * 255).astype(np.uint8)
        else:
            img[render[:, :, 3] == 1, -1] = 255
            img[render[:, :, 3] == 1, :3] = (render[:, :, :3][render[:, :, 3] == 1] * 255).astype(np.uint8)

        camera_t = pose_left[i, :3]
        global_orient = pose_left[i, 3:6]  # 1, 3
        hand_pose = pose_left[i, 6:51]  # 15, 3
        betas = np.zeros(10, dtype=np.float32)

        output = mano_layer['right'](global_orient=torch.tensor(global_orient).float().unsqueeze(0).cuda(),
                                     hand_pose=torch.tensor(hand_pose).float().unsqueeze(0).cuda(),
                                     betas=torch.tensor(betas).float().unsqueeze(0).cuda(),
                                     transl=torch.tensor(camera_t).float().unsqueeze(0).cuda())
        vertices = output.vertices
        vertices[:, :, 0] =  -1 * vertices[:, :, 0]

        with torch.no_grad():
            render = render_mesh(glctx, vertices, mano_layer['right'].faces, proj, w2c, RESOLUTION, 0)

        if video:
            img[render[:, :, 3] == 1] = (render[:, :, :3][render[:, :, 3] == 1] * 255).astype(np.uint8)
            videowriter.append_data(img[:, :, ::-1])
        else:
            img[render[:, :, 3] == 1, -1] = 255
            img[render[:, :, 3] == 1, :3] = (render[:, :, :3][render[:, :, 3] == 1] * 255).astype(np.uint8)
            cv2.imwrite(f'{out_path}/frame_{i}.png', img)
    if video:
        videowriter.close()
        video = VideoFileClip(os.path.dirname(out_path) + '/tmp.mp4')
        sf.write(os.path.dirname(out_path) + '/tmp.mp3', audio, 16000)
        video = video.set_audio(AudioFileClip(os.path.dirname(out_path) + '/tmp.mp3'))

        video.write_videofile(out_path, codec='libx264')
        os.remove(os.path.dirname(out_path) + '/tmp.mp3')
        os.remove(os.path.dirname(out_path) + '/tmp.mp4')

def render_result_frame(pose_right, pose_left, frame_id=100, idx_id=0):
    assert pose_right.shape[0] == pose_left.shape[0]
    px = RESOLUTION[1] / 2.
    py = RESOLUTION[0] / 2.

    proj = np.array([[FX, 0, px, 0],
                     [0, FX, py, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    R = np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    scale_mats = np.eye(4)
    scale_mats[:3, :3] = R
    cam_t = [0, 0, 0]
    scale_mats[:3, 3] = cam_t

    proj[0, 0] = proj[0, 0] / (RESOLUTION[1] / 2.)
    proj[0, 2] = proj[0, 2] / (RESOLUTION[1] / 2.) - 1
    proj[1, 1] = proj[1, 1] / (RESOLUTION[0] / 2.)
    proj[1, 2] = proj[1, 2] / (RESOLUTION[0] / 2.) - 1
    proj[2, 2] = 0.
    proj[2, 3] = -0.1
    proj[3, 2] = 1
    proj[3, 3] = 0.

    w2c = torch.from_numpy(scale_mats).unsqueeze(0).permute(0, 2, 1).float().cuda()
    proj = torch.from_numpy(proj).unsqueeze(0).permute(0, 2, 1).float().cuda()


    img = np.zeros((RESOLUTION[0], RESOLUTION[1], 4), dtype=np.uint8)

    camera_t = pose_right[frame_id, :3]
    global_orient = pose_right[frame_id, 3:6]  # 1, 3
    hand_pose = pose_right[frame_id, 6:51]  # 15, 3
    betas = np.zeros(10, dtype=np.float32)

    output = mano_layer['right'](global_orient=torch.tensor(global_orient).float().unsqueeze(0).cuda(),
                                 hand_pose=torch.tensor(hand_pose).float().unsqueeze(0).cuda(),
                                 betas=torch.tensor(betas).float().unsqueeze(0).cuda(),
                                 transl=torch.tensor(camera_t).float().unsqueeze(0).cuda())
    vertices = output.vertices

    with torch.no_grad():
        render = render_mesh(glctx, vertices, mano_layer['right'].faces, proj, w2c, RESOLUTION, 1)
    img[render[:, :, 3] == 1, -1] = 255
    img[render[:, :, 3] == 1, :3] = (render[:, :, :3][render[:, :, 3] == 1] * 255).astype(np.uint8)

    camera_t = pose_left[frame_id, :3]
    global_orient = pose_left[frame_id, 3:6]  # 1, 3
    hand_pose = pose_left[frame_id, 6:51]  # 15, 3
    betas = np.zeros(10, dtype=np.float32)

    output = mano_layer['right'](global_orient=torch.tensor(global_orient).float().unsqueeze(0).cuda(),
                                 hand_pose=torch.tensor(hand_pose).float().unsqueeze(0).cuda(),
                                 betas=torch.tensor(betas).float().unsqueeze(0).cuda(),
                                 transl=torch.tensor(camera_t).float().unsqueeze(0).cuda())
    vertices = output.vertices
    vertices[:, :, 0] =  -1 * vertices[:, :, 0]

    with torch.no_grad():
        render = render_mesh(glctx, vertices, mano_layer['right'].faces, proj, w2c, RESOLUTION, 0)

    img[render[:, :, 3] == 1, -1] = 255
    img[render[:, :, 3] == 1, :3] = (render[:, :, :3][render[:, :, 3] == 1] * 255).astype(np.uint8)
    cv2.imwrite(f'figs/{idx_id}_{frame_id}.png', img)

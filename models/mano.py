import smplx
import torch

def build_mano():
    # load mano
    mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True)}
                  # 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}
    # # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    # if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
    #     print('Fix shapedirs bug of MANO')
    #     mano_layer['left'].shapedirs[:, 0, :] *= -1
    return mano_layer
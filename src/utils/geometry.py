import numpy as np
import torch

def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    """

    if torch.is_tensor(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1)
        ], dim=-2)
    else:
        R = T[..., :3, :3]
        t = T[..., :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((*T.shape[:-2], 1, 4), dtype=T.dtype)
        bottom_row[..., :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=-2)

    return T_inv

def get_pixel(H, W):
    # get 2D pixels (u, v) for image_a in cam_a pixel space
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    # u_a = np.flip(u_a, axis=1)
    # v_a = np.flip(v_a, axis=0)
    pixels_a = np.stack([
        u_a.flatten() + 0.5,
        v_a.flatten() + 0.5,
        np.ones_like(u_a.flatten())
    ], axis=0)

    return pixels_a

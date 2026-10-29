"""Rotate raw [0,1] tensors before normalization."""
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate as tv_rotate


def rotate(x: torch.Tensor, angle: float, interpolation='bilinear', fill=0.0):
    """Counterclockwise rotation, fixed square canvas, no expansion.

    Multiples of 90 use exact permutations. Other angles interpolate with
    constant fill and crop to the input canvas; corner content may be lost.
    """
    if x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation protocol requires square images')
    if not np.isfinite(angle):
        raise ValueError('Angle must be finite')
    if float(angle) % 90 == 0:
        return torch.rot90(x, int(angle / 90) % 4, (-2, -1))
    return tv_rotate(x, float(angle), interpolation=InterpolationMode(interpolation),
                     expand=False, fill=float(fill))


def sample_angle(policy, seed: int, epoch: int, original_id: int):
    """Stateless RNG independent of worker scheduling; balanced cycles K epochs."""
    angles = policy['angles']
    if policy['mode'] == 'balanced':
        return angles[(original_id + seed + epoch) % len(angles)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, original_id]))
    return float(rng.choice(angles, p=policy['probabilities']))

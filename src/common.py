"""Shared configuration, reproducibility and serialization helpers."""
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def settings(path):
    with open(path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get('selection_metric', 'clean_validation_accuracy') != 'clean_validation_accuracy':
        raise ValueError('Only preregistered clean_validation_accuracy selection is implemented')
    conv = cfg.get('conventional_augmentation', {})
    if (set(conv) - {'crop_padding', 'horizontal_flip_probability'} or
            not isinstance(conv.get('crop_padding', 0), int) or conv.get('crop_padding', 0) < 0 or
            not 0 <= conv.get('horizontal_flip_probability', 0) <= 1):
        raise ValueError('Invalid conventional augmentation')
    # Keep the historical identifier readable without rewriting saved run metadata.
    if cfg.get('protocol') in ('c4-rotation', 'phase3-primary'):
        if (cfg['augmentation'] != dict(mode='online', angles=[0,90,180,270], probabilities=[.25]*4)
                or cfg['test_angles'] != [0,90,180,270]):
            raise ValueError('Primary protocol requires uniform online C4 augmentation and four test angles')
    if cfg['dataset'] not in ('mnist', 'cifar10', 'breastmnist'):
        raise ValueError('Unsupported dataset')
    if cfg['model'] not in ('cnn', 'c4', 'c8'):
        raise ValueError('model must be cnn, c4 or c8')
    if cfg['training_regime'] not in ('original', 'rotation'):
        raise ValueError('training_regime must be original or rotation')
    for key in ('batch_size', 'max_steps', 'validate_every', 'num_threads'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg.get('checkpoint_every', cfg['validate_every']) < 1:
        raise ValueError('checkpoint_every must be positive')
    if cfg['num_workers'] < 0 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid worker count or learning rate')
    policy = cfg['augmentation']
    angles, probs = policy['angles'], policy['probabilities']
    if (not angles or len(angles) != len(probs) or
            not np.isfinite(angles).all() or not np.isfinite(probs).all() or
            any(p < 0 for p in probs) or not np.isclose(sum(probs), 1)):
        raise ValueError('Angles/probabilities must define a distribution')
    if 0 not in angles or probs[angles.index(0)] <= 0:
        raise ValueError('Augmentation must include identity with positive probability')
    if policy['mode'] not in ('online', 'balanced'):
        raise ValueError('Augmentation mode must be online or balanced')
    if policy['mode'] == 'balanced' and not np.allclose(probs, 1 / len(probs)):
        raise ValueError('Balanced coverage requires uniform probabilities')
    angles = cfg['test_angles']
    if not angles or angles[0] != 0 or len(set(angles)) != len(angles) or not np.isfinite(angles).all():
        raise ValueError('test_angles must be unique, finite and start at zero')
    if cfg['rotation']['interpolation'] not in ('bilinear', 'nearest'):
        raise ValueError('Supported interpolation: bilinear, nearest')
    for widths in cfg['channels'].values():
        if len(widths) != 4 or any(w < 1 for w in widths):
            raise ValueError('Exactly four positive stage widths are required')


def seed_everything(seed: int, deterministic: bool = True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # TF32 convolution can produce visible C4 discrepancies despite float32 tensors.
    # Keep the numerical protocol explicit on Ampere and later CUDA devices.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_model(cfg):
    from src.models.cnn import BaselineCNN
    from src.models.gcnn import C4GCNN
    opts = dict(num_classes=2 if cfg.get('dataset') == 'breastmnist' else 10,
                in_channels=3 if cfg.get('dataset') == 'cifar10' else 1)
    if 'channels' in cfg:
        opts['channels'] = tuple(cfg['channels'][cfg['model']])
    if cfg['model'] == 'cnn':
        return BaselineCNN(**opts)
    if cfg['model'] == 'c4':
        return C4GCNN(**opts)
    if cfg['model'] == 'c8':
        from src.models.gcnn import CyclicGCNN
        return CyclicGCNN(order=8, **opts)
    raise ValueError('Unknown model')


def accuracy(model, loader, device, angle=0):
    from src.data.rotations import rotate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]
            x, y = rotate(x, angle).to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    if not total:
        raise ValueError('Cannot evaluate an empty loader')
    return correct / total


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to write')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    versions = {}
    for name in ('torch', 'torchvision', 'escnn', 'numpy', 'scipy', 'PyYAML', 'medmnist', 'matplotlib',
                 'pandas', 'scikit-learn', 'scikit-image', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    processor = platform.processor()
    if Path('/proc/cpuinfo').exists():
        processor = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), 'unknown')
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=processor, versions=versions,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                precision_settings=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                        cudnn_tf32=torch.backends.cudnn.allow_tf32))

"""Dataset adapters, split provenance and image views."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms.functional import to_tensor, normalize, pad, crop, hflip
from src.common import digest, seed_worker, write_json
from src.data.rotations import rotate, sample_angle

NORMALIZATION = {'mnist': ((.1307,), (.3081,)),
                 'cifar10': ((.4914, .4822, .4465), (.247, .2435, .2616)),
                 'breastmnist': ((.5,), (.5,))}


def raw_dataset(cfg, split):
    if cfg['dataset'] not in NORMALIZATION or split not in ('train', 'val', 'test'):
        raise ValueError('Unknown dataset or split')
    if split == 'val' and cfg['dataset'] != 'breastmnist':
        raise ValueError('MNIST/CIFAR validation must come from fixed_split, not a separate API split')
    root = Path(cfg['data_root'])
    root.mkdir(parents=True, exist_ok=True)
    name = cfg['dataset']
    if name in ('mnist', 'cifar10'):
        cls = datasets.MNIST if name == 'mnist' else datasets.CIFAR10
        return cls(root=str(root), train=split != 'test', download=cfg['download'])
    try:
        from medmnist import BreastMNIST
    except ImportError as exc:
        raise ImportError('BreastMNIST requires pip install medmnist==3.0.2') from exc
    return BreastMNIST(root=str(root), split=split, download=cfg['download'], size=28)


def dataset_fingerprint(base):
    """Hash actual images and labels, not just their count."""
    images = base.data if hasattr(base, 'data') else base.imgs
    labels = base.targets if hasattr(base, 'targets') else base.labels
    h = hashlib.sha256()
    for array in (images, labels):
        arr = np.asarray(array)
        h.update(str((arr.shape, arr.dtype)).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def supplied_groups(base):
    """Preserve supplied patient/group IDs; never infer them from image order."""
    for name in ('patient_ids', 'group_ids'):
        if hasattr(base, name):
            values = np.asarray(getattr(base, name)).reshape(-1)
            if len(values) != len(base):
                raise ValueError(f'{name} length does not match dataset')
            return values.astype(str).tolist()
    return None


def labels_of(base):
    return np.asarray(base.targets if hasattr(base, 'targets') else base.labels).reshape(-1)


def class_report(base, ids, name):
    """Observed split counts, with official label semantics when available."""
    labels = labels_of(base)[list(ids)]
    classes = 2 if name == 'breastmnist' else 10
    counts = np.bincount(labels.astype(int), minlength=classes)
    info = getattr(base, 'info', {})
    return dict(n=len(labels), class_counts=counts.tolist(),
                class_fractions=(counts / len(labels)).tolist(),
                label_names=info.get('label', getattr(base, 'classes', None)),
                task=info.get('task', 'multi-class'),
                supplied_group_ids=supplied_groups(base) is not None)


def stratified_partition(labels, n, rng):
    """Proportional allocation with largest remainders; no sklearn dependency."""
    classes, counts = np.unique(labels, return_counts=True)
    if not 0 < n < len(labels):
        raise ValueError('validation_size must leave nonempty train and validation sets')
    exact = counts * n / len(labels)
    allocated = np.floor(exact).astype(int)
    order = np.argsort(-(exact - allocated), kind='stable')
    allocated[order[:n - allocated.sum()]] += 1
    train, val = [], []
    for label, count in zip(classes, allocated):
        ids = rng.permutation(np.flatnonzero(labels == label))
        val.extend(ids[:count]); train.extend(ids[count:])
    return rng.permutation(train), rng.permutation(val)


def fixed_split(cfg, train_base, val_base=None):
    """Save and validate original-example IDs, independent of training seed."""
    spec = dict(dataset=cfg['dataset'], split_seed=cfg['split_seed'],
                train_fingerprint=dataset_fingerprint(train_base),
                val_fingerprint=dataset_fingerprint(val_base) if val_base is not None else None,
                validation_size=cfg['validation_size'], train_limit=cfg['train_limit'],
                val_limit=cfg['val_limit'], train_groups=supplied_groups(train_base),
                val_groups=supplied_groups(val_base) if val_base is not None else None)
    spec['split_method'] = ('official' if val_base is not None else
                            'stratified-v1' if cfg['dataset'] == 'cifar10' else 'permutation-v1')
    rng = np.random.default_rng(cfg['split_seed'])
    ids = rng.permutation(len(train_base))
    if val_base is None:
        n = cfg['validation_size']
        if not 0 < n < len(ids):
            raise ValueError('validation_size must leave nonempty train and validation sets')
        val_ids, train_ids = ids[:n], ids[n:]
        if cfg['dataset'] == 'cifar10':
            train_ids, val_ids = stratified_partition(labels_of(train_base), n, rng)
    else:
        train_ids = ids
        val_ids = rng.permutation(len(val_base))
    def limited(ids, limit):
        if limit is not None and not 0 < limit <= len(ids):
            raise ValueError('Subset limit must be positive and within split size')
        return ids[:limit].tolist() if limit else ids.tolist()
    manifest = dict(spec=spec, train_ids=limited(train_ids, cfg['train_limit']),
                    val_ids=limited(val_ids, cfg['val_limit']))
    manifest['class_reports'] = dict(
        train=class_report(train_base, manifest['train_ids'], cfg['dataset']),
        validation=class_report(val_base if val_base is not None else train_base,
                                manifest['val_ids'], cfg['dataset']))
    if val_base is not None and spec['train_groups'] is not None and spec['val_groups'] is not None:
        if set(spec['train_groups']) & set(spec['val_groups']):
            raise ValueError('Supplied patient/group IDs overlap official training and validation splits')
    path = Path(cfg['data_root']) / 'splits' / f"{cfg['dataset']}_{digest(spec)[:16]}.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Saved split was altered: {path}')
    else:
        write_json(path, manifest)
    manifest['hash'] = digest(manifest)
    return manifest


class ImageView(Dataset):
    """One retrieval per original ID; augmentation is train-only and epoch-aware."""
    def __init__(self, base, ids, cfg, training=False, angle=0):
        self.base, self.ids, self.cfg = base, list(ids), cfg
        self.training, self.angle, self.epoch = training, angle, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        original_id = self.ids[index]
        x, y = self.base[original_id]
        x = to_tensor(x) if not torch.is_tensor(x) else x.float()
        expected = (3, 32, 32) if self.cfg['dataset'] == 'cifar10' else (1, 28, 28)
        if tuple(x.shape) != expected or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(f'Expected finite raw [0,1] image of shape {expected}')
        conventional = self.cfg.get('conventional_augmentation', {})
        if self.training and conventional:
            # Separate RNG stream from the rotation distribution and model RNG.
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg['seed'], self.epoch, original_id, 991]))
            padding = conventional.get('crop_padding', 0)
            if padding:
                height, width = x.shape[-2:]
                x = pad(x, padding, fill=0, padding_mode='constant')
                top, left = rng.integers(2 * padding + 1, size=2)
                x = crop(x, int(top), int(left), height, width)
            if rng.random() < conventional.get('horizontal_flip_probability', 0):
                x = hflip(x)
        angle = self.angle
        if self.training and self.cfg['training_regime'] == 'rotation':
            key = index if self.cfg['augmentation']['mode'] == 'balanced' else original_id
            angle = sample_angle(self.cfg['augmentation'], self.cfg['seed'], self.epoch, key)
        x = rotate(x, angle, **self.cfg['rotation'])
        x = normalize(x, *NORMALIZATION[self.cfg['dataset']])
        return x, int(np.asarray(y).item()), original_id


def loader(view, cfg, shuffle=False):
    return DataLoader(view, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg['num_workers'], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(cfg['seed']),
                      persistent_workers=False, drop_last=False,
                      multiprocessing_context='spawn' if cfg['num_workers'] else None)


def training_data(cfg):
    base = raw_dataset(cfg, 'train')
    val_base = raw_dataset(cfg, 'val') if cfg['dataset'] == 'breastmnist' else None
    manifest = fixed_split(cfg, base, val_base)
    train = ImageView(base, manifest['train_ids'], cfg, training=True)
    val = ImageView(val_base if val_base is not None else base, manifest['val_ids'], cfg)
    return loader(train, cfg, True), loader(val, cfg), manifest


def test_data(cfg):
    base = raw_dataset(cfg, 'test')
    limit = cfg['test_limit']
    if limit is not None and not 0 < limit <= len(base):
        raise ValueError('test_limit out of range')
    ids = list(range(len(base) if limit is None else limit))
    return base, ids, dict(fingerprint=dataset_fingerprint(base), ids=ids,
                          class_report=class_report(base, ids, cfg['dataset']),
                          normalization=NORMALIZATION[cfg['dataset']], rotation=cfg['rotation'],
                          group_ids=supplied_groups(base)[:len(ids)] if supplied_groups(base) is not None else None)

"""Evaluate a saved checkpoint at all configured angles, restoring its weights."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.common import build_model, digest, environment, write_csv, write_json, seed_everything
from src.datasets import ImageView, loader, test_data, training_data, dataset_fingerprint, class_report, NORMALIZATION
from src.evaluation.metrics import classification_metrics, probability_consistency


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint['config']).to(device)
    model.train()  # remove constructor-created ESCNN filter caches before strict load
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def evaluate(path, device=None, extended=False, *, validation=False):
    """Evaluate test data by default; smoke callers explicitly use validation.

    Validation outputs live under diagnostics/validation and cannot overwrite or
    be aggregated as primary test predictions. Training/split settings are intact.
    """
    path = Path(path)
    # Read config on CPU first, never rely on the filename for output paths.
    model, checkpoint = load_checkpoint(path)
    cfg = dict(checkpoint['config'])
    seed_everything(cfg['seed'], cfg['deterministic'])
    if extended:
        cfg['test_angles'] = list(range(0, 360, 45))
    torch.set_num_threads(cfg['num_threads'])
    device = torch.device(device or cfg['device'])
    model.to(device).eval()
    if validation:
        _, val, split = training_data(cfg)
        base, ids = val.dataset.base, val.dataset.ids
        manifest = dict(split='validation', split_hash=split['hash'], ids=ids,
                        fingerprint=dataset_fingerprint(base), group_ids=None,
                        class_report=class_report(base, ids, cfg['dataset']),
                        normalization=NORMALIZATION[cfg['dataset']], rotation=cfg['rotation'])
    else:
        base, ids, manifest = test_data(cfg)
    if not validation and manifest['group_ids'] is not None:
        split = json.loads((path.parent / 'split.json').read_text())['spec']
        known = set((split.get('train_groups') or []) + (split.get('val_groups') or []))
        if known & set(manifest['group_ids']):
            raise ValueError('Supplied patient/group IDs overlap development and test splits')
    manifest['angles'] = cfg['test_angles']
    manifest['transform_versions'] = {k:v for k,v in environment()['versions'].items() if k in ('torch','torchvision')}
    test_hash = digest(manifest)
    rows, predictions, probabilities = [], [], []
    inference_seconds = 0.0
    all_labels = None
    for angle in cfg['test_angles']:
        batches = loader(ImageView(base, ids, cfg, angle=angle), cfg)
        pred, labels, sample_ids, probs = [], [], [], []
        with torch.no_grad():
            for x, y, index in batches:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                probability = model(x.to(device)).softmax(1)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - tick
                probs.extend(probability.cpu().tolist())
                pred.extend(probability.argmax(1).cpu().tolist())
                labels.extend(y.tolist())
                sample_ids.extend(index.tolist())
        assert sample_ids == ids
        if all_labels is not None:
            assert labels == all_labels
        all_labels = labels
        metrics = classification_metrics(labels, pred, 2 if cfg['dataset'] == 'breastmnist' else 10, probs)
        metrics['output_consistency_l1'] = probability_consistency(probabilities[0] if probabilities else probs, probs)
        rows.append(dict(dataset=cfg['dataset'], model=cfg['model'],
                         training_regime=cfg['training_regime'], seed=cfg['seed'],
                         angle=angle, n=len(ids), **metrics))
        predictions.append(pred)
        probabilities.append(probs)
    clean = rows[0]['accuracy']
    for row in rows:
        row['degradation_from_zero'] = clean - row['accuracy']
    destination = path.parent / 'diagnostics' / 'validation' if validation else path.parent
    out = destination / 'diagnostics' / 'extended_rotation' if extended else destination
    write_csv(out / 'evaluation.csv', rows)
    np.savez_compressed(out / 'predictions.npz', predictions=np.array(predictions),
                        probabilities=np.array(probabilities, dtype=np.float32),
                        labels=np.array(all_labels), ids=np.array(ids), angles=np.array(cfg['test_angles']))
    write_json(out / 'evaluation_metadata.json', dict(test_hash=test_hash, test_manifest=manifest,
               environment=environment(), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checkpoint_step=checkpoint['step'], device=str(device),
               benchmark=('validation smoke diagnostic' if validation else
                          'extended diagnostic' if extended else 'configured evaluation'),
               binary_positive_label=1 if cfg['dataset']=='breastmnist' else None,
               auroc_unavailable_reason='requires probabilities and both classes in evaluated subset',
               inference_seconds=inference_seconds, inference_examples=len(ids)*len(rows),
               timing_scope='transfer + forward + softmax; synchronized; excludes data loading; no warmup',
               batch_size=cfg['batch_size'], precision='float32'))
    primary_rows = [r for r in rows if r['angle'] in (0,90,180,270)]
    write_json(out / 'evaluation_summary.json', dict(clean_accuracy=clean,
               mean_accuracy=float(np.mean([r['accuracy'] for r in primary_rows])),
               worst_angle_accuracy=min(r['accuracy'] for r in primary_rows),
               rotated_accuracy=float(np.mean([r['accuracy'] for r in primary_rows[1:]])) if len(primary_rows) > 1 else None,
               configured_angle_mean_accuracy=float(np.mean([r['accuracy'] for r in rows])),
               primary_angles=[r['angle'] for r in primary_rows]))
    return rows

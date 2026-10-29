"""Six training configurations per dataset/seed; each checkpoint gets every angle."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
from src.common import settings
from src.training.engine import train, run_path
from src.training.recovery import TrainingInterrupted, interruption_flag
from src.evaluate import evaluate


def configurations(cfg):
    for seed in cfg['seeds']:
        for model in cfg['models']:
            for regime in ('original', 'rotation'):
                item = copy.deepcopy(cfg)
                item.update(seed=seed, model=model, training_regime=regime)
                yield item


def is_validation_config(path, cfg):
    """Recognize bundled smoke configs without changing their training settings."""
    return ('validation' in Path(path).parts or 'smoke' in Path(path).stem
            or 'validation' in cfg.get('track', '') or 'smoke' in cfg.get('track', ''))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/breastmnist.yaml')
    ap.add_argument('--resume', action='store_true', help='Continue interrupted runs and skip completed evaluations')
    ap.add_argument('--status', action='store_true', help='Show progress without starting training')
    ap.add_argument('--analysis-output', help='Aggregate and plot after the whole matrix completes')
    args = ap.parse_args()
    cfg = settings(args.config)
    if not args.status and is_validation_config(args.config, cfg):
        ap.error('Validation configurations are not research runs; use synthetic tests for pipeline checks')
    if args.status:
        for item in configurations(cfg):
            folder = run_path(item)
            metadata = folder / 'metadata.json'
            state = json.loads(metadata.read_text()) if metadata.exists() else {}
            print(f"{item['model']}/{item['training_regime']}/{item['seed']}: "
                  f"{state.get('status', 'pending')} "
                  f"{state.get('optimizer_steps', 0)}/{item['max_steps']} steps; "
                  f"evaluation={'complete' if evaluation_complete(folder) else 'pending'}")
        return
    try:
        with interruption_flag() as stopping:
            for index, item in enumerate(configurations(cfg), 1):
                if stopping():
                    raise TrainingInterrupted('Matrix interrupted; rerun with --resume to continue')
                print(f"Run {index}/{len(cfg['models']) * len(cfg['seeds']) * 2}: "
                      f"{item['model']}/{item['training_regime']}/{item['seed']}", flush=True)
                checkpoint = train(item, resume=args.resume, stop_requested=stopping)
                if stopping():
                    raise TrainingInterrupted('Training saved; evaluation is pending')
                if not (args.resume and evaluation_complete(checkpoint.parent)):
                    evaluate(checkpoint)
            if stopping():
                raise TrainingInterrupted('Training/evaluation saved; analysis is pending')
            if args.analysis_output:
                from src.analysis.aggregate import aggregate
                from src.analysis.plots import plot_results
                aggregate(Path(cfg['output_dir']) / cfg['dataset'], args.analysis_output)
                plot_results(args.analysis_output, Path(args.analysis_output) / 'figures')
    except TrainingInterrupted as exc:
        print(str(exc), flush=True)
        raise SystemExit(130)


def evaluation_complete(folder):
    required = ('best.pt', 'evaluation.csv', 'evaluation_metadata.json',
                'evaluation_summary.json', 'predictions.npz')
    if not all((folder / name).exists() for name in required):
        return False
    try:
        metadata = json.loads((folder / 'evaluation_metadata.json').read_text())
        json.loads((folder / 'evaluation_summary.json').read_text())
        with (folder / 'evaluation.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        with np.load(folder / 'predictions.npz', allow_pickle=False) as arrays:
            labels, ids, angles, predictions = (arrays[key] for key in ('labels', 'ids', 'angles', 'predictions'))
            if (len(ids) != len(labels) or predictions.shape != (len(angles), len(ids))
                    or [float(row['angle']) for row in rows] != angles.tolist()
                    or any(int(row['n']) != len(ids) for row in rows)):
                return False
    except (ValueError, OSError, KeyError, EOFError, BadZipFile):
        return False
    return metadata.get('checkpoint_sha256') == hashlib.sha256((folder / 'best.pt').read_bytes()).hexdigest()


if __name__ == '__main__':
    main()

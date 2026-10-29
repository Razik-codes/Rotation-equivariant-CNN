"""BreastMNIST architecture-by-augmentation study using the existing models, trainer and statistics."""
import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.train_matrix import configurations, evaluation_complete, is_validation_config
from src.common import environment, seed_everything, settings, write_csv, write_json
from src.datasets import ImageView, dataset_fingerprint, raw_dataset, training_data
from src.evaluate import evaluate, load_checkpoint
from src.evaluation.equivariance import diagnose
from src.training.engine import run_path, train
from src.training.recovery import interruption_flag


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_design(cfg):
    if (cfg['models'] != ['cnn', 'c4'] or cfg['dataset'] != 'breastmnist'
            or cfg['augmentation'] != dict(mode='online', angles=[0,90,180,270], probabilities=[.25]*4)
            or cfg['test_angles'] != [0,90,180,270]
            or cfg['selection_metric'] != 'clean_validation_accuracy'
            or cfg['conventional_augmentation'] != dict(crop_padding=0, horizontal_flip_probability=0.0)
            or len(set(cfg['seeds'])) != len(cfg['seeds'])):
        raise ValueError('Study requires the frozen BreastMNIST CNN/C4 2x2 design')


def freeze(cfg, config_path, root):
    """Record the protocol/data before training; never silently replace a freeze."""
    root.mkdir(parents=True, exist_ok=True)
    evidence = dict(config=cfg, config_sha256=sha(config_path),
                    protocol_sha256=sha('docs/methodology.md'),
                    source_hashes={str(p): sha(p) for d in ('src','scripts')
                                   for p in sorted(Path(d).rglob('*.py'))},
                    environment=environment(), dataset_sha256=sha(Path(cfg['data_root'])/'breastmnist.npz'))
    provenance = root/'provenance'
    provenance.mkdir(exist_ok=True)
    path = provenance/'freeze.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous[k] != v for k,v in evidence.items()):
            raise ValueError('Frozen configuration, source, protocol, data or environment changed')
        return
    # Check counts and exact pixel duplicates, without viewing test predictions.
    bases = {s: raw_dataset(cfg,s) for s in ('train','val','test')}
    hashes = {s: [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in b.imgs]
              for s,b in bases.items()}
    development_hashes=set(hashes['train']) | set(hashes['val'])
    evidence.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
        label_validity='exploratory; arbitrary ultrasound rotations not clinically validated',
        dataset={s:dict(n=len(b), labels=np.bincount(b.labels.reshape(-1)).tolist(),
                       fingerprint=dataset_fingerprint(b),
                       repeated_exact_images=len(hashes[s])-len(set(hashes[s]))) for s,b in bases.items()},
        cross_split_exact_image_matches={f'{a}/{b}':len(set(hashes[a]) & set(hashes[b]))
                                        for a,b in itertools.combinations(bases,2)},
        sensitivity_excluded_test_ids=[i for i,h in enumerate(hashes['test']) if h in development_hashes])
    write_json(path,evidence)
    shutil.copy2(config_path,provenance/'config.yaml')
    shutil.copy2('docs/methodology.md',provenance/'protocol.md')
    (provenance/'requirements.txt').write_text(subprocess.check_output(
        [__import__('sys').executable,'-m','pip','freeze'],text=True))


def verify_run(path, repeat=False):
    model, checkpoint = load_checkpoint(path)
    cfg = checkpoint['config']
    seed_everything(cfg['seed'],cfg['deterministic'])
    device = cfg['device']
    torch.set_num_threads(cfg['num_threads'])
    model.to(device).eval()
    _, val, split = training_data(cfg)
    x, _, ids = next(iter(val))
    # Fixed first 16 validation IDs, independent of the training seed.
    x, ids = x[:16].to(device), ids[:16]
    if cfg['model'] == 'c4':
        rows = diagnose(model,x)
        if not all(np.isfinite(r['full_error']) for r in rows):
            raise ValueError('Non-finite equivariance diagnostic')
        if max(r['full_error'] for r in rows) >= 2e-5:
            raise ValueError('Validation C4 equivariance error exceeds the predefined 2e-5 tolerance')
        write_csv(path.parent/'equivariance.csv',rows)
        write_json(path.parent/'equivariance_metadata.json',dict(
            validation_ids=ids.tolist(),split_hash=split['hash'],device=device,
            dtype=str(x.dtype),epsilon=1e-8,interior_margin=1,
            feature_error='mean per-image ||actual-expected||2 / (||actual||2 + 1e-8)',
            action='ESCNN FieldType fiber action plus exact spatial torch.rot90',
            output_error='mean per-image probability L1; evaluation mode'))
    with np.load(path.parent/'predictions.npz') as arrays:
        first={k:arrays[k].copy() for k in arrays.files}
    if repeat:
        evaluate(path)
        with np.load(path.parent/'predictions.npz') as arrays:
            if not all(np.array_equal(v,arrays[k]) for k,v in first.items()):
                raise ValueError('Repeated evaluation changed predictions/probabilities')
    # Record actual transformed tensors to verify identical evaluation across runs.
    base=raw_dataset(cfg,'test')
    transforms={}
    for angle in cfg['test_angles']:
        view=ImageView(base,first['ids'],cfg,angle=angle)
        h=hashlib.sha256()
        for i in range(len(view)):
            h.update(view[i][0].numpy().tobytes())
        transforms[str(angle)]=h.hexdigest()
    write_json(path.parent/'validation_checks.json',dict(
        repeated_evaluation='bitwise identical' if repeat else 'not requested',
        test_transform_sha256=transforms, split_hash=split['hash'],
        checkpoint_step=checkpoint['step'], checkpoint_sha256=sha(path)))


def archive_and_analyze(cfg, root):
    """Publish compact raw evidence; exclude weights, recovery files and images."""
    from src.analysis.aggregate import aggregate
    from src.analysis.plots import plot_results
    reference=None
    for item in configurations(cfg):
        folder=run_path(item)
        if not evaluation_complete(folder):
            raise ValueError(f'Missing or invalid evaluation: {folder}')
        checks=json.loads((folder/'validation_checks.json').read_text())
        comparable=(checks['split_hash'],checks['test_transform_sha256'])
        if reference is not None and comparable != reference:
            raise ValueError('Split or transformed test tensors differ between runs')
        reference=comparable
        dest=root/'records'/item['model']/item['training_regime']/str(item['seed'])
        dest.mkdir(parents=True,exist_ok=True)
        for pattern in ('*.json','*.csv','*.yaml','predictions.npz'):
            for source in folder.glob(pattern):
                shutil.copy2(source,dest/source.name)
    aggregate(root/'records',root)
    # Wide table makes the 48 seed/angle measurements immediately reviewable.
    rows=[]
    for path in sorted((root/'records').rglob('predictions.npz')):
        local=settings(path.parent/'config.yaml')
        with np.load(path) as arr:
            rows.extend(dict(model=local['model'],regime=local['training_regime'],seed=local['seed'],
                angle=int(angle),n=len(arr['ids']),accuracy=float((arr['predictions'][i]==arr['labels']).mean()),
                prediction_agreement_with_zero=float((arr['predictions'][i]==arr['predictions'][0]).mean()))
                for i,angle in enumerate(arr['angles']))
    write_csv(root/'per_angle_accuracy.csv',rows)
    # Predeclared sensitivity: exclude exact test/development image duplicates.
    # Keep the primary official split results intact and use no model predictions
    # to decide exclusions. This still cannot establish patient independence.
    from src.analysis.statistics import seed_summary, paired_seed_summary
    from src.analysis.interactions import interactions
    excluded=json.loads((root/'provenance'/'freeze.json').read_text())['sensitivity_excluded_test_ids']
    sensitivity=[]
    values={}
    for path in sorted((root/'records').rglob('predictions.npz')):
        local=settings(path.parent/'config.yaml')
        with np.load(path) as arr:
            mask=~np.isin(arr['ids'],excluded)
            acc=(arr['predictions'][:,mask]==arr['labels'][mask]).mean(1)
            metric=dict(clean_accuracy=float(acc[0]),mean_accuracy=float(acc.mean()),
                        worst_angle_accuracy=float(acc.min()),rotated_accuracy=float(acc[1:].mean()))
            metric.update({f'accuracy@{int(a)}':float(v) for a,v in zip(arr['angles'],acc)})
            values.setdefault((local['model'],local['training_regime']),{})[local['seed']]=metric
            sensitivity.extend(dict(model=local['model'],regime=local['training_regime'],seed=local['seed'],
                angle=int(a),n=int(mask.sum()),accuracy=float(v)) for a,v in zip(arr['angles'],acc))
    write_csv(root/'duplicate_sensitivity_per_angle.csv',sensitivity)
    write_csv(root/'duplicate_sensitivity_seed_summary.csv',[
        dict(model=m,regime=r,metric=key,**seed_summary([v[key] for v in seeded.values()]))
        for (m,r),seeded in values.items() for key in next(iter(seeded.values()))])
    contrasts,_,_,_=interactions(values)
    write_csv(root/'duplicate_sensitivity_interactions.csv',contrasts)
    write_csv(root/'duplicate_sensitivity_model_comparisons.csv',[
        dict(comparison=f'c4 minus cnn / {r}',metric='mean_accuracy',**paired_seed_summary(
            {s:v['mean_accuracy'] for s,v in values['c4',r].items()},
            {s:v['mean_accuracy'] for s,v in values['cnn',r].items()})) for r in ('original','rotation')])
    write_json(root/'duplicate_sensitivity_metadata.json',dict(excluded_test_ids=excluded,
        criterion='exact raw image match to train or validation; frozen before training',
        limitation='Does not remove unknown near duplicates or patient dependence'))
    plot_results(root,root/'figures')
    write_json(root/'manifest.json',{str(p.relative_to(root)):sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--results',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--repeat-evaluation',action='store_true',help='Validation: require bitwise repeat probabilities')
    args=ap.parse_args()
    cfg=settings(args.config)
    if is_validation_config(args.config, cfg):
        ap.error('Validation configurations are not research runs; use synthetic tests for pipeline checks')
    validate_design(cfg)
    seed_everything(cfg['seed'],cfg['deterministic'])
    root=Path(args.results)
    freeze(cfg,args.config,root)
    with interruption_flag() as stopping:
        for i,item in enumerate(configurations(cfg),1):
            print(f'Study run {i}/{4*len(cfg["seeds"])}',flush=True)
            path=train(item,resume=args.resume,stop_requested=stopping)
            evaluate(path)
            verify_run(path,repeat=args.repeat_evaluation)
    archive_and_analyze(cfg,root)


if __name__=='__main__':
    main()

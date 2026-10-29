"""Aggregate comparable saved runs and matched raw predictions."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import yaml
from src.common import digest, write_csv, write_json
from src.analysis.statistics import seed_summary, paired_seed_summary, clustered_bootstrap_difference
from src.evaluation.metrics import classification_metrics, probability_consistency


def read_runs(root):
    runs = []
    for path in sorted(Path(root).rglob('evaluation.csv')):
        folder = path.parent
        if 'diagnostics' in folder.relative_to(root).parts:
            continue  # optional diagnostics are not independent trained runs
        cfg = yaml.safe_load((folder/'config.yaml').read_text())
        metadata = json.loads((folder/'metadata.json').read_text())
        evaluation = json.loads((folder/'evaluation_metadata.json').read_text())
        if metadata['status'] != 'complete':
            raise ValueError(f'Incomplete run: {folder}')
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        with np.load(folder/'predictions.npz', allow_pickle=False) as arrays:
            predictions = {key: arrays[key].copy() for key in arrays.files}
        # Changes in budgets, preprocessing, architecture specifications or software
        # define a separate experiment and must not be silently pooled.
        excluded = {'model','training_regime','seed','seeds','models','data_root','output_dir','device','num_threads','num_workers','download'}
        protocol = {key:value for key,value in cfg.items() if key not in excluded}
        signature = digest(dict(protocol=protocol, split=metadata['split_hash'],
                                test=evaluation['test_hash'], versions=metadata['environment']['versions'],
                                source=metadata.get('source_hashes')))
        runs.append(dict(path=str(folder), cfg=cfg, metadata=metadata, evaluation=evaluation,
                         rows=rows, predictions=predictions, signature=signature))
    if not runs:
        raise ValueError(f'No evaluation.csv files found under {root}')
    if len({r['signature'] for r in runs}) != 1:
        raise ValueError('Mixed experimental protocols found; aggregate each protocol separately')
    keys = [(r['cfg']['model'],r['cfg']['training_regime'],r['cfg']['seed']) for r in runs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate model/regime/seed runs')
    reference = runs[0]['predictions']
    for run in runs:
        for key in ('labels','ids','angles'):
            if not np.array_equal(reference[key], run['predictions'][key]):
                raise ValueError(f'Cannot pair mismatched {key}')
        validate_measurements(run)
    return runs


def validate_measurements(run):
    """Reject stale/corrupted metric tables rather than silently reporting them."""
    arrays, rows = run['predictions'], run['rows']
    labels, ids, angles = (arrays[k] for k in ('labels','ids','angles'))
    predictions = arrays['predictions']
    if (labels.ndim != 1 or len(ids) != len(labels) or len(set(ids.tolist())) != len(ids)
            or predictions.shape != (len(angles),len(labels))
            or [float(r['angle']) for r in rows] != angles.tolist()
            or angles.tolist() != run['cfg']['test_angles']):
        raise ValueError('Saved evaluation dimensions/angles do not match the protocol')
    if ids.tolist() != run['evaluation']['test_manifest']['ids']:
        raise ValueError('Saved IDs differ from test manifest')
    classes = 2 if run['cfg']['dataset'] == 'breastmnist' else 10
    probabilities = arrays.get('probabilities')
    if probabilities is not None:
        if probabilities.shape != (len(angles),len(ids),classes) or not np.array_equal(probabilities.argmax(2), predictions):
            raise ValueError('Saved probabilities disagree with predictions')
    clean = float((predictions[0] == labels).mean())
    for i,row in enumerate(rows):
        computed = classification_metrics(labels,predictions[i],classes,
                                           probabilities[i] if probabilities is not None else None)
        computed['degradation_from_zero'] = clean - computed['accuracy']
        if probabilities is not None:
            computed['output_consistency_l1'] = probability_consistency(probabilities[0],probabilities[i])
        for metric, value in computed.items():
            if metric not in row:  # preserve legacy files lacking new metrics
                continue
            if value is None:
                if row[metric] != '':
                    raise ValueError(f'Undefined {metric} must be missing')
            elif not np.isclose(float(row[metric]),value,rtol=1e-6,atol=1e-7):
                raise ValueError(f'Saved {metric} disagrees with raw predictions')
        if int(row['n']) != len(ids):
            raise ValueError('Saved example count disagrees with predictions')


def run_metrics(run):
    rows = run['rows']
    metrics = {}
    for row in rows:
        for name in ('accuracy','macro_f1','binary_f1','malignant_f1','auroc','output_consistency_l1','degradation_from_zero'):
            if row.get(name) not in (None, ''):
                metrics[f"{name}@{float(row['angle']):g}"] = float(row[name])
    accuracies = [float(row['accuracy']) for row in rows if float(row['angle']) in (0,90,180,270)]
    metrics.update(clean_accuracy=accuracies[0], mean_accuracy=float(np.mean(accuracies)),
                   worst_angle_accuracy=min(accuracies))
    if len(rows) > 1:
        metrics['rotated_accuracy'] = float(np.mean(accuracies[1:]))
    return metrics


def aggregate(root, output, repetitions=2000):
    runs = read_runs(root)
    groups = {}
    for run in runs:
        key = (run['cfg']['model'],run['cfg']['training_regime'])
        groups.setdefault(key,{})[run['cfg']['seed']] = run
    summaries = []
    for (model,regime), seeded in groups.items():
        values = {seed:run_metrics(run) for seed,run in seeded.items()}
        for metric in sorted(set().union(*(v.keys() for v in values.values()))):
            summaries.append(dict(model=model,training_regime=regime,metric=metric,
                                  **seed_summary([v[metric] for v in values.values() if metric in v])))
    measurements = {g:{seed:run_metrics(run) for seed,run in seeded.items()} for g,seeded in groups.items()}
    from src.analysis.interactions import interactions
    interaction_rows, interaction_raw, augmentation_rows, skipped_interactions = interactions(measurements)
    comparisons, raw_comparisons, bootstraps, skipped = [], [], [], []
    for left, right in itertools.combinations(groups,2):
        if left[0] != right[0] and left[1] != right[1]:
            continue  # model comparisons within regime, regime comparisons within model
        a,b = groups[left],groups[right]
        if set(a) != set(b):
            skipped.append(dict(left=left,right=right,reason='unequal seed sets'))
            continue
        name = f'{left[0]}/{left[1]} minus {right[0]}/{right[1]}'
        common = set.intersection(*(set(run_metrics(r)) for r in [*a.values(), *b.values()]))
        for metric in sorted(common):
            if 'f1' in metric:
                continue
            summary = paired_seed_summary({s:run_metrics(r)[metric] for s,r in a.items()},
                                          {s:run_metrics(r)[metric] for s,r in b.items()})
            comparisons.append(dict(comparison=name, metric=metric, **summary))
            raw_comparisons.extend(dict(comparison=name,metric=metric,seed=s,
                difference=run_metrics(a[s])[metric]-run_metrics(b[s])[metric]) for s in sorted(a))
        def correctness(group):
            return np.stack([group[s]['predictions']['predictions'] == group[s]['predictions']['labels'][None,:]
                             for s in sorted(group)]).astype(float)
        primary = np.isin(runs[0]['predictions']['angles'], [0,90,180,270])
        ca, cb = correctness(a)[:,primary],correctness(b)[:,primary]
        for metric, select in [('clean_accuracy',slice(0,1)),('mean_accuracy',slice(None)),('rotated_accuracy',slice(1,None))]:
            if ca[:,select].shape[1]:
                bootstraps.append(dict(comparison=name, metric=metric,
                    **clustered_bootstrap_difference(ca[:,select],cb[:,select],repetitions=repetitions,
                        cluster_ids=runs[0]['evaluation']['test_manifest'].get('group_ids'))))
    output = Path(output)
    write_csv(output/'seed_summary.csv', summaries)
    write_csv(output/'run_measurements.csv', [dict(dataset=r['cfg']['dataset'],model=r['cfg']['model'],
        training_regime=r['cfg']['training_regime'],seed=r['cfg']['seed'],metric=k,value=v)
        for r in runs for k,v in run_metrics(r).items()])
    for filename,records in [('run_level_differences.csv',raw_comparisons),
            ('interactions.csv',interaction_rows),('interaction_run_differences.csv',interaction_raw),
            ('augmentation_effects.csv',augmentation_rows)]:
        if records:
            write_csv(output/filename,records)
    if comparisons:
        write_csv(output/'paired_seed_differences.csv',comparisons)
    write_json(output/'test_sample_bootstrap.json',bootstraps)
    write_json(output/'analysis_metadata.json',dict(sources=[r['path'] for r in runs],
               signature=runs[0]['signature'], skipped_pairings=skipped, skipped_interactions=skipped_interactions,
               dataset=runs[0]['cfg']['dataset'],
               pairing_basis='identical split, test examples, seed-block data order and stateless augmentation; not identical parameter initialization',
               missing_conditions=[f'{m}/{regime}/{s}' for m in runs[0]['cfg']['models']
                   for regime in ('original','rotation') for s in runs[0]['cfg']['seeds']
                   if (m,regime) not in groups or s not in groups[(m,regime)]],
               expected_seeds=runs[0]['cfg']['seeds'],
               observed={f'{k[0]}/{k[1]}':sorted(v) for k,v in groups.items()},
               uncertainty='t intervals across seeds; image-cluster bootstrap conditional on fixed models; unadjusted exploratory intervals'))
    costs = [dict(model=r['cfg']['model'],training_regime=r['cfg']['training_regime'],seed=r['cfg']['seed'],
                  parameters=r['metadata']['trainable_parameters'],steps=r['metadata']['optimizer_steps'],
                  train_seconds=r['metadata']['train_seconds'],gpu_training_seconds=r['metadata']['gpu_training_seconds'],
                  exposures=r['metadata']['training_exposures'],total_seconds=r['metadata']['total_seconds'],
                  peak_gpu_memory_bytes=r['metadata'].get('peak_gpu_memory_bytes'),
                  inference_seconds=r['evaluation'].get('inference_seconds'),
                  inference_examples=r['evaluation'].get('inference_examples'),
                  batch_size=r['cfg']['batch_size'],precision=r['metadata'].get('precision','float32'),
                  device=r['metadata']['device']) for r in runs]
    write_csv(output/'training_costs.csv',costs)
    return summaries


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--bootstrap-repetitions',type=int,default=2000)
    args=ap.parse_args()
    aggregate(args.runs,args.output,args.bootstrap_repetitions)


if __name__ == '__main__':
    main()

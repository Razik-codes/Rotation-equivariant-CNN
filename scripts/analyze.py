"""Reproduce accuracy summaries from published seed/angle measurements, not predictions."""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from src.analysis.statistics import paired_seed_summary, seed_summary
from src.common import write_csv


def read_measurements(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(m, r, s, a) for m in ('cnn', 'c4') for r in ('original', 'rotation')
                for s in (42, 123, 456) for a in (0, 90, 180, 270)}
    observed = set()
    for row in rows:
        key = (row['model'], row['regime'], int(row['seed']), int(row['angle']))
        if key not in expected or key in observed:
            raise ValueError('Unexpected or duplicate model/regime/seed/angle')
        observed.add(key)
        n, accuracy = int(row['n']), float(row['accuracy'])
        if n != 156 or not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError('Expected finite accuracy on all 156 test images')
        if not np.isclose(accuracy*n, round(accuracy*n), rtol=0, atol=1e-10):
            raise ValueError('Accuracy must correspond to an integer number of correct images')
    if observed != expected:
        raise ValueError('Expected the complete 48-row CNN/C4 study')
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = row['model'], row['regime']
        grouped.setdefault(key, {}).setdefault(int(row['seed']), {})[int(row['angle'])] = float(row['accuracy'])
    means = {key: {seed: float(np.mean([angles[a] for a in (0, 90, 180, 270)]))
                   for seed, angles in seeds.items()} for key, seeds in grouped.items()}
    summary = [dict(model=m, training_regime=r, metric='mean_accuracy',
                    **seed_summary(list(means[m, r].values()))) for m, r in sorted(means)]
    comparisons = []
    for a, b in itertools.combinations(sorted(means), 2):
        if a[0] != b[0] and a[1] != b[1]:
            continue
        comparisons.append(dict(comparison=f'{a[0]}/{a[1]} minus {b[0]}/{b[1]}',
            metric='mean_accuracy', **paired_seed_summary(means[a], means[b])))
    return summary, comparisons


def check_tables(summary, comparisons, reference):
    """Check all advertised numeric cells; harmless floating-point ordering may differ."""
    for filename, actual in (('accuracy_summary.csv', summary), ('paired_accuracy.csv', comparisons)):
        with (Path(reference) / filename).open() as stream:
            expected = list(csv.DictReader(stream))
        if len(expected) != len(actual):
            raise ValueError(f'Reference row count mismatch: {filename}')
        for a, b in zip(actual, expected):
            if set(a) != set(b):
                raise ValueError(f'Reference columns mismatch: {filename}')
            for key, value in a.items():
                equal = (np.isclose(value, float(b[key]), rtol=0, atol=1e-12)
                         if isinstance(value, (int, float)) else value == b[key])
                if not equal:
                    raise ValueError(f'Reference value mismatch: {filename}/{key}')


def plot(rows, comparisons, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, regime in sorted({(r['model'], r['regime']) for r in rows}):
        angles = [0, 90, 180, 270]
        values = [[float(r['accuracy']) for r in rows if
                   (r['model'], r['regime'], int(r['angle'])) == (model, regime, a)] for a in angles]
        mean, sd = np.mean(values, axis=1), np.std(values, axis=1, ddof=1)
        ax.plot(angles, mean, 'o-', label=f'{model} / {regime}')
        ax.fill_between(angles, mean-sd, mean+sd, alpha=.12)
    ax.set(xlabel='Test rotation (degrees)', ylabel='Accuracy (mean ± seed SD)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'accuracy_by_angle.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    means = np.array([r['mean'] for r in comparisons])
    errors = np.array([[r['mean']-r['ci_low'] for r in comparisons],
                       [r['ci_high']-r['mean'] for r in comparisons]])
    ax.errorbar(means, np.arange(len(means)), xerr=errors, fmt='o', capsize=3)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(np.arange(len(means)), [r['comparison'] for r in comparisons], fontsize=7)
    ax.set_xlabel('Four-angle accuracy difference (paired-seed 95% t interval)')
    fig.tight_layout()
    fig.savefig(output / 'paired_confidence_intervals.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/per_angle_accuracy.csv'))
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--check', type=Path, help='Compare with published accuracy tables')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    rows = read_measurements(args.input)
    summary, comparisons = summarize(rows)
    if args.check:
        check_tables(summary, comparisons, args.check)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'accuracy_summary.csv', summary)
    write_csv(args.output / 'paired_accuracy.csv', comparisons)
    if args.plots:
        plot(rows, comparisons, args.output)
    print('48 seed/angle measurements; 4 summaries; 4 paired contrasts.'
          + (' Published numeric cells match within 1e-12.' if args.check else ''))


if __name__ == '__main__':
    main()

"""Seed uncertainty and conditional test-sample uncertainty, kept separate."""
import numpy as np
from scipy.stats import t


def seed_summary(values, confidence=.95):
    """Student-t CI across independent training seeds (sample SD, ddof=1)."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 < confidence < 1:
        raise ValueError('Expected nonempty finite 1D values and confidence in (0,1)')
    result = dict(n=len(x), mean=float(x.mean()), sd=None, ci_low=None, ci_high=None)
    if len(x) >= 2:
        sd = float(x.std(ddof=1))
        half = float(t.ppf((1 + confidence)/2, len(x)-1) * sd / np.sqrt(len(x)))
        result.update(sd=sd, ci_low=float(x.mean()-half), ci_high=float(x.mean()+half))
    return result


def paired_seed_summary(a, b):
    """Difference A-B using matched raw seed values, never marginal SDs."""
    if set(a) != set(b) or not a:
        raise ValueError('Paired summaries require the exact same nonempty seed set')
    return seed_summary([a[s]-b[s] for s in sorted(a)])


def clustered_bootstrap_difference(a, b, *, cluster_ids=None, repetitions=2000, seed=0):
    """Paired conditional CI, resampling original images or supplied groups.

    Arrays end in the original-example axis; leading seed/angle axes are
    averaged, then every sampled image retains all its angles and fixed
    trained models. Optional group IDs resample whole groups (e.g. patients),
    preserving image-weighted estimands when groups differ in size.
    Does not estimate joint seed-and-sample uncertainty.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] < 2 or repetitions < 2:
        raise ValueError('Equal nonempty paired arrays and >=2 samples/replicates required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Bootstrap inputs must be finite')
    difference = (a-b).reshape(-1,a.shape[-1]).mean(0)
    groups = np.arange(a.shape[-1]) if cluster_ids is None else np.asarray(cluster_ids)
    if groups.shape != (a.shape[-1],):
        raise ValueError('One cluster ID per original example required')
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('At least two independent clusters required')
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for i in range(repetitions):
        selected = rng.integers(len(unique), size=len(unique))
        values[i] = sums[selected].sum()/counts[selected].sum()
    low, high = np.quantile(values,[.025,.975])
    return dict(mean=float(difference.mean()), ci_low=float(low), ci_high=float(high),
                clusters=len(unique), repetitions=repetitions, bootstrap_seed=seed,
                unit='original image' if cluster_ids is None else 'supplied group',
                uncertainty='test sampling conditional on fixed trained models and angles')

"""Descriptive contrasts from seed-block measurements; no independence across angles."""
import itertools
from src.analysis.statistics import seed_summary


def interactions(values):
    """values[(architecture, regime)][seed][metric] -> float."""
    models = sorted({m for m,r in values})
    summaries, raw, augmentations, skipped = [], [], [], []
    def contrast(name, metric, terms, destination):
        if any(group not in values for group,coefficient,key in terms):
            skipped.append(dict(contrast=name, metric=metric, reason='missing condition'))
            return
        seed_sets = [set(values[group]) for group,coefficient,key in terms]
        if not seed_sets[0] or any(s != seed_sets[0] for s in seed_sets):
            skipped.append(dict(contrast=name, metric=metric, reason='unequal seed sets'))
            return
        if any(key not in values[group][s] for group,c,key in terms for s in seed_sets[0]):
            skipped.append(dict(contrast=name, metric=metric, reason='missing metric'))
            return
        diffs = [sum(c*values[group][s][key] for group,c,key in terms) for s in sorted(seed_sets[0])]
        raw.extend(dict(contrast=name,metric=metric,seed=s,value=d) for s,d in zip(sorted(seed_sets[0]),diffs))
        destination.append(dict(contrast=name,metric=metric,**seed_summary(diffs)))
    for model in models:
        for metric in ['clean_accuracy','mean_accuracy','worst_angle_accuracy','rotated_accuracy']:
            contrast(model,metric,[((model,'rotation'),1,metric),((model,'original'),-1,metric)],augmentations)
    for a,b in itertools.combinations(models,2):
        for metric in ['clean_accuracy','mean_accuracy','worst_angle_accuracy','rotated_accuracy']:
            contrast(f'augmentation effect: {a} minus {b}',metric,
                [((a,'rotation'),1,metric),((a,'original'),-1,metric),
                 ((b,'rotation'),-1,metric),((b,'original'),1,metric)],summaries)
        for regime in ['original','rotation']:
            for angle in [90,180,270]:
                contrast(f'angle effect: {a} minus {b} / {regime}',f'accuracy@{angle} minus accuracy@0',
                    [((a,regime),1,f'accuracy@{angle}'),((a,regime),-1,'accuracy@0'),
                     ((b,regime),-1,f'accuracy@{angle}'),((b,regime),1,'accuracy@0')],summaries)
    return summaries,raw,augmentations,skipped

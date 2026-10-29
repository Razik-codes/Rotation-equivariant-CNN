import pytest
from src.analysis.interactions import interactions
from src.analysis.statistics import seed_summary


def test_interactions_are_seed_block_differences():
    values={}
    for model in ('cnn','c4'):
        for regime in ('original','rotation'):
            values[(model,regime)]={}
            for seed,offset in [(42,0),(123,.05),(456,.1)]:
                clean=.5+offset
                gain=(.2 if model=='c4' else .1) if regime=='rotation' else 0
                values[(model,regime)][seed]={'clean_accuracy':clean+gain,'mean_accuracy':clean+gain,
                    'worst_angle_accuracy':clean+gain,'rotated_accuracy':clean+gain,
                    **{f'accuracy@{angle}':clean+gain-(.1 if model=='cnn' and angle else 0) for angle in [0,90,180,270]}}
    summaries,raw,augmentation,skipped=interactions(values)
    row=next(r for r in summaries if r['contrast']=='augmentation effect: c4 minus cnn' and r['metric']=='mean_accuracy')
    assert row['mean']==pytest.approx(.1)
    assert row['n']==3 and row['sd']==pytest.approx(0,abs=1e-15)
    angle=next(r for r in summaries if r['contrast']=='angle effect: c4 minus cnn / original')
    assert angle['mean']==pytest.approx(.1)
    assert not skipped
    assert len([r for r in raw if r['contrast']==row['contrast'] and r['metric']==row['metric']])==3
    del values[('c4','rotation')][456]
    summaries,_,_,skipped=interactions(values)
    assert skipped
    assert not any(r['contrast']=='augmentation effect: c4 minus cnn' for r in summaries)

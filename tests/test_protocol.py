import copy
import json
import numpy as np
import pytest
import torch
from PIL import Image
from src.common import validate_config
from src.datasets import ImageView, fixed_split, loader
from src.data.rotations import rotate, sample_angle
from src.evaluation.metrics import classification_metrics
from scripts.train_matrix import configurations


class TinyImages:
    def __init__(self, n=32):
        self.data = np.random.default_rng(1).integers(0, 256, (n,28,28), dtype=np.uint8)
        self.targets = np.arange(n) % 10
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, i):
        return Image.fromarray(self.data[i]), int(self.targets[i])


def test_split_is_fixed_disjoint_and_persisted(cfg):
    base = TinyImages()
    a = fixed_split(cfg, base)
    cfg['seed'] = 456
    b = fixed_split(cfg, base)
    assert a == b
    assert not set(a['train_ids']) & set(a['val_ids'])
    assert len(a['train_ids']) == 16


def test_rotation_and_balanced_coverage(cfg):
    x = torch.arange(25).reshape(1, 5, 5).float()
    assert torch.equal(rotate(x, 90), torch.rot90(x, 1, (-2,-1)))
    assert torch.equal(rotate(x, 360), x)
    assert rotate(x, 45).shape == x.shape
    policy = copy.deepcopy(cfg['augmentation'])
    policy['mode'] = 'balanced'
    assert set(sample_angle(policy, 42, epoch, 0) for epoch in range(4)) == {0,90,180,270}
    policy['probabilities'] = [1,0,0,0]
    policy['mode'] = 'online'
    assert all(sample_angle(policy,42,0,i) == 0 for i in range(20))


def test_workers_and_epoch_reproducibility(cfg):
    cfg['training_regime'] = 'rotation'
    base = TinyImages()
    a = ImageView(base, range(16), cfg, training=True)
    def collect(view, workers):
        local = dict(cfg, num_workers=workers)
        batches = list(loader(view, local, shuffle=True))
        return tuple(torch.cat([b[i] for b in batches]) for i in range(3))
    first, second = collect(a, 0), collect(a, 2)
    assert all(torch.equal(x,y) for x,y in zip(first, second))
    assert len(set(first[2].tolist())) == 16
    a.epoch = 1
    assert not torch.equal(first[0], collect(a,0)[0])
    val = ImageView(base, range(16), cfg)
    before = val[0][0]
    val.epoch = 100
    assert torch.equal(before, val[0][0])


def test_matrix_has_six_training_runs(cfg):
    cfg['models'] = ['cnn','c4','c8']
    assert len(list(configurations(cfg))) == 6


def test_f1_includes_absent_classes():
    m = classification_metrics([0,0,1,1], [0,1,1,1], 2)
    assert m['accuracy'] == .75
    assert m['macro_f1'] == pytest.approx((2/3 + 4/5)/2)


@pytest.mark.parametrize('model', ['cnn','c4','c8'])
def test_training_restores_checkpoint_and_evaluates_both_conditions(cfg, monkeypatch, model):
    import src.datasets as data
    from src.training.engine import train
    from src.evaluate import evaluate, load_checkpoint
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    cfg.update(model=model, training_regime='rotation')
    path = train(cfg)
    loaded, saved = load_checkpoint(path)
    x = torch.rand(2,1,28,28)
    expected = loaded(x).detach().clone()
    with torch.no_grad():
        for p in loaded.parameters():
            p.zero_()
    restored, _ = load_checkpoint(path)
    assert torch.equal(restored(x), expected)
    rows = evaluate(path)
    assert [r['angle'] for r in rows] == [0,90,180,270]
    assert all(r['n'] == 8 for r in rows)
    assert json.loads((path.parent/'metadata.json').read_text())['optimizer_steps'] == 2
    assert (path.parent/'predictions.npz').exists()
    with pytest.raises(FileExistsError):
        train(cfg)


def test_invalid_policies_rejected(cfg):
    cfg['augmentation']['probabilities'] = [.1,.1,.1,.1]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_saved_split_tampering_rejected(cfg):
    from pathlib import Path
    fixed_split(cfg,TinyImages())
    path=next(Path(cfg['data_root']).rglob('*.json'))
    payload=json.loads(path.read_text())
    payload['train_ids'][0]=999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        fixed_split(cfg,TinyImages())


def test_official_patient_overlap_rejected(cfg):
    a,b=TinyImages(),TinyImages()
    a.patient_ids=np.arange(len(a))
    b.patient_ids=np.arange(len(b))
    with pytest.raises(ValueError,match='overlap'):
        fixed_split(cfg,a,b)


def test_checkpoint_selection_uses_validation_and_earliest_tie(cfg, monkeypatch):
    import src.datasets as data
    import src.training.engine as engine
    from src.evaluate import load_checkpoint
    def raw(config, split):
        assert split != 'test', 'Training must never load test data'
        return TinyImages()
    monkeypatch.setattr(data,'raw_dataset',raw)
    scores=iter([.75,.75])
    monkeypatch.setattr(engine,'accuracy',lambda *args: next(scores))
    path=engine.train(cfg)
    _,checkpoint=load_checkpoint(path)
    assert checkpoint['step'] == 1
    assert checkpoint['validation_accuracy'] == .75


def test_cpu_training_is_repeatable(cfg, monkeypatch):
    import src.datasets as data
    from src.training.engine import train
    from src.evaluate import load_checkpoint
    monkeypatch.setattr(data,'raw_dataset',lambda config,split:TinyImages())
    first=train(cfg)
    cfg['output_dir'] += '-repeat'
    second=train(cfg)
    a, _=load_checkpoint(first)
    b, _=load_checkpoint(second)
    assert all(torch.equal(a.state_dict()[key],b.state_dict()[key]) for key in a.state_dict())


def test_aggregation_pairs_raw_runs_and_rejects_changed_ids(cfg, monkeypatch, tmp_path):
    import src.datasets as data
    from src.training.engine import train
    from src.evaluate import evaluate
    from src.analysis.aggregate import aggregate, read_runs
    monkeypatch.setattr(data,'raw_dataset',lambda config,split:TinyImages())
    folders=[]
    for seed in (42,123):
        for regime in ('original','rotation'):
            local=dict(cfg,seed=seed,training_regime=regime,seeds=[42,123])
            path=train(local)
            evaluate(path)
            folders.append(path.parent)
    output=tmp_path/'analysis'
    rows=aggregate(cfg['output_dir'],output,repetitions=20)
    assert all(r['n']==2 for r in rows)
    assert (output/'paired_seed_differences.csv').exists()
    source=folders[0]/'predictions.npz'
    with np.load(source) as arrays:
        payload={key:arrays[key].copy() for key in arrays.files}
    payload['ids'][0]=999
    np.savez_compressed(source,**payload)
    with pytest.raises(ValueError,match='mismatched ids'):
        read_runs(cfg['output_dir'])

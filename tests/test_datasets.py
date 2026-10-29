import copy
import numpy as np
import pytest
import torch
from PIL import Image
from src.datasets import ImageView, fixed_split, raw_dataset
from src.common import build_model, validate_config
from src.evaluation.metrics import classification_metrics, probability_consistency


class OfficialLike:
    def __init__(self, dataset, n=100):
        shape = (32, 32, 3) if dataset == 'cifar10' else (28, 28)
        self.data = np.random.default_rng(5).integers(0, 256, (n, *shape), dtype=np.uint8)
        self.targets = np.arange(n) % (2 if dataset == 'breastmnist' else 10)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, i):
        return Image.fromarray(self.data[i]), np.array([self.targets[i]])


def test_cifar_stratification_and_no_leakage(cfg):
    cfg.update(dataset='cifar10', validation_size=20, train_limit=None, val_limit=None)
    base = OfficialLike('cifar10')
    a = fixed_split(cfg, base)
    b = fixed_split(dict(cfg, seed=123, model='c8'), base)
    assert a == b
    assert a['class_reports']['validation']['class_counts'] == [2]*10
    assert a['class_reports']['train']['class_counts'] == [8]*10
    assert not set(a['train_ids']) & set(a['val_ids'])
    assert set(a['train_ids']) | set(a['val_ids']) == set(range(100))


def test_official_breast_split_preserved(cfg):
    cfg.update(dataset='breastmnist', validation_size=None, train_limit=None, val_limit=None)
    a = fixed_split(cfg, OfficialLike('breastmnist', 100), OfficialLike('breastmnist', 20))
    assert set(a['train_ids']) == set(range(100))
    assert set(a['val_ids']) == set(range(20))
    assert a['spec']['split_method'] == 'official'
    assert not a['class_reports']['train']['supplied_group_ids']


@pytest.mark.parametrize('dataset', ['mnist', 'cifar10', 'breastmnist'])
@pytest.mark.parametrize('model', ['cnn', 'c4', 'c8'])
@pytest.mark.parametrize('regime', ['original', 'rotation'])
def test_dataset_model_backward(cfg, dataset, model, regime):
    cfg.update(dataset=dataset, model=model, training_regime=regime)
    view = ImageView(OfficialLike(dataset), range(4), cfg, training=True)
    x = torch.stack([view[i][0] for i in range(4)])
    y = torch.tensor([view[i][1] for i in range(4)])
    model = build_model(cfg)
    logits = model(x)
    assert logits.shape == (4, 2 if dataset == 'breastmnist' else 10)
    assert x.dtype == torch.float32
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_conventional_augmentation_separate_deterministic_train_only(cfg):
    cfg.update(dataset='cifar10', conventional_augmentation={'crop_padding':4,'horizontal_flip_probability':1.0})
    base = OfficialLike('cifar10')
    view = ImageView(base, range(10), cfg, training=True)
    assert torch.equal(view[0][0], view[0][0])
    clean = ImageView(base, range(10), cfg)[0][0]
    assert not torch.equal(clean, view[0][0])
    other = copy.deepcopy(cfg); other['conventional_augmentation'] = {}
    assert torch.equal(clean, ImageView(base, range(10), other)[0][0])


def test_binary_metrics_ties_and_undefined_auc():
    p = [[.8,.2],[.6,.4],[.6,.4],[.1,.9]]
    m = classification_metrics([0,0,1,1], [0,0,0,1], 2, p)
    assert m['auroc'] == .875
    assert m['binary_f1'] == pytest.approx(2/3)
    assert m['malignant_f1'] == .8
    assert classification_metrics([0,0],[0,0],2,[[1,0],[1,0]])['auroc'] is None
    assert probability_consistency(p,p) == 0
    assert probability_consistency([[1,0]], [[0,1]]) == 2


def test_primary_protocol_guards(cfg):
    cfg['protocol']='c4-rotation'
    cfg['augmentation']['angles']=[0,45,90,135]
    with pytest.raises(ValueError, match='uniform online C4'):
        validate_config(cfg)


def test_official_api_arguments(cfg, monkeypatch):
    import src.datasets as data
    calls=[]
    monkeypatch.setattr(data.datasets, 'CIFAR10', lambda **kw: calls.append(kw))
    cfg['dataset']='cifar10'
    raw_dataset(cfg,'train'); raw_dataset(cfg,'test')
    assert [c['train'] for c in calls] == [True,False]
    import medmnist
    calls=[]
    monkeypatch.setattr(medmnist,'BreastMNIST',lambda **kw: calls.append(kw))
    cfg['dataset']='breastmnist'
    for split in ['train','val','test']: raw_dataset(cfg,split)
    assert [c['split'] for c in calls] == ['train','val','test']
    assert all(c['size']==28 for c in calls)


def test_extended_evaluation_is_separate_and_primary_agrees(cfg, monkeypatch):
    import src.datasets as data
    import json
    from src.training.engine import train
    from src.evaluate import evaluate
    from src.analysis.aggregate import read_runs, run_metrics
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: OfficialLike('mnist'))
    path = train(cfg)
    evaluate(path)
    original = (path.parent/'evaluation.csv').read_bytes()
    evaluate(path, extended=True)
    assert (path.parent/'evaluation.csv').read_bytes() == original
    with np.load(path.parent/'predictions.npz') as a, np.load(path.parent/'diagnostics/extended_rotation/predictions.npz') as b:
        assert np.array_equal(a['probabilities'], b['probabilities'][::2])
    assert len(read_runs(cfg['output_dir'])) == 1
    mixed = {'rows':[{'angle':str(a), 'accuracy':str(float(a % 90 == 0)), 'macro_f1':'0', 'degradation_from_zero':'0'} for a in range(0,360,45)]}
    assert run_metrics(mixed)['mean_accuracy'] == 1.0


def test_constant_classifier_invariant_but_inaccurate():
    probabilities = np.tile([1.,0.], (4,1))
    assert probability_consistency(probabilities, probabilities) == 0
    assert classification_metrics([1,1,1,1], [0,0,0,0], 2, probabilities)['accuracy'] == 0


def test_no_accidental_api_validation_alias(cfg):
    cfg['dataset']='cifar10'
    with pytest.raises(ValueError, match='fixed_split'):
        raw_dataset(cfg, 'val')


def test_invalid_raw_input_rejected(cfg):
    base=OfficialLike('mnist')
    cfg['dataset']='cifar10'
    with pytest.raises(ValueError,match='shape'):
        ImageView(base,[0],cfg)[0]


def test_aggregation_rejects_corrupt_saved_metric(cfg, monkeypatch):
    import csv
    import src.datasets as data
    from src.training.engine import train
    from src.evaluate import evaluate
    from src.analysis.aggregate import read_runs
    from src.common import write_csv
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: OfficialLike('mnist'))
    path=train(cfg)
    evaluate(path)
    with (path.parent/'evaluation.csv').open() as f:
        rows=list(csv.DictReader(f))
    rows[0]['accuracy']=str(1-float(rows[0]['accuracy']))
    write_csv(path.parent/'evaluation.csv',rows)
    with pytest.raises(ValueError,match='disagrees with raw predictions'):
        read_runs(cfg['output_dir'])

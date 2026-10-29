import copy
import pytest
import torch

from scripts.train_matrix import configurations
from scripts.train import validate_design
from src.common import settings
from src.evaluation.equivariance import diagnose, representation_action
from src.models.gcnn import C4GCNN


def test_seed_setup_disables_tf32_for_numerical_equivariance():
    from src.common import seed_everything, environment
    seed_everything(42)
    assert environment()['precision_settings']==dict(matmul_tf32=False,cudnn_tf32=False)


def test_study_is_complete_and_uniform():
    cfg=settings('configs/breastmnist.yaml')
    validate_design(cfg)
    matrix=list(configurations(cfg))
    assert len(matrix)==12
    assert {(c['model'],c['training_regime'],c['seed']) for c in matrix} == {
        (m,r,s) for m in ('cnn','c4') for r in ('original','rotation') for s in (42,123,456)}
    assert all(c['max_steps']==100 and c['batch_size']==128 for c in matrix)
    for key in ('models','test_angles'):
        bad=copy.deepcopy(cfg)
        bad[key]=bad[key][:-1]
        with pytest.raises(ValueError):
            validate_design(bad)


@pytest.mark.parametrize('size',[28,32])
def test_c4_intermediate_features_use_native_group_action(size):
    torch.manual_seed(71)
    model=C4GCNN(channels=(2,2,2,2)).eval()
    x=torch.randn(2,1,size,size)
    with torch.no_grad():
        ref=model.enn.GeometricTensor(x,model.input_type)
        for layer in model.features:
            ref=layer(ref)
            for k in range(4):
                g=model.gspace.fibergroup.element(k)
                assert torch.allclose(representation_action(ref,g,90*k),ref.transform(g).tensor,atol=2e-5)
        rows=[r for r in diagnose(model,x) if r['layer']!='classifier_probability_l1']
    assert len(rows)==4*len(model.features)
    assert max(r['full_error'] for r in rows)<2e-5


@pytest.mark.parametrize('size',[28,32])
def test_c4_final_output_invariance_separately(size):
    torch.manual_seed(73)
    model=C4GCNN(channels=(2,2,2,2)).eval()
    x=torch.randn(2,1,size,size)
    with torch.no_grad():
        logits=model(x)
        for k in range(4):
            assert torch.allclose(model(torch.rot90(x,k,(-2,-1))),logits,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('config_path', [
    'configs/breastmnist.yaml',
])
def test_current_and_historical_protocols_reject_changed_augmentation(config_path):
    from src.common import validate_config
    config = settings(config_path)
    config['augmentation']['probabilities'] = [.1, .2, .3, .4]
    with pytest.raises(ValueError, match='uniform online C4'):
        validate_config(config)


def test_freeze_preserves_original_inputs_and_rejects_changed_protocol(tmp_path, monkeypatch):
    import json
    import numpy as np
    from pathlib import Path
    import scripts.train as study

    class Images:
        imgs = np.zeros((2, 28, 28), dtype=np.uint8)
        labels = np.array([[0], [1]])

        def __len__(self):
            return 2

    monkeypatch.chdir(tmp_path)
    for directory in ('src', 'scripts', 'docs', 'data'):
        Path(directory).mkdir()
    Path('src/model.py').write_text('# synthetic fixture\n')
    Path('scripts/train.py').write_text('# synthetic fixture\n')
    Path('docs/methodology.md').write_text('Fixed methodology\n')
    Path('config.yaml').write_text('Synthetic fixture\n')
    Path('data/breastmnist.npz').write_bytes(b'synthetic fixture; no medical data')
    monkeypatch.setattr(study, 'raw_dataset', lambda cfg, split: Images())
    monkeypatch.setattr(study, 'environment', lambda: {'fixture': True})
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs: 'fixture==1\n')
    cfg = {'data_root': 'data'}
    destination = Path('results')
    study.freeze(cfg, 'config.yaml', destination)
    provenance = destination / 'provenance'
    before = {p.name: p.read_bytes() for p in provenance.iterdir()}
    frozen = json.loads(before['freeze.json'])
    assert set(frozen['source_hashes']) == {'src/model.py', 'scripts/train.py'}
    assert before['config.yaml'] == Path('config.yaml').read_bytes()
    assert before['protocol.md'] == Path('docs/methodology.md').read_bytes()
    study.freeze(cfg, 'config.yaml', destination)
    assert before == {p.name: p.read_bytes() for p in provenance.iterdir()}
    Path('docs/methodology.md').write_text('Changed methodology\n')
    with pytest.raises(ValueError, match='changed'):
        study.freeze(cfg, 'config.yaml', destination)
    assert before == {p.name: p.read_bytes() for p in provenance.iterdir()}

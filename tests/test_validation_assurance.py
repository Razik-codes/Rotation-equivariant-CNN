"""Scientific validation controls; no experiment training or archive writes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.evaluation.equivariance import diagnose, per_image_relative_errors, representation_action
from src.models.gcnn import C4GCNN


@pytest.fixture
def verification(tmp_path, monkeypatch, cfg):
    import scripts.train as study
    from test_protocol import TinyImages
    cfg.update(model='c4', device='cpu')
    path = tmp_path / 'best.pt'
    path.write_bytes(b'checkpoint fixture')
    payload = dict(predictions=np.zeros((4, 2), dtype=int), ids=np.arange(2))
    np.savez_compressed(tmp_path / 'predictions.npz', **payload)
    monkeypatch.setattr(study, 'load_checkpoint', lambda p: (torch.nn.Identity(), dict(config=cfg, step=1)))
    batch = (torch.zeros(2, 1, 28, 28), torch.zeros(2, dtype=torch.long), torch.arange(2))
    monkeypatch.setattr(study, 'training_data', lambda c: (None, [batch], {'hash': 'fixture'}))
    monkeypatch.setattr(study, 'raw_dataset', lambda c, split: TinyImages())
    state = SimpleNamespace(error=1e-6, calls=0, changed=False)
    monkeypatch.setattr(study, 'diagnose', lambda *a: [dict(full_error=state.error)])
    def evaluate(p):
        state.calls += 1
        if state.changed:
            np.savez_compressed(p.parent / 'predictions.npz', **dict(payload, predictions=np.ones((4, 2), dtype=int)))
    monkeypatch.setattr(study, 'evaluate', evaluate)
    return study, path, state


@pytest.mark.parametrize('error', [2e-5, 3e-5, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('repeat', [False, True])
def test_c4_rejects_invalid_error_independently_of_repeat(verification, error, repeat):
    study, path, state = verification
    state.error = error
    with pytest.raises(ValueError, match='equivariance'):
        study.verify_run(path, repeat=repeat)
    assert state.calls == 0
    assert not (path.parent / 'equivariance.csv').exists()


@pytest.mark.parametrize('repeat', [False, True])
def test_c4_accepts_below_tolerance_and_repeat_remains_optional(verification, repeat):
    study, path, state = verification
    study.verify_run(path, repeat=repeat)
    assert state.calls == int(repeat)
    checks = json.loads((path.parent / 'validation_checks.json').read_text())
    assert checks['repeated_evaluation'] == ('bitwise identical' if repeat else 'not requested')


def test_repeated_prediction_change_still_fails(verification):
    study, path, state = verification
    state.changed = True
    with pytest.raises(ValueError, match='Repeated evaluation changed'):
        study.verify_run(path, repeat=True)
    assert state.calls == 1


def regular_action(x, k):
    """Independent left action on [batch, field, orientation, y, x]."""
    return torch.rot90(x[:, :, [(j-k) % 4 for j in range(4)]], k, (-2, -1))


@pytest.mark.parametrize('k', [1, 2, 3])
def test_explicit_regular_action_and_negative_controls(k):
    model = C4GCNN(channels=(2, 2, 2, 2)).eval()
    # All axes carry distinct values; no rotational or orientation symmetry.
    x = torch.arange(2*2*4*7*7, dtype=torch.float32).reshape(2, 2, 4, 7, 7)
    feature = model.enn.GeometricTensor(x.flatten(1, 2), model.features.out_type)
    actual = representation_action(feature, model.gspace.fibergroup.element(k), 90*k)
    expected = regular_action(x, k).flatten(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
    wrong_orientation = torch.rot90(x[:, :, [(j-k-1) % 4 for j in range(4)]], k, (-2, -1))
    omitted_orientation = torch.rot90(x, k, (-2, -1))
    wrong_spatial = torch.rot90(x[:, :, [(j-k) % 4 for j in range(4)]], (k+1) % 4, (-2, -1))
    for wrong in (wrong_orientation, omitted_orientation, wrong_spatial):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(actual, wrong.flatten(1, 2), atol=2e-5, rtol=0)


def check_features(model, x):
    """Nondegeneracy makes this empirical test informative, not equivariance valid."""
    refs = []
    with torch.no_grad():
        z = model.enn.GeometricTensor(x, model.input_type)
        for layer in model.features:
            z = layer(z)
            refs.append(z.tensor.clone())
            assert torch.isfinite(z.tensor).all()
            assert (z.tensor - z.tensor[:1]).abs().max() > 1e-5, 'Uninformative constant activations'
        for k in (1, 2, 3):
            z = model.enn.GeometricTensor(torch.rot90(x, k, (-2, -1)), model.input_type)
            for layer, ref in zip(model.features, refs):
                z = layer(z)
                b, c, h, w = ref.shape
                expected = regular_action(ref.reshape(b, c//4, 4, h, w), k).reshape_as(ref)
                assert per_image_relative_errors(z.tensor, expected).max() < 2e-5
    rows = diagnose(model, x)
    assert all(row['full_error_max'] >= row['full_error'] for row in rows)
    assert max(row['full_error_max'] for row in rows) < 2e-5
    return max(row['full_error'] for row in rows), max(row['full_error_max'] for row in rows)


@pytest.mark.parametrize('size', [28, 32])
def test_asymmetric_inputs_intermediate_features(size):
    torch.manual_seed(71)
    x = torch.zeros(3, 1, size, size)
    x[0, :, 2:7, 4:9] = 1
    x[1, :, 11:15, 1:6] = 2
    x[2, :, 3:18, 19:22] = .7
    check_features(C4GCNN(channels=(2, 2, 2, 2)).eval(), x)


def test_relative_error_max_does_not_hide_one_bad_image():
    actual = torch.ones(2, 1, 4, 4)
    expected = actual.clone()
    expected[1] = 0
    errors = per_image_relative_errors(actual, expected)
    assert float(errors.mean()) == .5
    assert float(errors.max()) == 1
    # Constant features can be equivariant; nondegeneracy is a separate assertion.
    assert per_image_relative_errors(torch.zeros_like(actual), torch.zeros_like(actual)).eq(0).all()







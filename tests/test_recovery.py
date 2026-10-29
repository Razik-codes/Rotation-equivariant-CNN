import pytest
import torch

from src.training.engine import train, run_path
from src.training.recovery import TrainingInterrupted
from test_protocol import TinyImages


def checkpoint(cfg, name='last.pt'):
    return torch.load(run_path(cfg) / name, map_location='cpu', weights_only=True)


@pytest.mark.parametrize('model', ['cnn', 'c4', 'c8'])
@pytest.mark.parametrize('interrupt_at', [1, 2, 3])
def test_resumed_weights_match_uninterrupted_across_epoch_boundaries(cfg, monkeypatch, model, interrupt_at):
    import src.datasets as data
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    cfg.update(model=model, max_steps=5, validate_every=2, checkpoint_every=2,
               training_regime='rotation')
    train(cfg)
    reference = checkpoint(cfg)
    other = dict(cfg, output_dir=cfg['output_dir'] + '-resumed')
    steps = 0
    original = torch.optim.Adam.step
    def tracked(self, *args, **kwargs):
        nonlocal steps
        result = original(self, *args, **kwargs)
        steps += 1
        return result
    monkeypatch.setattr(torch.optim.Adam, 'step', tracked)
    with pytest.raises(TrainingInterrupted):
        train(other, stop_requested=lambda: steps >= interrupt_at)
    interrupted = checkpoint(other)
    assert interrupted['step'] == interrupt_at
    assert interrupted['metadata']['status'] == 'interrupted'
    train(other, resume=True)
    actual = checkpoint(other)
    assert all(torch.equal(reference['state_dict'][k], actual['state_dict'][k])
               for k in reference['state_dict'])
    for a, b in zip(reference['history'], actual['history'], strict=True):
        assert {k:v for k,v in a.items() if k != 'train_seconds'} == {
            k:v for k,v in b.items() if k != 'train_seconds'}
    assert reference['exposures'] == actual['exposures']
    assert reference['seen_ids'] == actual['seen_ids']
    assert reference['best'] == actual['best']
    assert torch.equal(reference['rng']['torch'], actual['rng']['torch'])
    for param, state in reference['optimizer']['state'].items():
        for key, value in state.items():
            assert torch.equal(value, actual['optimizer']['state'][param][key])


def test_abrupt_failure_replays_only_since_last_atomic_save(cfg, monkeypatch):
    import src.datasets as data
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    cfg.update(max_steps=5, validate_every=2, checkpoint_every=2)
    train(cfg)
    reference = checkpoint(cfg)
    other = dict(cfg, output_dir=cfg['output_dir'] + '-crash')
    original = torch.optim.Adam.step
    steps = 0
    def crash(self, *args, **kwargs):
        nonlocal steps
        result = original(self, *args, **kwargs)
        steps += 1
        if steps == 3:
            raise RuntimeError('simulated power loss')
        return result
    monkeypatch.setattr(torch.optim.Adam, 'step', crash)
    with pytest.raises(RuntimeError, match='power loss'):
        train(other)
    assert checkpoint(other)['step'] == 2
    monkeypatch.setattr(torch.optim.Adam, 'step', original)
    train(other, resume=True)
    actual = checkpoint(other)
    assert all(torch.equal(reference['state_dict'][k], actual['state_dict'][k])
               for k in reference['state_dict'])


def test_resume_rejects_changed_protocol_and_preserves_completed_checkpoint(cfg, monkeypatch):
    import src.datasets as data
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    path = train(cfg)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='configuration'):
        train(dict(cfg, learning_rate=.2), resume=True)
    assert train(cfg, resume=True) == path
    assert path.read_bytes() == before
    saved = checkpoint(cfg)
    saved['metadata']['source_hashes'] = {}
    torch.save(saved, run_path(cfg) / 'last.pt')
    with pytest.raises(ValueError, match='Source changed'):
        train(cfg, resume=True)


def test_cuda_request_never_silently_falls_back_to_cpu(cfg, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cfg['device'] = 'cuda'
    with pytest.raises(RuntimeError, match='CPU fallback is disabled'):
        train(cfg)
    assert not run_path(cfg).exists()


def test_sigterm_saves_and_restores_signal_handler(cfg, monkeypatch):
    import os
    import signal
    import src.datasets as data
    from src.training.recovery import interruption_flag
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    original = torch.optim.Adam.step
    previous = signal.getsignal(signal.SIGTERM)
    def interrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return result
    monkeypatch.setattr(torch.optim.Adam, 'step', interrupt)
    with pytest.raises(TrainingInterrupted):
        with interruption_flag() as stopping:
            train(cfg, stop_requested=stopping)
    assert checkpoint(cfg)['step'] == 1
    assert signal.getsignal(signal.SIGTERM) == previous


def test_resume_restores_stochastic_model_rng(cfg, monkeypatch):
    import random
    import numpy as np
    import src.datasets as data
    import src.training.engine as engine
    from src.common import build_model
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    class StochasticScale(torch.nn.Module):
        def forward(self, x):
            return x * (random.random() + np.random.random()) if self.training else x
    monkeypatch.setattr(engine, 'build_model', lambda config: torch.nn.Sequential(
        StochasticScale(), torch.nn.Dropout(.2), build_model(config)))
    cfg.update(max_steps=5, validate_every=2, checkpoint_every=1)
    train(cfg)
    reference = checkpoint(cfg)
    other = dict(cfg, output_dir=cfg['output_dir'] + '-stochastic')
    calls = 0
    def stopping():
        nonlocal calls
        calls += 1
        return calls == 2
    with pytest.raises(TrainingInterrupted):
        train(other, stop_requested=stopping)
    train(other, resume=True)
    actual = checkpoint(other)
    assert all(torch.equal(reference['state_dict'][k], actual['state_dict'][k])
               for k in reference['state_dict'])


def test_matrix_resume_keeps_completed_evaluations(cfg, monkeypatch, tmp_path):
    import sys
    import yaml
    import src.datasets as data
    from scripts.train_matrix import main, configurations, evaluation_complete
    monkeypatch.setattr(data, 'raw_dataset', lambda config, split: TinyImages())
    cfg['models'] = ['cnn', 'c4', 'c8']
    path = tmp_path / 'matrix.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, 'argv', ['matrix', '--config', str(path), '--resume'])
    main()
    files = [run_path(item) / 'evaluation.csv' for item in configurations(cfg)]
    timestamps = [file.stat().st_mtime_ns for file in files]
    main()
    assert [file.stat().st_mtime_ns for file in files] == timestamps
    assert all(evaluation_complete(file.parent) for file in files)
    # An interrupted/incomplete evaluation is regenerated without retraining.
    (files[0].parent / 'predictions.npz').unlink()
    main()
    assert all(evaluation_complete(file.parent) for file in files)
    predictions = files[0].parent / 'predictions.npz'
    predictions.write_bytes(predictions.read_bytes()[:100])
    assert not evaluation_complete(predictions.parent)
    main()
    assert evaluation_complete(predictions.parent)

import pytest
import torch
from src.common import settings


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture
def cfg(tmp_path):
    config = settings('tests/fixtures.yaml')
    config.update(data_root=str(tmp_path / 'data'), output_dir=str(tmp_path / 'runs'),
                  validation_size=8, train_limit=16, val_limit=8, test_limit=8,
                  max_steps=2, validate_every=1, batch_size=8)
    return config

import torch
from src.models.cnn import BaselineCNN
from src.common import accuracy


def test_cnn_output_shape():
    model = BaselineCNN()
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_c4_output_shape():
    import pytest
    pytest.importorskip('escnn')
    from src.models.gcnn import C4GCNN
    model = C4GCNN()
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)

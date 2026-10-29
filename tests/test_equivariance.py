import pytest
import torch
from src.models.gcnn import CyclicGCNN
from src.evaluation.equivariance import diagnose, representation_action


@pytest.mark.parametrize('order', [4,8])
@pytest.mark.parametrize('size', [28,32])
def test_identity_right_angles_and_representation_action(order, size):
    torch.manual_seed(7)
    model = CyclicGCNN(channels=(2,2,2,2), order=order).eval()
    x = torch.randn(2,1,size,size)
    rows = diagnose(model, x)
    for row in rows:
        assert torch.isfinite(torch.tensor(row['full_error']))
        if row['angle'] % 90 == 0:
            assert row['full_error'] < 2e-5, row
    feature = model.features(model.enn.GeometricTensor(x, model.input_type))
    element = model.gspace.fibergroup.element(order//4)
    expected = feature.transform(element).tensor
    actual = representation_action(feature, element, 90)
    assert torch.allclose(actual, expected, atol=2e-5)
    if order == 8:
        assert any(r['full_error'] > 1e-4 for r in rows if r['angle'] == 45)

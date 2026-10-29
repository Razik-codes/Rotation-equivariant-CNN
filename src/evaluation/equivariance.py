"""Feature equivariance and classifier invariance are distinct diagnostics."""
import argparse
from pathlib import Path
import torch
from src.common import write_csv, write_json
from src.data.rotations import rotate
from src.datasets import training_data
from src.evaluate import load_checkpoint


def representation_action(feature, element, angle, interpolation='bilinear'):
    """Correct ESCNN fiber action plus explicit spatial pullback on the same grid.

    This uses each feature's FieldType; it is not a hand-coded channel roll.
    Right angles are exact spatial permutations. Other angles use the same
    fixed-canvas interpolator on both sides, with zero feature-space fill.
    """
    return rotate(feature.transform_fibers(element).tensor, angle, interpolation, 0.0)


def per_image_relative_errors(actual, expected, margin=0, eps=1e-8):
    if margin:
        if 2 * margin >= min(actual.shape[-2:]):
            raise ValueError('Crop margin removes the whole feature map')
        actual = actual[..., margin:-margin, margin:-margin]
        expected = expected[..., margin:-margin, margin:-margin]
    # Keep one normalized L2 discrepancy per example, not one batch ratio.
    numerator = (actual - expected).flatten(1).norm(dim=1)
    denominator = actual.flatten(1).norm(dim=1)
    return numerator / (denominator + eps)


def relative_error(actual, expected, margin=0, eps=1e-8):
    return float(per_image_relative_errors(actual, expected, margin, eps).mean())


@torch.no_grad()
def diagnose(model, x, interpolation='bilinear', margin=1):
    if not hasattr(model, 'gspace'):
        raise ValueError('Equivariance diagnostics require a C4 or C8 model')
    model.eval()
    original = model.enn.GeometricTensor(x, model.input_type)
    reference = []
    z = original
    for layer in model.features:
        z = layer(z)
        reference.append(z)
    logits = model(x)
    rows = []
    for k in range(model.order):
        element = model.gspace.fibergroup.element(k)
        angle = 360 * k / model.order
        transformed_input = representation_action(original, element, angle, interpolation)
        z = model.enn.GeometricTensor(transformed_input, model.input_type)
        for i, layer in enumerate(model.features):
            z = layer(z)
            expected = representation_action(reference[i], element, angle, interpolation)
            full = per_image_relative_errors(z.tensor, expected)
            interior = per_image_relative_errors(z.tensor, expected, margin)
            rows.append(dict(angle=angle, layer=f'{i}:{type(layer).__name__}',
                             feature_channels=z.type.size, spatial_size=z.tensor.shape[-1],
                             full_error=float(full.mean()), full_error_max=float(full.max()),
                             interior_error=float(interior.mean()), interior_error_max=float(interior.max()),
                             interior_margin=margin))
        distances = (model(transformed_input).softmax(1)-logits.softmax(1)).abs().sum(1)
        rows.append(dict(angle=angle, layer='classifier_probability_l1', feature_channels=logits.shape[1],
                         spatial_size=1, full_error=float(distances.mean()), full_error_max=float(distances.max()),
                         interior_error=None, interior_error_max=None, interior_margin=0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--margin', type=int, default=1)
    args = ap.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    cfg = checkpoint['config']
    torch.set_num_threads(cfg['num_threads'])
    _, val, manifest = training_data(cfg)
    x, _, ids = next(iter(val))
    rows = diagnose(model, x, cfg['rotation']['interpolation'], args.margin)
    out = Path(args.checkpoint).parent
    write_csv(out / 'equivariance.csv', rows)
    write_json(out / 'equivariance_metadata.json', dict(validation_ids=ids.tolist(),
               split_hash=manifest['hash'], dtype=str(x.dtype), interpolation=cfg['rotation']['interpolation'],
               fill='zero in normalized input/feature space', margin=args.margin,
               feature_aggregation='mean of per-example normalized L2 errors',
               output_aggregation='mean per-example probability L1 distance', eps=1e-8))
    print(out / 'equivariance.csv')


if __name__ == '__main__':
    main()

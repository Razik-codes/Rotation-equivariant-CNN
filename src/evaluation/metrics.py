"""Metrics in fractions, macro F1 includes every dataset class."""
import numpy as np


def classification_metrics(labels, predictions, num_classes, probabilities=None):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError('Labels/predictions must have equal, nonempty shapes')
    if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or
            not np.issubdtype(predictions.dtype, np.integer) or
            min(labels.min(), predictions.min()) < 0 or max(labels.max(), predictions.max()) >= num_classes):
        raise ValueError('Labels/predictions must be integer class indices in range')
    confusion = np.bincount(num_classes * labels + predictions,
                            minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(num_classes, dtype=float), where=denominator != 0)
    result = dict(accuracy=float((labels == predictions).mean()), macro_f1=float(f1.mean()))
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if (probabilities.shape != (len(labels), num_classes) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
            raise ValueError('Expected normalized class probabilities')
    if num_classes == 2:
        # Official BreastMNIST label 1 is normal/benign; label 0 is malignant.
        result.update(binary_f1=float(f1[1]), malignant_f1=float(f1[0]), auroc=None)
        if probabilities is not None and len(np.unique(labels)) == 2:
            from scipy.stats import rankdata
            positives = labels == 1
            n_pos, n_neg = int(positives.sum()), int((~positives).sum())
            ranks = rankdata(probabilities[:, 1], method='average')
            result['auroc'] = float((ranks[positives].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg))
    return result


def probability_consistency(reference, transformed):
    """Mean per-image probability L1 distance; a constant classifier scores zero."""
    a, b = np.asarray(reference), np.asarray(transformed)
    if a.shape != b.shape or a.ndim != 2 or not len(a):
        raise ValueError('Matching nonempty [example, class] arrays required')
    return float(np.abs(a-b).sum(axis=1).mean())

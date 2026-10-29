# Methodology

## Design and implementation

Two architectures × no augmentation/uniform C4 augmentation × seeds 42/123/456.
The baseline CNN has four convolutional stages of widths 13/28/43/43 and 43,029
trainable parameters. The C4 G-CNN uses 7/14/28/28 regular fields, group pooling
and spatial averaging, with 38,710 parameters. Model definitions are in
[`src/models`](../src/models); the shared training engine is in
[`src/training/engine.py`](../src/training/engine.py).
The generic implementation retains C8 and other dataset support for compatibility;
no additional experiments are presented here.

The study uses the official 546/78/156 BreastMNIST training/validation/test split,
28×28 grayscale images, split seed 2026, and malignant label 0 / normal-or-benign
label 1. Convert pixels to float32 [0,1], rotate, then normalize by mean/std .5/.5.
Right-angle rotations use exact `torch.rot90` permutations. Online augmentation is
a uniform draw from 0/90/180/270 degrees keyed by seed, epoch and original image ID.
Validation is clean and deterministic. Additional crops/flips are disabled.

Adam lr=.001, betas=(.9,.999), eps=1e−8, zero weight decay, no scheduler,
cross-entropy and batch size 128. Each epoch has four batches of 128 and one of 34.
All runs used 100 updates (20 epochs; 10,920 exposures). Validate every five updates
and select the earliest maximum clean-validation accuracy. No test-driven selection.
CUDA float32 disables matmul/cuDNN TF32; matching steps does not match compute.

## Evaluation and statistics

Each selected checkpoint receives the same 156 test images at 0/90/180/270 degrees.
First average accuracy over angles within each seed, then calculate mean and sample
SD across three seeds. Paired 95% Student-t intervals use within-seed differences,
df=2 and no multiplicity correction. Matching seeds aligns data order/augmentation,
not initial weights. Rotated copies are repeated measurements of 156 images,
not 624 independent samples.

The public [measurement table](../results/per_angle_accuracy.csv) is an unchanged
copy of the archived 48 seed/angle rows. Its SHA-256 is
`b2ce3206b948640dfee24436b44bc8fb95e2da2cdea87c8a65d7692fc52f85a3`.
The two public summary tables preserve the archived `mean_accuracy` rows; the
compact analysis checks their numeric values within 1e−12. Public plots show
sample-SD bands and paired t intervals, respectively. Only this accuracy-level
reanalysis is advertised; the full prediction-level archive is not bundled.

## Numerical symmetry and instability

C4 respects the square grid for the tested 28→14→7 pooling geometry. Numerical
tests compare actual ESCNN fiber action plus spatial rotation, intermediate
features and final output invariance. Independent orientation-index controls and
wrong-action negative tests are included. Reported archived checks across six
trained C4 checkpoints and 14 layers had maximum batch-mean relative feature error
8.55e−7 with TF32 disabled (acceptance tolerance 2e−5). This reported historical
measurement cannot be rerun without the archived checkpoints and validation images;
public tests check the implementation on synthetic inputs. It is not a worst-case bound.

The original CNN/no-augmentation/456 validation accuracy was 83.33% at update 90,
44.87% at 95 and 88.46% at 100. Update 95 was not retained. Retained-state inference
and code checks found no demonstrated validation-mode, preprocessing, accuracy or
checkpoint-selection defect. BN-statistic sensitivity was observed, but no cause
of the historical collapse was established. Historical interval training loss is
training-mode CE, not clean eval-mode training loss. Convergence remains unverified.

## Commands requiring separately generated or archived inputs

From the repository root, after training or supplying compatible local weights/data:

```bash
python -m scripts.evaluate --checkpoint CHECKPOINT.pt
python -m src.evaluation.equivariance --checkpoint CHECKPOINT.pt
python -m src.analysis.aggregate --runs runs/breastmnist/breastmnist --output .reproduced/full_analysis
python -m src.analysis.plots --analysis .reproduced/full_analysis --output .reproduced/full_analysis/figures
```

`CHECKPOINT.pt` is a placeholder; the two checkpoint commands write diagnostics or
evaluation files beside it. The full aggregator needs run configurations, predictions,
metadata and matching protocols. The plotter additionally needs histories and available
diagnostic records. These inputs are produced by the main training runner but are not
part of this compact public export. The main runner also preserves the predeclared
duplicate-exclusion sensitivity in its full output. Use fresh destinations.

Historical source and development provenance are preserved in the local research
archive, without redaction or rewriting for this export. Core model/data/training/
evaluation code is copied unchanged; entry-point plumbing and the public accuracy
analysis are adapted for this smaller layout. Exact optimizer continuation from
historical weights requires the original source/configuration/environment, not this
export. No exact clean-machine installation or cross-hardware training guarantee is made.

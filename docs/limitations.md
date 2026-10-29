# Scientific limitations

- **Exploratory budget:** reconstructed code, one small dataset, three seeds and
  100 updates do not establish convergence, novelty or general model superiority.
  Large CNN validation fluctuations remain unexplained. Missing intermediate
  states prevent a causal attribution; BatchNorm is not an established cause.
- **Data integrity:** one exact train/test pixel duplicate has conflicting labels.
  The archived sensitivity excludes test ID 76 and changes four-angle means by
  less than 0.13 percentage points; that analysis is not reproduced by the compact
  public tables. It does not repair contamination or detect near duplicates.
  Patient IDs are unavailable, and earlier smoke/pilot runs accessed test prefixes.
- **Label validity:** ultrasound rotations have not been clinically validated as
  preserving diagnostic interpretation. No clinical or deployment claim is made.
- **Uncertainty:** three-seed t intervals have df=2, are unadjusted and unstable.
  The test set is imbalanced (42 malignant / 114 normal-or-benign). Rotations are
  not independent examples; seed intervals do not jointly quantify training and
  finite-test uncertainty or unknown patient dependence.
- **Comparability:** widths, parameters, operations and compute differ. Shared
  seeds align experimental blocks, not initial weights. No parameter-matched or
  longer-budget results are included. Short timing measurements are not universal
  compute comparisons.
- **Equivariance:** guarantees depend on group, grid, pooling, boundaries and
  precision. Finite batch-mean errors are not worst-case per-image bounds.
  Train-mode dropout is not pathwise invariant; accuracy alone does not establish
  symmetry, and a constant classifier may be invariant but inaccurate.
- **Reproduction scope:** the public export reproduces accuracy summaries and
  paired intervals from recorded seed/angle accuracies, not from images or raw
  predictions. Historical diagnostics, image bootstraps, F1/AUROC, loss histories,
  duplicate sensitivity and checkpoint continuation require omitted local archive
  inputs. Figures can be re-rendered but pixel-identical rendering is not promised.
  Cross-platform training equivalence and interrupted CUDA continuation are unverified.

See [methodology](methodology.md) for the precise design and available commands.
Original histories, predictions, hashes, tool/development records and frozen source
remain intact in the local archive; omission from this export does not erase those
records or make the original test holdout pristine.

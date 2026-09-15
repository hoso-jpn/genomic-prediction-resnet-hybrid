# Controlled marker-QC comparison (#29)

The existing SoyNAM pipeline-vs-pipeline comparison and defaults stay intact.
A **separate** generic-dataset run can now select shared QC:

```bash
uv run --frozen python gs_evaluate.py plan --dataset dataset.json \
  --split controlled.json --comparison-mode controlled --max-epochs 200 --seed 42
uv run --frozen python gs_evaluate.py run --dataset dataset.json \
  --split controlled.json --output-dir controlled-001
```

Use `--comparison-mode legacy` (default) for the original model-specific QC.
Never combine legacy and controlled runs into one architecture-only table.
Common marker QC does not eliminate differences in PCA, scaling, kernels,
regularization or epoch selection, and is not proof of a purely architectural
causal effect.

## Shared QC and budget

The controlled plan fixes observed rate **>= 0.9**, MAF **>= 0.05**, imputed
training variance **> 1e-6**. All-missing markers fail. `controlled_qc.py` provides
one function used by both baselines. Inner fit and outer training get separate
masks; each is fit using only that partition. Every controlled run asserts both
baselines' masks match for both stages and saves marker IDs/hash/boundaries.
Imputation statistics are train-only. Nothing is fitted to outer test genotypes.

There is one fixed ResNet hyperparameter candidate and one seed per plan, with
the recorded maximum-epoch budget. Only the saved inner validation partition
selects the epoch. The fixed-alpha=1 ridge comparator uses the same outer mask,
training mean imputation, unpenalized intercept and centered marker scale.
No outer score selects alpha, QC thresholds, seed, epoch or PCA policy.

Each run writes an adjacent `*.attempt-<id>.json` receipt with plan/configuration,
status and wall time; failures record an error and leave no completed run. A
process kill may leave `running`, which is explicitly incomplete. Successful
runs retain every model's predictions, source hashes and configuration. Reusing
an output directory fails. Manually trying multiple plans is not a validated
hyperparameter search; retain every receipt/plan and do not pick by outer score.
Nested candidate selection would require a separate bounded search contract.

## Prespecified small ablations

| Run | QC | ResNet features | Linear comparison |
| --- | --- | --- | --- |
| legacy | original per-model defaults | standardized SNP + PCA | original GBLUP |
| controlled-pca | shared | standardized SNP + training-fit PCA | GBLUP and fixed ridge |
| controlled-no-pca | shared | standardized SNP; PCA bypassed | same GBLUP and fixed ridge |

Add `--no-pca` to **plan**, save a new split file and output directory. No PCA
uses standardized SNPs in the linear path and records empty PCA arrays with
`use_pca=false`, not a dense identity matrix. Compare all prespecified runs.

GBLUP and marker ridge are the same additive model family under matched scale
and penalty. The equivalence test uses
`alpha = VanRaden denominator * (lambda_ratio + diagonal_jitter)` and verifies
numerically equal predictions. The fixed-alpha baseline is a penalty control,
not an independent architectural winner. Do not count redundant predictors as
multiple corroborating model discoveries.

## Adoption register (initial decision: defer new architectures)

| Candidate | Evidence required | Cost/resource evidence | Current decision |
| --- | --- | --- | --- |
| GBLUP | independent target-scenario correlation interval and selection diagnostics | fit/update time, dense training kernel and prediction storage | retain as primary baseline; sufficient GBLUP is a valid outcome |
| ResNet | additional effect under common QC, all seeds and independent trials | epoch budget, CPU/GPU time, inference memory | evaluate; no superiority claim |
| DPCformer / other Transformer | reproducible implementation/license, same split/QC and independent benefit | train/update budget and offline inference feasibility | defer implementation pending pilot and baseline evidence |

Every generic run records GBLUP/ResNet per-fold CPU wall time and the adoption
status. These are local execution costs, not GPU benchmarks or support for
10M-marker training. Scientific references motivate candidates, not measured
performance on this repository. Issue #13 GPU verification and #6's original
real-data baseline experiment remain prerequisites for a substantial additional
GPU comparison. This extension does not rewrite their acceptance conditions.

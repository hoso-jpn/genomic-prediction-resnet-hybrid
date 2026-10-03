# Offline GBLUP model delivery (#30)

This path separates **held-out evaluation** from **fit-final** and new-individual
prediction. Initial support is GBLUP only; no ResNet weights are exported.
Bundles are research artifacts pending a validated pilot/application scope.
Neither a successful fit nor a model file establishes GS feasibility.

## 1. Evaluate and declare use scope

Create a generic individual dataset and complete a saved scenario evaluation as
in [gs-dataset-contract.md](gs-dataset-contract.md). The baseline QC configuration
is fixed by its saved split. A controlled evaluation uses its recorded common
thresholds; a legacy evaluation preserves the original GBLUP thresholds.

Create `scope.json` explicitly, for example:

```json
{
  "cohort_ids": ["training-cohort", "next-cohort"],
  "generations": ["pilot-generation-1"],
  "update_trigger": "Review before a new season, generation, or target population",
  "rollback_bundle": "none_initial"
}
```

These are operator-declared application boundaries, not statistically inferred
out-of-distribution detection or proof those cohorts have been validated. Unknown
cohorts/generations fail prediction. Extend a scope only after checking the
application population; a new scope/bundle should preserve the prior bundle for
rollback. `none_initial` explicitly means no earlier bundle exists.

## 2. Refit the full training set

```bash
uv run --frozen python gs_model_bundle.py fit-final \
  --dataset dataset.json --evaluation evaluation-001 \
  --scope scope.json --prediction-qc prediction-qc.json --output-dir model-001
```

New bundles use schema **v2**. Freeze sample QC at fit-final; no threshold is
estimated from a prediction batch. An illustrative **synthetic-only** policy is:

```json
{
  "min_retained_marker_call_rate": 0.9,
  "basis": "Synthetic demonstration; select a customer threshold in the pilot",
  "evidence_references": []
}
```

This is not a recommended commercial cutoff. The inclusive threshold is in
`(0, 1]` and counts only this model's retained markers. Evidence references record
the operator's basis; they are not approval verification. If omitted, the
conservative research fallback requires **all retained markers** (`1.0`); it is
not pilot validated. Zero observed retained markers always means withholding,
including samples with calls only at filtered markers.

The evaluation split, metadata, predictions, feasibility report and preprocessing
artifacts must exist. Dataset identity and canonical plan/configuration are
verified. Evaluation metadata binds the exact plan hash and the checksums of the
split, predictions, feasibility JSON and preprocessing archive, so artifacts from
different runs cannot be mixed accidentally. Evaluations created before this
binding was added must be regenerated. Their checksums are stored and checked
again after fitting. The full
supplied training observation set is then fitted with the chosen GBLUP QC;
training-mean imputation, marker selection, allele frequency/VanRaden scale and
REML are computed once. Schema v1 datasets retain observation rows and their
implicit weights. Schema v2 line datasets reuse their declared within-trial
line means and equal line weights; fit-final does not change the aggregation.
For line models, the hashed bundle carries phenotype processing and binds the
evaluation's line-candidate/processing artifacts as well.

GBLUP's dual coefficients are converted to marker coefficients:
`beta = centered_training_genotypes.T @ dual_coefficients / denominator`.
Prediction is then `intercept + centered_new_genotypes @ beta`. This needs no
training genotype matrix, phenotype table or evaluation directory at prediction
time. The coefficients still derive from training data and must receive the same
protection; the manifest also contains training IDs and provenance. Do not assume
a model is anonymous or freely redistributable.

The directory contains:

- `bundle.json`: schema, trait/unit/direction, marker/allele order, reference,
  encoding, fitted settings, scope, model card, evaluation/source checksums.
- `parameters.npz`: bool marker mask, float64 training means and marker effects;
  no pickle, executable model object or runtime deserialization of Python code.
- `model-card.md`: human-readable status and boundaries; authoritative scope is
  inside the hashed manifest.

Files are staged in a private temporary directory and published after validation.
Existing bundles are refused. Save/load predictions must agree at `atol=rtol=1e-10`
in the build environment. NPZ uses lossless float64 storage; cross-platform
numerical changes remain subject to the recorded dependency environment.

## 3. Predict with genotype files only

A new panel follows the existing producer contract. Its reference FASTA checksum
and declared species/assembly must match the bundle. Create `context.json`:

```json
{
  "species": "Vigna angularis",
  "assembly_id": "GCF_016808095.1",
  "cohort_id": "next-cohort",
  "generation": "pilot-generation-1"
}
```

```bash
uv run --frozen python gs_model_bundle.py predict \
  --bundle model-001 --panel-dir next-panel --context context.json \
  --output predictions-001.csv
```

No phenotype is read. Prediction performs no imputation/statistic/PCA/model fit,
uses no network, and changes none of the training/evaluation/model files. Missing
calls use the **stored** mean if sample QC passes. Output includes sample ID, trait, unit, prediction,
bundle hash, and whether a permitted reorder occurred.

For a schema v2 line dataset's model, add `"target_trial": {"year": 2025,
"site": "A"}` to the context, using the exact trial declared at training. Other
years/sites are rejected; this model predicts a within-trial raw line mean, not
environment-adjusted genetic value. The prediction CSV still uses new genotype
sample IDs, without inventing line IDs or merging new samples. Target and the
training phenotype-processing contract are carried into the delivery receipt.

The default preserves the input sample order and verifies it against panel
metadata. To assert a specific requested output order, pass `--sample-ids ids.txt`
(one opaque ID per line). A sample permutation or a marker permutation is
rejected unless `--allow-reorder` is explicitly supplied; sets must match exactly.
Reordering maps by ID. Unknown encoding, schema or checksum mismatch, missing or
extra markers (even training-filtered markers), allele/strand flips, reference
mismatch and out-of-scope context fail. No liftover or allele harmonization is
attempted. Existing prediction output files are never overwritten.

### Sample status and delivery artifacts (#41)

The CSV adds `prediction_status` (`available` / `withheld`), `reason_codes`,
`observed_retained_markers`, and `retained_marker_call_rate`. Withheld predictions
are empty/NaN and must be excluded from selection ranking. Reasons distinguish
zero observed markers, call rate below the fixed minimum, unavailable legacy QC,
and rejected model status. Availability means a point prediction passed this QC,
not permission for customer deployment.

The command also publishes `OUTPUT.csv.withheld.csv` for re-genotyping/review and
`OUTPUT.csv.metadata.json` as a final checksum-bound receipt. A delivery is complete
only when the receipt exists and both CSV checksums agree. Publication refuses an
existing artifact and rolls back files created by a failed publication. A process
kill before the final receipt may leave incomplete files; do not treat them as a
completed delivery or overwrite them on retry.

CSV and receipt preserve model status, declared scope, validation scenario,
unverified conditions (in the model card), and point-prediction-only limitations.
`customer_release_authorized` is always false here. There is no new approval
authority: an optional historical `review_reference` is carried through as text,
not verified acceptance. New fits remain `research_only`; conditional or rejected
states in a valid bundle are retained, and rejected models yield no predictions.
Customer acceptance remains in the readiness review/validation workflow.

Schema v1 bundles remain readable, but do not contain an adopted sample-QC
policy: all their predictions are withheld with `legacy_qc_policy_unavailable`.
Generate a new v2 bundle with explicit policy/evidence before using this delivery
path. A QC policy cannot silently be inserted into a schema v1 bundle. Allele,
reference, checksum, offline and no-refit checks continue to apply.

Checksums detect accidental changes relative to the declared manifest, not an
authentic origin: someone who can rewrite files and recompute every hash can
create a different bundle. There is no signature infrastructure in this version.

## Model card interpretation

`fit_final_is_oof=false`: fit-final is not cross-validation and does not replace
OOF scores. All current synthetic examples have `real_adzuki_performance=unverified`
and `status=research_only`. The card records the evaluation scenario, allowed
cohorts/generations, unverified conditions, update trigger and rollback reference.
Only point predictions are returned; seed variation is not a calibrated prediction
interval. Deployment still requires independently held-out target-population
validation and expert confirmation of the declared application scope.

Training and prediction remain dense in-memory operations, bounded by loader
admission checks. This bundle path does not claim large-scale or GPU validation.

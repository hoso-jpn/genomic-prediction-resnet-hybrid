# Individual GS dataset contract v1 (#27)

The existing SoyNAM adapter and CLI are retained. The new `gs_dataset.py`
adapter consumes the existing GS panel loader; `gs_evaluate.py` connects its
aligned arrays to the same GBLUP and ResNet implementations. Arbitrary named
continuous traits are supported. Real adzuki accuracy remains **unverified**.
GWAS summary statistics are not individual training data.

## Dataset manifest

Paths are relative to the directory containing `dataset.json`. Referenced files
must remain inside that directory. Checksums use each file's unique basename.
No data are downloaded and no W&B logger is started by this route.

```json
{
  "schema_version": 1,
  "kind": "gs_individual_dataset",
  "species": "Vigna angularis",
  "cohort_id": "cohort",
  "panel_manifest": "panel/cohort.gs_panel.manifest.json",
  "sample_metadata": "samples.tsv",
  "phenotypes": "phenotypes.tsv",
  "reference": {
    "assembly_id": "GCF_016808095.1",
    "fasta_file": "reference.fa",
    "fasta_sha256": "sha256:<64 lowercase hex characters>"
  },
  "effect_allele": "ALT",
  "trait": {
    "name": "seed_mass", "unit": "g", "direction": "higher",
    "type": "continuous", "method": "weigh", "scale": "mass"
  },
  "aggregation": {"method": "none", "measurement_type": "raw"},
  "checksums": {
    "cohort.gs_panel.manifest.json": "sha256:<hash>",
    "samples.tsv": "sha256:<hash>",
    "phenotypes.tsv": "sha256:<hash>"
  }
}
```

The reference checksum must match the producer manifest's reference FASTA
entry. This checks identity, not the biological truth of an assembly declaration.
Known defaults are `GCF_016808095.1` and `synthetic-v1`. Other crops/assemblies
require explicit `--expected-assembly ACCESSION` after the operator verifies the
reference; unknown assemblies are never silently accepted. No liftover, strand
flip, REF/ALT flip, marker substitution or unit conversion is performed. Variant
keys must agree with chrom/position/REF/ALT metadata. Only diploid biallelic SNP
hard calls with ALT dosage are supported.

## Tables (tab separated, exact columns, IDs are opaque strings)

`samples.tsv`: `sample_id`, `line_id`, `family_id`. Sample IDs must match the full
panel exactly, independently of row order. Each line belongs to one family.

`phenotypes.tsv`: `observation_id`, `sample_id`, `trait`, `unit`, `year`, `site`,
`replicate`, `value`. Observation IDs and `(sample_id, trait, year, site,
replicate)` keys must be unique; duplicate line/trait/year/site/replicate keys
are also refused. `year` is a four-digit integer. Required identifiers cannot be
blank. Trait and unit must match the manifest exactly.

Repeated observations are retained explicitly, without averaging. Each joins to
a genotype by sample ID; the baseline's row ID is the observation ID. Missing
values use only empty text, `NA` or `nan`. Their observation IDs and genotype
samples with no usable phenotype are recorded as exclusions. All-missing traits,
nonfinite nonmissing values and unmatched phenotype sample IDs fail.

Only raw measurements with `aggregation.method=none` are currently supported.
BLUE, BLUP and preaggregated values are rejected explicitly. A future adapter
must record the correction fit set and apply corrections within each split;
precorrection using outer test phenotypes is not permitted. Repeated observations
are not independent genetic samples and should not be counted as such in a
feasibility claim.

## Offline evaluation

```bash
uv run --frozen python gs_evaluate.py plan --dataset dataset.json --split split.json
OMP_NUM_THREADS=1 uv run --frozen python gs_evaluate.py run \
  --dataset dataset.json --split split.json --output-dir evaluation-001
```

The initial scenario is family LOFO, with at least three families and a whole
inner validation family. Model configuration and all partitions are fixed before
evaluation. Both baselines consume the saved plan. No seed/threshold is selected
using outer scores. Files, phenotype, metadata, trait and aggregation settings
are checksummed into dataset/split/run provenance and rechecked after evaluation.

Outputs are generic `predictions.csv` (trait/unit and observation/sample/line IDs),
`split.json`, `metadata.json`, and `preprocessing.npz`. Existing output directories
are refused. A staging directory is published only after every artifact succeeds.
These are held-out evaluation outputs, not a deployable fitted model. Existing
SoyNAM output column names and scripts remain compatible.

The input budget also checks expanded observation-by-marker storage. Training
still requires additional dense copies and kernels: this is a small/medium
input path, not end-to-end large-panel support.

## MIAPPE correspondence

| This contract | MIAPPE concept |
| --- | --- |
| species / reference | biological material and identification context |
| sample_id / line_id | observation unit / biological material links |
| year / site / replicate | study and observation-unit design context |
| trait.name / method / scale / unit | observed variable: trait, method, scale |
| observation_id / value | linked measurement |

This is an explicit correspondence, **not full MIAPPE validation or certification**.
See [MIAPPE specifications](https://www.miappe.org/). Study identifiers, complete
experimental design and ontology exports would require additional fields.

## Deployment scenarios and selection diagnostics (#28)

`gs_evaluate.py plan --policy policy.json ...` saves schema v2 partitions with
explicit `train`, `test`, inner `fit` and `validation` observation IDs. `run`
consumes these partitions for both baselines, including ResNet epoch selection.
The plan hash covers policy, data identity, target definition, all partitions and
model configuration. Reordering or adding a test ID to training invalidates it.

Default `family_lofo` is preserved. The original SoyNAM `evaluation_split.py`
contract is untouched. New plan policies:

```json
{
  "scenario": "future_year", "selection_fraction": 0.25,
  "test_year": 2026, "validation_year": 2025,
  "genotype_status": "known", "environment_status": "known"
}
```

All outer training years precede test; validation is the latest training year,
and inner fit precedes validation. `known` requires every queried line/site to
occur earlier; `new` requires disjoint lines/sites. The same constraint applies
to inner selection. Thus known-line prediction may legitimately use that line's
past observations, while new-line prediction rejects this overlap. Data after the
test year and all other unused observations are explicitly excluded.

```json
{
  "scenario": "family_environment", "selection_fraction": 0.25,
  "test_families": ["F3"], "test_sites": ["C"],
  "validation_families": ["F2"], "validation_sites": ["B"]
}
```

Test contains the declared family/site intersection. Outer training excludes the
**union** of held-out families and sites. Inner selection applies the same rule.
Cross cells are recorded as exclusions, not quietly used for training. This
models new families **and** new environments; it is not an additive G×E model.
Generation holdout can be represented by an explicitly curated family/cohort
partition; no unrecorded generation is inferred from a year or ID.

`feasibility.json` and its Markdown summary carry scenario, population,
Pearson/RMSE/Spearman, exact-k overlap and observed selected-mean minus overall
mean. Direction is declared; a lower-is-better trait also reports a sign-adjusted
differential. `k=ceil(n*fraction)`. Spearman uses average ranks; boundary ties
for exact-k use ascending opaque observation IDs. Missing pairs are counted and
excluded. Constant predictions/targets, fewer than two pairs, and all-missing
inputs produce unavailable ranking metrics, not a spurious favorable score.

**Selection unit is the observation**, explicitly, not an automatically averaged
breeding line. Repeated measurements retain their IDs. For actual line selection,
a trial-specific aggregation policy and fold-aware phenotype model are still
required; this implementation does not silently average test replicates.

Group bootstrap uses family for LOFO/family+environment and line for future-year
prediction. It resamples whole groups including repetitions; fewer than three
groups or too many undefined resamples yield no interval. The interval is
conditional on the measured year/environments, not an uncertainty estimate for
new environments, individual prediction intervals or future genetic gain.

Reports return `no-go` for insufficient evaluable/group evidence and otherwise
`conditional` pending independent trials and expert review. They never auto-assert
Go, real adzuki performance or yield improvement. Go requires a separate pilot
with prespecified acceptance criteria, available individual data and a confirmed
application population. Review before each season/generation/population change;
collect independent target-year/environment observations as the next trial.
Different scenarios remain separate runs/reports and are not pooled into a
cross-scenario superiority ranking. #6 remains the original SoyNAM experiment.

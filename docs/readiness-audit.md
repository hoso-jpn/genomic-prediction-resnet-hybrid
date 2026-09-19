# 再現性整備の受入状況（Issue #1）

**監査日**: 2026-09-19（UTC） / **確認基準**: `main@39ce3027bc3440545377d108e9a29714bdb45818`

親Issue #1の原文（進捗5項目・対応内容42項目・完了条件8項目、計55項目）を元の順序のまま照合した表です。
チェック欄やIssueのclosedだけを根拠に充足とはせず、実装・テスト関数・CIまたは実機検証の証跡を対応付けています。
リンクは確認基準SHAに固定しています。

判定の区分:

| 判定 | 意味 |
|---|---|
| 充足 | 実装を確認し、テスト・CIまたは実機証跡で裏付けられた |
| 未充足 | 実装または証跡が無い |
| 未確認 | 本棚卸しの範囲では裏付けを確認できなかった（実装の一部だけが確認できた場合を含む） |
| 実装方式置換・記述更新必要 | 要件は満たすが、原文が想定した方式と実装が異なり、原文側の記述更新が必要 |

集計: **充足 52 / 未確認 1 / 実装方式置換・記述更新必要 2 / 未充足 0**（全55項目）。


## 進捗

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] PR #2：再現可能なPython環境、リーク防止LOFO評価、GBLUP／ResNetベースライン、README、CI | 充足 | PR #2（squash merge `61bb937`） | CI passed（コメント 2026-08-11） | LOFO評価とベースライン導入時点の記録。実データ性能の保証は含まない |
| [x] #3：Docker Composeで検証済みGBLUP・ResNet実行経路を整備する | 充足 | PR #8 / `docker-compose.yml` | CI job「Docker Compose build and smoke」 | 検証済み経路のみ。実データserviceはCI対象外 |
| [x] #4：genotype・phenotype・marker ID整合性検証（PR #7で完了） | 充足 | PR #7 / [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py)（`SoynamDataTest`） | 検証済みloader経路の話で、legacy前処理全体を意味しない |
| [x] #5：split・前処理条件・実行メタデータの成果物保存 | 充足 | PR #9 / [`run_manifest.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/run_manifest.py) | [`tests/test_run_manifest.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_run_manifest.py) | 6ファイル構成のrun artifacts |
| [ ] #6：GPU本実験とGBLUP／ResNet比較 | 充足 | PR #38（merge commit `39ce302`）／実験commit `1b59f3b` | [実測レポート](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/experiments/issue-6-soynam-1.6.2.md) §10、Issue #6 完了記録（`issuecomment-5740311746`） | **今回更新**。共通LOFO 39 fold・5,142個体・GBLUP＋ResNet 3 seed。制約はレポート §8 |

## 対応内容

### 1. Docker実行環境の修正

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] Dockerイメージに必要なソースコードを配置する | 充足 | [`Dockerfile`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/Dockerfile) | CI「Verify the image contains the sources」（`test -d /workspace/tests`） | — |
| [x] `docker-compose.yml`の各サービスから対象スクリプトを実行できるようにする | 充足 | [`docker-compose.yml`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docker-compose.yml) | CI「Docker Compose build and smoke」 | — |
| [x] CPU環境で実行できる最小smoke testを用意する | 充足 | [`tests/test_cpu_smoke.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_cpu_smoke.py)／`cpu-smoke` service | CI job で実行 | 3 family syntheticのみ |
| [x] GPU依存処理とCPUで確認可能な処理を分離する | 充足 | 既定profile（CPU）と `--profile gpu`（[`Dockerfile.cuda`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/Dockerfile.cuda)） | CI はCPU経路のみ実行 | GPU経路はCI未実行。実機証跡は [`docs/gpu-verification.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/gpu-verification.md) |
| [x] Dockerを使用した実行手順をREADMEへ記載する | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) Docker節 | — | — |

### 2. 依存関係の明示

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] Python依存関係を固定または再現可能な範囲で制約する | 充足 | [`pyproject.toml`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/pyproject.toml)／`uv.lock`（`requires-python == 3.11.*`） | CI は `uv sync --frozen` | パッチ版完全固定ではなく3.11系列のprefix指定 |
| [x] Docker経路の依存関係インストールをrequirements.txt／pipからpyproject.toml・uv.lockへ統一する（#3） | 充足 | [`Dockerfile`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/Dockerfile)（`uv sync --frozen --extra gblup --dev`） | CI Docker job | ルート `requirements.txt` は存在しない |
| [x] GBLUPベースラインで不要なR・rpy2・sommerをDocker経路から除外する（#3） | 充足 | [`pyproject.toml`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/pyproject.toml)（`gblup = ["scipy>=1.17.1"]` のみ） | CI「Verify the image does not depend on rpy2/R」（`find_spec("rpy2") is None`） | — |
| [x] SoyNAM抽出用途に限られる`rpy2`（`soynam` optional dependency）を、検証済み実行経路から分離した状態のまま維持する | 実装方式置換・記述更新必要 | `rpy2` と `soynam` optional extra は**廃止**。canonical dataset生成は [`soynam_cran.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_cran.py) が [`scripts/build_soynam_canonical.R`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/scripts/build_soynam_canonical.R) を**外部プロセスのRscriptとして起動**する方式へ置換。R環境は [`environments/soynam-linux-64.lock`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/environments/soynam-linux-64.lock)（conda explicit lock、R 4.5.3 / lme4 2.0-6 / Matrix 1.7-5）で分離 | [`tests/test_soynam_cran.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_cran.py)／[`tests/test_soynam_cran_integration.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_cran_integration.py)、CIのrpy2非依存チェック | **親Issueの原文（`soynam` optional dependencyとして維持）とは方式が異なる。** rpy2は再導入しない。学習・評価イメージはR非依存だが、**リポジトリ全体でRが不要という意味ではない**（canonical dataset生成にはRscript環境が必要） |

### 3. 入力ファイルとIDの整合性検証

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] ファイル名の`sorted()`と`zip()`だけに依存した対応付けを廃止する（PR #7） | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) `list_family_files()` | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_pairing_mismatch_is_rejected` / `test_load_with_fixed_family_files_ignores_later_directory_changes` | — |
| [x] family IDまたは明示的なmanifestによってファイルを対応付ける（PR #7） | 充足 | `list_family_files()`（family ID対応付け） | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_loads_multiple_families` | — |
| [x] genotypeとphenotypeのsample ID一致を検証する（PR #7） | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py)（RIL sample集合の照合） | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_phenotype_only_sample_is_rejected` / `test_genotype_only_sample_is_rejected`（`"RIL sample sets differ"` を送出し、`phenotype_only=[...]` / `genotype_only=[...]` に差分IDが載ることをassert） | 空ID・sample列不在は別の検査（`test_phenotype_empty_sample_id_is_rejected` / `test_genotype_header_without_sample_columns_is_rejected`）で、集合不一致の根拠ではない |
| [x] marker IDの集合と順序を検証する（PR #7） | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_marker_set_mismatch_is_rejected` / `test_marker_order_mismatch_is_rejected` | 集合と順序の双方を検証 |
| [x] IDの重複、欠落、余剰サンプルを検出する（PR #7） | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) | 重複: [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_phenotype_duplicate_sample_id_is_rejected` / `test_genotype_duplicate_sample_header_is_rejected` / `test_marker_id_duplicate_is_rejected`。欠落・余剰: `test_phenotype_only_sample_is_rejected` / `test_genotype_only_sample_is_rejected` | — |
| [x] 不整合時には対象IDを含む明確なエラーを返す（PR #7） | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_phenotype_only_sample_is_rejected` / `test_genotype_only_sample_is_rejected`（例外メッセージに差分IDを含むことをassert）、`test_missing_phenotype_column_is_rejected`（欠落列名とファイル名を含むことをassert） | — |
| [x] 入力スキーマまたはサンプルmanifestの仕様を文書化する（PR #7） | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) 入力データ節／[`docs/gs-dataset-contract.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/gs-dataset-contract.md) | — | — |

### 4. 欠損値・imputation・MAF処理の修正

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] 欠損値をヘテロ接合の符号`0`と区別して保持する | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py)（欠損は `NaN` 保持、ヘテロ `0` と別） | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_raw_loader_excludes_parent_and_preserves_missing` | — |
| [x] 欠損率をsample単位・marker単位で計算できるようにする（PR #24） | 充足 | PR #24 / [`input_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/input_qc.py) | [`tests/test_input_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_input_qc.py) `test_default_retains_all_and_audits_missingness` | marker欠損率はsample QC前の記述統計で、marker選択には使わない |
| [x] 欠損率フィルターの閾値を設定可能にする（PR #24） | 充足 | PR #24 / `--max-sample-missing-rate`・`--min-marker-observed-rate` | [`tests/test_input_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_input_qc.py) `test_sample_threshold_is_inclusive_and_keeps_alignment` / `test_invalid_threshold_fails` / `test_eliminating_a_family_fails` | 閾値境界は下記「欠損率ポリシー」参照 |
| [x] MAFを欠損値を除外して計算する | 充足 | [`gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gblup_baseline.py) legacy経路は `np.nanmean(candidate_train)` から allele frequency を算出、[`controlled_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/controlled_qc.py) `marker_mask()` は `nansum / 観測数` を使用。いずれも欠損を除外し学習fold内で計算する | **実装読み取りによる確認**。MAFの算出値そのものをassertするテストは確認できなかった | 下記「MAF処理の根拠範囲」を参照 |
| [x] MAFフィルターの閾値を設定可能にする | 未確認 | 関数引数 `gblup_baseline.prepare_fold_relationships(maf_threshold=...)`（`0.0 <= maf_threshold < 0.5` を検証）、`resnet_baseline.ResNetConfig.maf_threshold`（既定0.01）、[`controlled_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/controlled_qc.py) `marker_mask(maf_threshold=...)` | **CLIフラグは存在しない**（`--maf-threshold` はどのスクリプトにも無い。`input_qc.add_arguments()` が公開するのは `--max-sample-missing-rate` と `--min-marker-observed-rate` のみ）。閾値を変えてMAF起因でmarkerが除外されることをassertするテストも確認できなかった（MAFを渡す既存テストは `maf_threshold=0.0`、または0.05でもMAF起因の除外をassertしていない） | **判定を充足から未確認へ変更した。** 実装上は関数引数・configフィールドとして設定可能だが、原文の「設定可能」がCLIを含むかを本棚卸しでは確定できない。今回はテストもCLIも追加していない。下記「MAF処理の根拠範囲」を参照 |
| [x] imputation方法を明示し、処理内容をログへ記録する | 充足 | `preprocessing.json` の `"imputation": "training_mean"` | [`tests/test_gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gblup_baseline.py) `test_fold_preprocessing_record_matches_relationships`、[`tests/test_resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_resnet_baseline.py) `test_build_transform_record_matches_fitted_transform` | — |
| [x] imputationや標準化のパラメータを学習データだけで推定する | 充足 | [`resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/resnet_baseline.py) `fit_feature_transform()` / [`gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gblup_baseline.py) | [`tests/test_resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_resnet_baseline.py) `test_feature_statistics_are_fitted_on_training_rows_only`、[`tests/test_gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gblup_baseline.py) `test_vanraden_relationship_uses_training_statistics` | ここでの「学習データだけでfit」は**モデル側のmarker選択・imputation・標準化・PCA等**を指す |

### 5. 評価リークの防止

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] train/validation/test分割後に前処理をfitする | 充足 | [`resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/resnet_baseline.py)／[`gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gblup_baseline.py) | [`tests/test_resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_resnet_baseline.py) `test_outer_test_phenotypes_do_not_affect_predictions`、[`tests/test_gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gblup_baseline.py) `test_test_genotypes_do_not_change_training_preprocessing` / `test_test_phenotypes_do_not_affect_predictions` | — |
| [x] 同一sampleが複数splitへ混入しないことを検証する | 充足 | [`gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gs_scenarios.py) `_fold()` が **observation_id** について「train/test」「fit/validation」「fit/test」「validation/test」の4組の非重複を検査し、あわせて **fit ∪ validation ⊆ train** を検査する。SoyNAM経路は [`evaluation_split.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/evaluation_split.py) の固定LOFOで各familyを1回だけouter testにする | [`tests/test_gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gs_scenarios.py) `test_family_environment_cross_cells_are_excluded`、[`tests/test_comparison.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_comparison.py) `test_fixed_split_roundtrip_and_tampering`。#6実測で5,142個体が各1回だけouter testへ出現 | `_fold()` が見るのは上記4組の非重複と内側が外側trainの部分集合であることで、**fit・validationは train の部分集合であるため4集合すべてが互いに排他的という意味ではない**。family / line / site の条件は各シナリオの構築処理と `_status_check()` が担当する。また、異なるfold間でtrain集合が重なるのはCVの正常な性質で、不具合ではない |
| [ ] familyまたは集団構造を考慮したgroup-aware CVを選択可能にする | 実装方式置換・記述更新必要 | **GS経路**: [`gs_evaluate.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gs_evaluate.py) `--policy`（plan時）が [`gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gs_scenarios.py) の3シナリオを選ぶ。**SoyNAM経路**: [`evaluation_split.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/evaluation_split.py) は `fixed_lofo_plan` のみを許可し、別CV方式を選ぶCLIは無い（#28受入条件「既存LOFOは維持」） | [`tests/test_gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gs_scenarios.py) `test_known_line_history_is_allowed_and_time_is_strict` / `test_family_environment_cross_cells_are_excluded` / `test_policy_cli_is_recorded`、[`docs/gs-dataset-contract.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/gs-dataset-contract.md) | シナリオごとの内訳は下記「GSシナリオの分割条件とbootstrap単位」を参照。#28（closed / PR #33）で具体化。新しい選択CLIは追加していない |
| [x] random seedを設定・記録する | 充足 | `--seed`／`fold_seed = seed + fold_index * 100` | `metadata.json` の `seed`、[`tests/test_resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_resnet_baseline.py) `test_validation_family_selection_is_deterministic` | — |
| [x] 使用したsplitと前処理条件を成果物として保存する | 充足 | [`run_manifest.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/run_manifest.py)／`split-plan.json`・`preprocessing.json` | [`tests/test_run_manifest.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_run_manifest.py) | — |
| [x] GBLUP等のベースラインと同一splitで比較できるようにする | 充足 | [`compare_baselines.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/compare_baselines.py)（`--split-file` を両baselineが消費） | #6実測で4 runが同一 `split_plan_hash sha256:a9ae6cde…` を消費 | — |

### 6. 科学的ドキュメントの修正

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] 「Linear PathはRR-BLUP相当」という説明を修正する | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) Linear Path節 | 該当節を読んだ文書レビュー（2026-09-19、`main@39ce3027bc3440545377d108e9a29714bdb45818`）。CIのRuff・単体テストはコード品質の検査で、説明文の科学的妥当性は保証しない | RR-BLUP相当という記述は削除済み |
| [x] Linear Pathを正則化線形予測器として説明する | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) | 該当節を読んだ文書レビュー（2026-09-19、`main@39ce3027bc3440545377d108e9a29714bdb45818`）。CIのRuff・単体テストはコード品質の検査で、説明文の科学的妥当性は保証しない | 正則化線形予測器として記述 |
| [x] GBLUPを独立したベースラインとして説明する | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md)／[`docs/comparison-experiment.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/comparison-experiment.md) | 該当節を読んだ文書レビュー（2026-09-19、`main@39ce3027bc3440545377d108e9a29714bdb45818`）。CIのRuff・単体テストはコード品質の検査で、説明文の科学的妥当性は保証しない | 独立baselineとして記述 |
| [x] 実装済み機能とexperimentalな機能を区別する | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) 機能表／[`legacy_guard.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/legacy_guard.py)（`--allow-legacy`），[`tests/test_legacy_guard.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_legacy_guard.py) | 該当節を読んだ文書レビュー（2026-09-19、`main@39ce3027bc3440545377d108e9a29714bdb45818`）。CIのRuff・単体テストはコード品質の検査で、説明文の科学的妥当性は保証しない | experimentalは検証済み経路と区別 |
| [x] 検証済み範囲、既知の制約および未検証事項をREADMEへ記載する | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) 既知の制約節／[`docs/experiments/issue-6-soynam-1.6.2.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/experiments/issue-6-soynam-1.6.2.md) §8 | 該当節を読んだ文書レビュー（2026-09-19、`main@39ce3027bc3440545377d108e9a29714bdb45818`）。CIのRuff・単体テストはコード品質の検査で、説明文の科学的妥当性は保証しない | #6の制約を反映済み |

### 7. 自動テストとCI

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] 小規模なsynthetic fixtureを追加する | 充足 | [`tests/fixtures/`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/fixtures/)（`gs_panel_producer_v2` ほか） | CI 単体テスト | syntheticのみ。個体別実データは含まない |
| [x] 正常な前処理を確認するテストを追加する | 充足 | [`tests/test_input_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_input_qc.py)／[`tests/test_controlled_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_controlled_qc.py) | CI 単体テスト | — |
| [x] sample ID不一致を検出する負のテストを追加する | 充足 | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_phenotype_missing_sample_id_is_rejected` ほか | CI 単体テスト | — |
| [x] marker IDまたは順序の不一致を検出する負のテストを追加する（PR #7） | 充足 | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_marker_set_mismatch_is_rejected` / `test_marker_order_mismatch_is_rejected` | CI 単体テスト | — |
| [x] 重複IDと必須列欠落を検出する負のテストを追加する（PR #7） | 充足 | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) | 重複ID: `test_phenotype_duplicate_sample_id_is_rejected` / `test_genotype_duplicate_sample_header_is_rejected` / `test_marker_id_duplicate_is_rejected`。必須列欠落: `test_missing_phenotype_column_is_rejected`（`"missing phenotype columns"` と列名・ファイル名をassert） | `test_marker_id_missing_or_empty_is_rejected` は**marker ID値**の欠落・空文字の検査であり、必須列欠落の根拠には含めない |
| [x] 欠損値、MAFおよびimputation処理の単体テストを追加する | 充足 | [`tests/test_input_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_input_qc.py)（6関数） | CI 単体テスト | — |
| [x] split間のsampleまたはgroup重複を検出するテストを追加する | 充足 | [`tests/test_comparison.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_comparison.py) `test_fixed_split_roundtrip_and_tampering`、[`tests/test_gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gs_scenarios.py) `test_family_environment_cross_cells_are_excluded` | CI 単体テスト | — |
| [x] CIで単体テストとCPU smoke testを実行する | 充足 | [`.github/workflows/ci.yml`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/.github/workflows/ci.yml) | main `39ce302` の run [35430336512](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/runs/35430336512) success | GPU runnerは無く、CI成功はGPU経路の検証にならない |

## 完了条件

| 元の項目 | 判定 | 対象実装・テスト | CI／実機検証の証跡 | 文言更新の必要性と残る制約 |
|---|---|---|---|---|
| [x] クリーン環境からREADMEの手順だけでCPU smoke testを実行できる | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) Quick start／[`tests/test_cpu_smoke.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_cpu_smoke.py) | CI「Lint, unit tests, and CPU smoke」 | synthetic 3 family |
| [x] Docker Compose経由で検証済みの単体テスト・CPU smokeを実行でき、実データ向けGBLUP・ResNetコマンドを開始できる（PR #8） | 充足 | PR #8 / [`docker-compose.yml`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docker-compose.yml) | CI「Docker Compose build and smoke」 | 実データservice（`real-data` profile）はCI未実行 |
| [x] 入力データの不整合が処理開始前に検出される（PR #7） | 充足 | PR #7 / [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py)（`SoynamDataTest` の各negative test） | 読み込み時点で失敗する |
| [x] 欠損値がヘテロ接合と区別される | 充足 | [`soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/soynam_data.py) | [`tests/test_soynam_data.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_soynam_data.py) `test_raw_loader_excludes_parent_and_preserves_missing` | — |
| [x] 前処理が学習データだけでfitされる | 充足 | [`resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/resnet_baseline.py)／[`gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gblup_baseline.py) | [`tests/test_resnet_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_resnet_baseline.py) `test_feature_statistics_are_fitted_on_training_rows_only`、[`tests/test_gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gblup_baseline.py) `test_vanraden_relationship_uses_training_statistics` / `test_test_genotypes_do_not_change_training_preprocessing` | **対象はモデル側の前処理**（marker選択・imputation・標準化・PCA等）。canonical phenotypeのBLUP調整は全familyの観測を用いて別途fitしており、これは**完全な外部検証を保証しない**（[#6レポート](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/docs/experiments/issue-6-soynam-1.6.2.md) §8） |
| [x] group-aware CVでgroupの重複がないことをテストで保証できる | 充足 | [`gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gs_scenarios.py) `_fold()` が **observation_id** について「train/test」「fit/validation」「fit/test」「validation/test」の4組の非重複を検査し、あわせて **fit ∪ validation ⊆ train** を検査する。SoyNAM経路は [`evaluation_split.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/evaluation_split.py) の固定LOFO | [`tests/test_gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gs_scenarios.py) `test_family_environment_cross_cells_are_excluded` / `test_known_line_history_is_allowed_and_time_is_strict`、[`tests/test_comparison.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_comparison.py) `test_fixed_split_roundtrip_and_tampering` | `_fold()` が見るのは上記4組の非重複と内側が外側trainの部分集合であることで、**fit・validationは train の部分集合であるため4集合すべてが互いに排他的という意味ではない**。family / line / site の条件は各シナリオの構築処理と `_status_check()` が担当する。また、異なるfold間でtrain集合が重なるのはCVの正常な性質で、不具合ではない。`bootstrap_group` は**不確実性評価の再標本化単位**であって、分割条件を決めるフィールドではない |
| [x] Linear Path、GBLUPおよびexperimental機能の位置付けがREADME上で明確になっている | 充足 | [`README.md`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/README.md) 機能表・Linear Path節・既知の制約節 | CI（Ruff・単体テスト） | — |
| [x] CIの単体テストとsmoke testが成功する | 充足 | [`.github/workflows/ci.yml`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/.github/workflows/ci.yml) | main `39ce302` の run [35430336512](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/runs/35430336512) **success**（2026-09-19T07:48:46Z, event=push, branch=main） | この成功は**既存main**に対するもの。本棚卸しPRのHEADに対するCIは別途記録する |

## GSシナリオの分割条件とbootstrap単位

[`gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/gs_scenarios.py) `_fold()` が検査するのは **observation_id** についての次の2点だけです。

- 「train/test」「fit/validation」「fit/test」「validation/test」の4組が互いに重複しないこと
- **fit ∪ validation ⊆ train** であること

fitとvalidationは外側trainの部分集合なので、**4集合すべてが互いに排他的という意味ではありません**。
family / line / site の条件は、各シナリオの構築処理と `_status_check()` が担当します。

`bootstrap_group` は**不確実性評価の再標本化単位**であり、分割条件を決めるフィールドではありません。

| シナリオ | splitの条件 | known/new の line・site 制約 | bootstrapの単位 |
|---|---|---|---|
| `family_lofo` | family を単位に outer test を取り、内側も family 単位で分ける | 宣言しない | `family_id` |
| `future_year` | 年次順。`test_year` / `validation_year` を明示し、時間逆転を拒否する | `genotype_status` を `line_id`、`environment_status` を `site` に適用。`known` は対象列のID集合について **test ⊆ train** を要求し、`new` は重複を拒否する | `line_id` |
| `family_environment` | 保留した family と site の組（セル）を outer test にし、交差セルを除外する | `test_families` と `validation_families`、`test_sites` と `validation_sites` の重複を拒否する | `family_id` |

`known` の「test ⊆ train」は、**対象列のID集合についての条件**です。test観測そのものがtrainへ混入するという意味ではありません。
`future_year` の known genotype では、**同一 `line_id` の過年度観測をtrainに持つ設計が意図されています**（[`tests/test_gs_scenarios.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gs_scenarios.py) `test_known_line_history_is_allowed_and_time_is_strict`）。
全シナリオで同一lineを禁止する要件ではありません。

## MAF処理の根拠範囲

MAFについては「設定できる実装」と「テストで確認した性質」を分けて記録します。

| 区分 | 内容 |
|---|---|
| 設定箇所（実装） | `gblup_baseline.prepare_fold_relationships(maf_threshold=...)`（`0.0 <= maf_threshold < 0.5` を検証）、`resnet_baseline.ResNetConfig.maf_threshold`（既定0.01）、[`controlled_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/controlled_qc.py) `marker_mask(maf_threshold=...)` |
| CLI | **`--maf-threshold` は存在しない。** `input_qc.add_arguments()` が公開するのは `--max-sample-missing-rate` と `--min-marker-observed-rate` のみ |
| 欠損除外（実装） | legacy経路は `np.nanmean`、controlled経路は `nansum / 観測数`。いずれも欠損を除外し学習fold内で計算する（実装読み取りによる確認） |
| テストで確認できた範囲 | **観測率フィルター**の挙動のみ。[`tests/test_gblup_baseline.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_gblup_baseline.py) `test_low_observation_marker_is_removed`（`maf_threshold=0.0`、`min_observed_rate=0.5`）と [`tests/test_input_qc.py`](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/blob/39ce3027bc3440545377d108e9a29714bdb45818/tests/test_input_qc.py) `test_marker_threshold_uses_only_training_calls`（`min_observed_rate` の strict/loose と学習fold限定）は、いずれもMAFの検査ではない |
| 未確認 | 閾値を変えてMAF起因でmarkerが除外されることをassertするテストは確認できなかった。MAFを渡す既存テストは `maf_threshold=0.0` か、`0.05` でもMAF起因の除外をassertしていない |

このため「MAFフィルターの閾値を設定可能にする」の判定を**未確認**としています。今回テストやCLIは追加していません。


## 今回の棚卸しで更新したもの

| 論点 | 内容 |
|---|---|
| #6 GPU本実験 | 完了。PR #38（merge commit `39ce302`）で実測レポートをマージし、進捗欄の該当項目を充足へ更新した。実験・精度・計算コスト・保存の監査はレポート §10 とIssue #6の完了記録を根拠とし、本棚卸しではやり直していない |
| group-aware CV | GS経路の `--policy`（3シナリオ）として実装済み。#28（closed / PR #33）が親Issueの未完部分を具体化した。SoyNAM経路は固定LOFOのままで、別CV方式を選ぶCLIは提供しない。シナリオごとの分割条件・known/new制約・bootstrap単位は「GSシナリオの分割条件とbootstrap単位」節に分けて記載 |
| `rpy2` / `soynam` extra | 廃止済み。現在は学習・評価環境から独立したRscript環境（conda explicit lock）を外部プロセスとして呼び出す方式。rpy2は再導入しない |
| READMEの家系数 | LOFOは入力の各familyを保持する方式で、CRAN canonical datasetでは39 family。GBLUP CLIの `--expected-families` 既定は39（`gblup_baseline.py` の `EXPECTED_FAMILY_COUNT`）。legacy local datasetの16 familyとは区別する |

## 残る制約

- **MAFフィルターの閾値にCLIフラグは無く、閾値変更時の挙動を裏付けるテストも確認できていない。** 該当項目の判定は未確認とした（「MAF処理の根拠範囲」節）。本棚卸しではテスト・CLIを追加していない。

- **GPU経路はCIで実行していない。** CI成功はGPU経路の検証にならない。実機証跡は [gpu-verification.md](gpu-verification.md) と [#6 実測レポート](experiments/issue-6-soynam-1.6.2.md)。
- **#6は事前定義した1候補による単一実験**で、ハイパーパラメータ探索や独立な反復実験ではない。モデル間でmarker QC・前処理が異なり、アーキテクチャ単独の比較ではない。
- **canonical phenotypeのBLUP調整は全familyの観測を用いている。** 「前処理が学習データだけでfitされる」はモデル側の前処理を指し、完全な外部検証を保証するものではない。
- **SoyNAM経路にCV方式の選択CLIは無い**（設計上の決定。#28受入条件「既存LOFOは維持」）。
- #6で未取得のまま残る計測（ホスト全体CPU時間・RSS、fold単位wall、`cuda_driver_api_version`、選択段階の停止epoch、ResNet artifactの `external_logging`）は既知の制約として維持する。
- 実データの性能保証・生物学的な閾値の妥当性は本リポジトリの検証範囲外。

## 欠損率ポリシー

両baselineに次の引数があります。閾値は結果を見る前に確定してください。

- `--max-sample-missing-rate`：0〜1。欠損率がこの値を超える個体を除外。既定1は全個体を維持します。各個体自身の遺伝型だけで判定し、家系が丸ごと消える場合は失敗します。
- `--min-marker-observed-rate`：学習fold内の観測率フィルター。GBLUPの既定0.1は厳密な`>`、ResNetの既定0.9は`>=`です。既存計算を維持しており、同じ数値を指定しても境界値の扱いは異なります。GBLUPでは0と1を引数検証で拒否します。

MAFの既定値もGBLUP 0.05 / ResNet 0.01、分散判定も異なります。同一splitだけで前処理が同一になったとは扱わず、#6の比較でそれぞれの条件を記録します。imputationは既存どおり学習foldの平均値です。

`metadata.json.input_qc`に対象個体の順序、閾値、除外前後の件数、配列参照を記録し、`preprocessing_arrays.npz`にsample欠損率・採用mask・marker欠損率を保存します。既存の6ファイル構成を維持します。ここでのmarker欠損率は欠損表現型・founder除外後、sample QC前の記述統計で、marker選択には使いません。marker選択・imputation・標準化・PCAは引き続き各学習fold内でfitします。

テストはsyntheticデータによる、閾値境界・個体と表現型の対応・家系消失の拒否・学習fold限定のmarker選択・両CLIの成果物保存の確認です。生物学的な閾値の妥当性を示すものではありません。

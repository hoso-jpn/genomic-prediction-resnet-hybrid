# 履歴に残るデータ・学習済み重みの出典調査

`docs/generated-artifacts.md` §3.4で記録した「現在`main`に存在しないが公開履歴に残るファイル」について、出典と再配布条件を調査した記録です。**調査時点で結論が出ていない項目は「未確認」と明記し、事実・推定・未確認を分けています。取得経緯など利用者にしか分からない事項は、回答待ちとして未確認のまま残しています（推測で補いません）。** 本調査のためにデータ・重みを新たに公開・再配布していません（内容の再掲もしていません）。

調査日: 2026-08-31 / 調査対象リポジトリ: `hoso-jpn/genomic-prediction-resnet-hybrid`

## 0. 前提

- 本リポジトリの`LICENSE`（MIT）は、**このリポジトリのコードに対するライセンス**です。第三者が配布するデータや、そこから派生した成果物へ自動的に適用されるものではありません。本文書ではデータ・重みをコードとは別に扱います。
- 対象は次の3区分です。

| 区分 | 対象 | 履歴上の期間 |
|---|---|---|
| A. 生データ | `4J105-3-4_geno.csv` → `data/05-3-4_geno.csv`、`4J105-3-4_pheno.csv` → `data/4J105-3-4_pheno.csv` | 2026-01-30 追加 → 2026-03-06 削除 |
| B. 加工済み配列・表現型 | `processed_data_hy/X_genotype_int8.npy`、`processed_data_hy/y_phenotype_hy.csv` | 2026-03-19 追加 → 同日削除 |
| C. 学習済み重み | `genomic_resnet_hybrid.pth` | 2026-03-19 追加（`15000b6`）→ 同日削除（`417ee4c`） |

## 1. A. 生データ（`data/*.csv`）

### 確認できた事実

- 追加commitは`02ce200`（2026-01-30、メッセージは`data/`のみ）。その後`58e2762` / `c98ffb3`で`data/`配下へリネーム、`7e710c2` / `1cf1c2b`（2026-03-06）で削除。**取得元URL・取得日・配布版を記録したcommit message、スクリプト、ドキュメントは履歴内に存在しない**（`git log -S`および`git grep`で確認）。
- 遺伝型CSVの列構成（先頭8列）: `dbSNP_ID`, `DEPRECATED-BARCSOYNAMSNP6K_ID`, `Chr_Wm82.a1`, `Position_Wm82.a1`, `Parent_IA3023`, `4J105-3-4`, `DS11-03003`, `DS11-03004`, …（4,312 marker行 × 143列）。
- 表現型CSVの列構成: `Loc`, `Year`, `Env`, `Plant date`, `Corrected Strain`, `Original Strain`, `Family`, `FamNo`, `Set`, `FamSet`, `Entry`, `Test`, `Name`, `Plot`, `R1`, `Ht (in)`, `Ht (cm)`, `Mat`, `Days to Mat`, `Lod`, `Yld (bu/a)`, `Yld (kg/ha)`, `notes`, `Moisture`, `Protein`, `Oil`, `Fiber`, `100 sdwt (g)`（1,920行 × 28列）。
- 現行READMEは、検証済み経路の入力としてSoyNAM（SoyBase配布）を挙げ、「データの利用条件と引用方法は配布元の案内に従ってください」と記載しています。

### 推定（確定ではない）

- 列名（`Parent_IA3023`はSoyNAMの共通親、`4J105-3-4`はfamily名、`DS11-*`はRIL ID、`Chr_Wm82.a1`はWilliams 82 assembly a1）と、現行の検証済み経路が扱うSoyNAMファイル形式との整合から、**SoyNAMプロジェクトの公開データに由来する**と推定されます。
- 形式（family単位・`dbSNP_ID`列を含むCSV）は、CRANの`SoyNAM` Rパッケージが提供する`data(swat)` / `data(G2f)`の構造とは異なるため、**Rパッケージ経由ではなくSoyBaseのダウンロード配布物に由来する**可能性が高いと考えられます。

### 未確認

- 取得元の具体的URL・ページ、取得日時、配布版（リリース／更新日）。**利用者しか知りえない情報です（§5の質問1）。**
- 取得時に同意した利用条件（クリックスルー規約等）の有無。
- SoyBase側の利用・再配布条件および必要な引用・表示。**`https://www.soybase.org/projects/SoyNAM/` と `https://www.soybase.org/data_policy.php` は自動取得がHTTP 403で拒否されたため、本調査では条文を確認できていません**（§5の作業1）。
- 列名からは第三者の非公開データが含まれないことまでは確定できません。列名の一致は形式の一致であり、出所の証明ではありません。

## 2. B. 加工済み配列・表現型（`processed_data_hy/*`）

### 確認できた事実

- 追加commitは`417ee4c`（2026-03-19）、削除commitは`bf801b1`（同日、メッセージ: "chore: remove processed data from remote for privacy and repo size"）。
- 生成元スクリプトは同commit時点の`preprocess.py`で、入力は`./data/*_phenotype_data.tsv.gz`と`./data/*_SNP_genotype_Wm82.a1.tsv.gz`（gzip TSV、family単位）。**この入力ファイル自体はGitに追跡されたことがありません**（区分Aの`.csv`とは別形式）。
- `X_genotype_int8.npy`は`shape (2180, 4312)`・`int8`。`y_phenotype_hy.csv`は2,180行・列は`Corrected Strain`と`Yld (kg/ha)`のみ（この時点ではfamily_id列なし）。サンプルIDの接頭辞は`DS`が2,164件で大半を占めます。
- 4,312という marker 数は、区分Aの遺伝型CSVの marker 行数と一致します。

### 生成コードとの整合（＝実際の生成経緯の確認ではありません）

- 同commit時点の`preprocess.py`は、`./data/*_phenotype_data.tsv.gz`と`*_SNP_genotype_Wm82.a1.tsv.gz`（**Gitに追跡されたことのない別形式のファイル**）を入力とし、`processed_data_hy/`配下へ`.npy`と`.csv`を書き出します。出力の形状・列構成はこのコードの出力仕様と整合します。
- marker数4,312は区分AのCSVの marker 行数とも一致しますが、**これは数の一致であり、区分AのCSVそのものから生成されたことの確認ではありません**。実際の入力は上記の未追跡tsv.gzであり、そのtsv.gzと区分AのCSVの関係（同一配布物か、別取得か、同一版か）は確認できていません。

### 未確認

- 実際に入力へ使ったtsv.gzの取得元・取得日・版・利用条件（**利用者の回答待ち**。§5の質問1・2）。
- 生成実行そのものの記録（いつ・どの入力ファイル・どのコード版で実行したか）。
- 区分AのCSVと当該tsv.gzが同一の配布物・同一版に由来するかどうか。
- 派生物の再配布可否は、元データの条件に依存します（元条件が未確認のため未確定）。

## 3. C. 学習済み重み（`genomic_resnet_hybrid.pth`）

### 確認できた事実（推定ではありません）

- 追加commitは`15000b6`（2026-03-19）。同じcommitに`train.py`が含まれ、その49行目が`torch.save(model.state_dict(), "genomic_resnet_hybrid.pth")`です。
- その`train.py`の学習入力は**`torch.randn`で生成した乱数**です（`X_train = torch.randn(100, INPUT_DIM)` / `y_train = torch.randn(100, 1)`、コメントに「ダミーデータの生成 (本番はここを実データ読み込みに差し替え)」）。同commitの`data_loader.py`も同様に乱数を返す実装でした。
- ハイパーパラメータは`INPUT_DIM = 1000`, `HIDDEN_DIM = 256`, `NUM_BLOCKS = 3`。
- 実際の`.pth`（履歴から取り出して`weights_only=True`で読み込み、内容は再配布していません）は、`input_layer.0.weight`が`(256, 1000)`、`output_layer.weight`が`(1, 256)`、`res_blocks`のインデックスは`0,1,2`。**`train.py`のハイパーパラメータと完全に一致します。**
- 一方、SoyNAM由来データのmarker数は4,312です。入力次元1,000はこれと一致しません。

### 現時点の整理（確定ではありません）

確認できているのは次の3点です。

1. 同じcommit（`15000b6`）に、**乱数データで学習して同名ファイルを保存するコード**（`train.py`）が含まれる。
2. 重みの構造・形状（`input_layer.0.weight = (256, 1000)`、`res_blocks` 0〜2、`output_layer.weight = (1, 256)`）が、そのコードのモデル設定（`INPUT_DIM=1000` / `HIDDEN_DIM=256` / `NUM_BLOCKS=3`）と**一致する**。
3. **当該ファイルと、その生成実行を結びつける記録（実行ログ、checksum、成果物マニフェスト等）は確認できていない。**

したがって、この重みについて言えるのは「**乱数による学習コードと整合するが、当該ファイルの生成実行との対応は未確認**」までです。次のような断定はしません。

- 「乱数由来と確定」— 生成実行との対応が未確認のため。
- 「第三者データ由来の情報を含まない」— 上と同じ理由。構造の一致は、同設定の別実行（異なる入力データ）を排除しません。
- 「公開継続で問題なし」— 判断はデータ・成果物の保有者が行うものです。

一方で、次は維持します。

- この重みは現行の`GatedGenomicResNet`（`cnn_path` / `linear_path` / `gate`）とは異なる旧アーキテクチャ（`input_layer` / `res_blocks` / `output_layer`）のもので、現在のコードからは読み込めません。**現行モデルの予測性能の証跡としては使えません。**
- 入力次元1,000はSoyNAM由来データのmarker数4,312と一致しません（これは形状に関する事実であり、生成経緯の確定ではありません）。

### 未確認

- 当該ファイルを生成した実行そのもの（いつ・どのコード・どの入力で実行されたか）。履歴に含まれるW&B run（18件、2026-03-18T23:10Z〜2026-03-19T11:32Z）のプログラムは`train.py`（4件）と`main.py`（14件）、引数は全件なしでしたが、**これらのrunと当該ファイルを結びつける記録はありません**。
- 同設定で別の入力データを用いた実行が行われた可能性（構造の一致だけでは排除できません）。
- この点の確定には、生成当時の実行記録や成果物のchecksumが必要です。**本調査では再学習や大規模な追加調査は行っていません。**

## 4. 評価と提案

| 区分 | 現時点の評価 | 提案 |
|---|---|---|
| C. 学習済み重み | 乱数による学習コードと整合するが、**当該ファイルの生成実行との対応は未確認**。現行モデルの性能証跡には使えない | 位置付け（旧アーキテクチャの成果物であり性能証跡ではないこと）の明記を維持したうえで、**扱いは保有者の判断**とする。生成実行との対応を確定させるには当時の実行記録・checksumが必要 |
| A. 生データ / B. 派生物 | SoyNAM由来と推定。取得経緯は**利用者の回答待ち**、配布元の条件は**未確認**のため、再配布の可否を判断できない | 次の順で進めることを提案します |

1. **条件の確認（まず実施）**: SoyBaseのSoyNAM配布ページと利用条件を、ブラウザ等の手動アクセスで確認する（自動取得は403）。確認できた条文・引用要件を本文書へ追記する。
2. **条件を満たす場合 → 公開継続 + 表示の補完**: 出典（配布元・取得日・版）、必要な引用（SoyNAM関連論文・SoyBase）、「MITはコードのみに適用され、データは配布元条件に従う」旨をREADMEと本文書へ明記する。履歴に残る事実も併記する。
3. **条件が再配布を許さない、または不明のまま残る場合 → 公開制限・履歴清掃の検討**: 影響範囲（既存clone・fork・PR参照の失効、GitHub側に残る到達不能オブジェクト、GitHub Supportへの依頼要否）と承認手順を整理したうえで判断する。**本文書の作成時点では、履歴清掃・force push・公開範囲の変更は一切行っていません。**

区分Cは区分A・Bとは独立した論点ですが、**生成実行との対応が未確認である以上、「問題なし」と結論づけていません**。

**本調査に伴って、データ・重みを新たに公開・再配布していません。履歴清掃・force push・公開範囲の変更も行っていません。配布元への問い合わせは文案作成のみで、送信していません。** 再配布条件の確認は、コード修正（各PRの実装変更）とは分けて管理します。

## 5. 利用者への確認事項と、配布元への問い合わせ文案（未送信）

### 質問（利用者にしか分からない取得経緯）

1. `4J105-3-4_geno.csv` / `4J105-3-4_pheno.csv` は、どこから・いつ・どの版を取得したものですか（SoyBaseのダウンロードページ、共同研究者からの受領、その他）。取得時に同意した利用条件があれば、その内容も。
2. `processed_data_hy/*` の生成に使った`data/*.tsv.gz`は、上記と同じ取得元・同じ版ですか。
3. 取得元がSoyBase以外（例: 共同研究機関、育種機関）である場合、その配布物に非公開データが含まれる可能性はありますか。

### 配布元への問い合わせ文案（**送信していません**。必要なら利用者が送付してください）

> Subject: Question about redistribution terms for SoyNAM genotype/phenotype data
>
> Dear SoyBase team,
>
> I am working on a public research repository that uses the SoyNAM genotype and phenotype data distributed via SoyBase. Some earlier commits of the repository's public git history still contain copies of per-family SoyNAM genotype/phenotype files (for example a genotype table with `dbSNP_ID`, `Chr_Wm82.a1`, `Position_Wm82.a1` columns and a phenotype table with `Corrected Strain` and `Yld (kg/ha)` columns), even though the files have since been removed from the current tree.
>
> Could you clarify:
> 1. Whether redistribution of unmodified SoyNAM data files in a public repository is permitted, and under what conditions;
> 2. Whether derived data (e.g. a numeric genotype matrix produced from those files) may be redistributed;
> 3. What citation or acknowledgement you require for use and for redistribution.
>
> I would like to comply with your terms, including removing the files from the published history if that is required.
>
> Thank you for your help.

## 6. 調査方法と限界

- 調査に用いたのは、リポジトリのgit履歴（commit message、`git log -S`、`git grep`、`git show`によるファイル構造の確認）、ローカルに残るW&B run metadata、および公開されているWeb情報のみです。
- SoyBaseの利用条件ページは自動取得が403で拒否されたため未確認です。CRANの`SoyNAM` Rパッケージ（License: GPL-3、USB資金によるSoyNAMデータ）は確認しましたが、区分Aのファイル形式とは異なるため、直接の出典とは判断していません。
- 列名や形式の一致は「形式が一致する」ことの確認であり、出所の証明ではありません。第三者の非公開データが含まれないことは、この方法では確定できません（区分Cを除く）。

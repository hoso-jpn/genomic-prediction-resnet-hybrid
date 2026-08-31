# 生成物と再現性証跡の管理

Issue #15で、Gitに追跡されていたW&Bログとダミー重みの追跡を停止しました。本ドキュメントは、対象の一覧・確認方法・確認結果（値は記載しない）・再生成手順を記録します。

## 1. 追跡を停止した対象

| 対象 | 内容 | 停止方法 | 再生成 |
|---|---|---|---|
| `wandb/`（162エントリ、うち18はsymlink） | 2026-03-19に実行された旧runのローカルW&Bログ（`config.yaml`、`output.log`、`requirements.txt`、`wandb-metadata.json`、`debug*.log`、`run-*.wandb`） | `git rm -r --cached wandb`（ローカルファイルは保持） | W&Bを`--wandb-mode offline`で実行すると再生成される |
| `pretrained_models/dummy_cnn_weights.pt`（2,787,062 bytes） | ランダム初期化したCNN重み。事前学習済みモデルではない | `git rm --cached`（ローカルファイルは保持） | `python create_dummy_pretrained_weights.py`（既定 seed=42） |

`.gitignore`では以前から`wandb/`と`*.pt`を除外していましたが、**既に追跡されているファイルには遡及しない**ため、明示的に追跡を停止しました。`pretrained_models/`ディレクトリ自体は削除していません（手書きコードと生成物を区別するため、生成物のみを対象にしています）。

作業者のローカルファイルは削除していません。`git rm --cached`はインデックスからのみ削除するため、`wandb/`配下のログも`pretrained_models/dummy_cnn_weights.pt`も作業ツリーに残ります。追跡停止後は`.gitignore`により、これらは未追跡差分としても現れません。

## 2. 秘密情報の確認

### 対象と方法

- 対象: 追跡されていた`wandb/`配下の全ファイル（symlinkを除く）と`pretrained_models/dummy_cnn_weights.pt`。テキスト・バイナリを問わずバイト列として検査した。
- 方法: 正規表現による機械的な検査。検出パターンは、40桁hex文字列（W&B API keyと同じ形）、`api_key` / `token` / `secret` / `password` / `authorization` のkey-value形式、メールアドレス形式、`/home/<user>`形式の絶対パス、IPv4形式、PEM秘密鍵ヘッダー。
- 検出値は本ドキュメント・Issue・PR・CIログのいずれにも記載していない。分類のみを記録する。

### 結果（値は非掲載）

| 分類 | 結果 |
|---|---|
| PEM秘密鍵 | 検出なし |
| `api_key` / `token` / `secret` / `password` / `authorization` のラベル付き値 | 検出なし |
| 40桁hex文字列 | 4種類。うち3種類は本リポジトリのgitオブジェクト（W&Bが記録したgit commit）として解決した。残り1種類はバイナリのrun記録内にのみ現れ、当初は判断を保留していた。§3.3でW&Bクライアントのバージョン／ビルド識別子（`<semver>+<40桁hex>`という単一文字列の一部）と判定済み |
| メールアドレス形式 | 3種類。`wandb-metadata.json`と`config.yaml`の`email`（gitのauthor email）と`remote`（gitのremote URL）フィールドに出現 |
| `/home/<user>`形式の絶対パス | 多数（実行時の作業ディレクトリ・依存パス） |
| IPv4形式 | `requirements.txt`のみ。バージョン文字列の誤検出と判断 |

### 判断と限界

- 認証情報として断定できる値は検出されなかった。この節はHEADの追跡ツリーだけを対象にした検査であり、公開済みコミット全体の検査は§3にある（§3は既存履歴と今回のPRを区別して記録している）。過去のcommitに含まれる内容は、追跡停止では消えない。
- 実行環境のユーザー名を含む絶対パスとgitのauthor emailが公開履歴に残っている。これはコミット作者情報として既に公開されている情報と同種であり、失効・ローテーション対象ではない。
- もし今後、実際の認証情報が見つかった場合は、**まず失効・ローテーションを行う**。履歴の書き換えは影響範囲と承認を別途確認する（本対応では履歴書換え・force pushは行っていない）。

## 3. 公開済みコミット全体の検査（PR #17〜#23）

§2はHEADの追跡ツリーだけを対象にした検査です。これとは別に、今回作成した7本のPRで公開した**全コミット**と、取得できる公開ref（ブランチ・タグ）から到達可能な**既存の履歴**を検査しました。

### 3.1 ツールと対象

| 項目 | 内容 |
|---|---|
| ツール1 | `detect-secrets` 1.5.0（Yelp。keyword・high-entropy各plugin。scratch venvへ導入し、リポジトリには追加しない） |
| ツール2 | 自作の正規表現スキャナ（下記パターン。バイナリ・gzipを含む全blobをバイト列として走査） |
| git | 2.43.0 / Python 3.11.15 |
| 検査日 | 2026-08-31 |
| 対象ref（新規） | `origin/feat/10-adzuki-gs-panel-loader` `8a7381f`、`origin/feat/11-gblup-cli-wandb-optin` `9f12a41`、`origin/feat/12-legacy-guardrails` `4a0d2ed`、`origin/feat/13-cuda-environment` `0143523`、`origin/feat/14-gnn-edge-contract` `72a489a`、`origin/chore/15-untrack-generated-artifacts` `3c9cf71`、`origin/chore/16-ruff-all-code` `9f53123` |
| 対象ref（既存） | `origin/main` `3ea388e`、`origin/fix/3-docker-compose-baselines` `c70af28`（タグは0件） |
| 対象範囲（新規） | `git rev-list --objects <7 heads> --not origin/main` = 7 commit / 54 blob / 638,769 bytes。スタック間で重複するcommitは重複排除済み。**途中のコミットで追加し後で削除した内容も、この object set に含まれるため対象**。加えて7件のcommit message、および公開したPR #17〜#23のタイトル・本文（42,606 bytes） |
| 対象範囲（既存） | `git rev-list --objects origin/main origin/fix/3-docker-compose-baselines` = 56 commit / 272 blob / 18,383,442 bytes（`wandb/`の追跡ログ、削除済みファイルを含む） |
| 除外範囲 | どのrefからも到達できない（dangling）オブジェクト、GitHub側にのみ残る可能性のある削除済みref・PRのforce push前スナップショット、`.gitignore`済みのローカル作業ファイル、`.venv`等の環境 |

検出パターン（認証情報クラス）: PEM秘密鍵ヘッダー、`api_key`/`token`/`secret`/`password`/`authorization`等のkey-value、`WANDB_API_KEY`の代入、GitHub token（`ghp_`等・`github_pat_`）、AWSアクセスキー、Slack token、Google APIキー、JWT、netrcのpassword行。
検出パターン（識別子・PIIクラス）: メールアドレス、`/home/<user>`・`/Users/<user>`、IPv4、40桁hex、64桁hex。

### 3.2 結果（値は非掲載）

**今回新しく公開した内容（7 commit / 54 blob / commit message / PR本文）**

| 分類 | 結果 |
|---|---|
| `detect-secrets` | 検出0件 |
| 認証情報クラス（全パターン） | 検出0件 |
| メールアドレス | blobに0件。commit messageに2種類（コミット作者のアドレスと`noreply@anthropic.com`。gitのcommitter情報として元から公開される情報） |
| `/home/<user>`形式の絶対パス | 0件 |
| 40桁hex | `.github/workflows/ci.yml`のGitHub Actionsのpin（`actions/checkout`・`astral-sh/setup-uv`のcommit SHA。公開が前提の値）と、commit message中の自コミットSHAのみ |
| 64桁hex | `cuda/uv.lock`の375件のみ（wheelのsha256。lockの正当な内容） |
| バイナリ・非UTF-8 blob | **0件**（新規公開分はすべてUTF-8テキスト） |

**以前から公開されていた内容（56 commit / 272 blob）**

| 分類 | 結果 |
|---|---|
| 認証情報クラス（全パターン） | 検出0件 |
| `detect-secrets` | 38件。すべて`wandb/`配下の`config.yaml`・`wandb-metadata.json`。内訳は Hex High Entropy 36件（下記の40桁hex）と Base64 High Entropy 2件（キー名は`writerId`。W&Bの内部run識別子） |
| 40桁hex | 6種類。3種類は本リポジトリのcommitオブジェクトとして解決し、いずれも`origin/main`の祖先。`commit`キーの値として出現（W&Bが記録したgit commit）。2種類は`ci.yml`のActions pin（他リポジトリのSHAなので当然ここでは解決しない）。残り1種類が下記§3.3 |
| メールアドレス | 実体は2種類。（a）`email`フィールドのコミット作者アドレス、（b）`remote`フィールドのSSH remote URL中の`git@github.com`。第三者のアドレスは無し（3種類目は同一アドレスにバイナリ境界のバイトが連結された正規表現アーティファクト） |
| `/home/<user>`形式 | 608件だが**実体は1種類**（実行者自身のログイン名を含む作業ディレクトリ） |
| IPv4形式 | 7種類。`requirements.txt`等の4桁区切りバージョン文字列の誤検出 |

### 3.3 PR #20で未解決だった40桁文字列の判定

PR #20では「バイナリのrun記録内に、リポジトリのオブジェクトとして解決しない40桁hexが1種類ある」と記録し、判断を保留していました。今回、値そのものではなく**記録形式**から根拠を確認しました。

- 出現位置は18個の`run-*.wandb`のみで、`config.yaml`・`wandb-metadata.json`には現れない。
- 直前は常に`<semver>+`という形（例: `0.25.1+`）。
- protobuf形式の長さ前置を読むと、長さバイトは`0x2f`＝47で、`"0.25.1+"`（7バイト）＋40桁hex＝47バイトと一致する。つまりこの40桁hexは**単一の可変長文字列フィールドの一部（バージョン文字列の末尾）**であり、独立したフィールドではない。
- 認証情報を示すキーに隣接しておらず、18 run すべてで同一（同一バージョンのW&Bクライアントで実行された記録と整合）。

以上から、この値は**W&Bクライアントのバージョン／ビルド識別子の一部**であり、**認証情報であることを示す証拠は無い**と判定しました（本リポジトリのcommit SHAとしては解決しません）。一方で、ビルド識別子として埋め込まれた**別リポジトリ（W&B自身など）のcommit SHAである可能性は否定していません**。ここで確認できたのは「独立したフィールドではなく、バージョン文字列の一部である」ことまでです。

### 3.4 認証情報以外の公開範囲の指摘（既存履歴・今回のPRとは無関係）

認証情報ではありませんが、**以前の履歴に、現在は`main`に存在しない研究データとモデルが残っています**。追跡停止・ファイル削除では履歴から消えないため、事実として記録します。

| パス | サイズ | 追加 | 削除 |
|---|---:|---|---|
| `data/05-3-4_geno.csv`（4,312 marker × 141列。列名は`dbSNP_ID`・`Parent_IA3023`等） | 1.4 MB | 2026-01-30 | 2026-03-06 |
| `data/4J105-3-4_pheno.csv`（1,920行 × 28列） | 279 KB | 2026-01-30 | 2026-03-06 |
| `processed_data_hy/X_genotype_int8.npy`（shape 2180 × 4312） | 9.4 MB | 2026-03-19 | 2026-03-19 |
| `processed_data_hy/y_phenotype_hy.csv`（2,180行） | 66 KB | 2026-03-19 | 2026-03-19 |
| `genomic_resnet_hybrid.pth`（学習済みモデルの重み） | 2.6 MB | 2026-03-19 | 2026-03-19 |

列名（`dbSNP_ID`、`DEPRECATED-BARCSOYNAMSNP6K_ID`、`Parent_IA3023`、`Corrected Strain`）はSoyNAMの配布形式と一致しますが、**列名の一致は形式の一致であって出所の証明ではありません**。出典・再配布条件の調査結果は[docs/data-provenance.md](data-provenance.md)にまとめています（`genomic_resnet_hybrid.pth`については、同一commitの`train.py`が乱数データで学習して保存したものであることを、入力次元・アーキテクチャの一致から確認済み。生データ・派生物については配布元の条件が未確認）。

本対応では履歴の書き換え・force pushは行いません。必要と判断される場合も、影響範囲（既存clone・fork・PR参照の失効、GitHub側に残る到達不能オブジェクトの扱い）と承認を別途確認してください。

### 3.5 差分検査（2026-08-31、更新後の最終headに対して）

§3.1〜§3.4の検査後、7本のPRへ追加のcommit（レビュー対応・スタックへのforward merge・ドキュメント更新）を行いました。既存履歴は変化していないため再検査せず、**前回検査済み集合との差分だけ**を対象にしています。

| 項目 | 内容 |
|---|---|
| 前回検査済みhead | `8a7381f` `9f12a41` `4a0d2ed` `0143523` `72a489a` `3c9cf71` `9f53123` |
| 今回の検査対象head | `8a7381f`（#10）、`fc725de`（#11）、`35e4057`（#12）、`67d01ae`（#13）、`1876380`（#14）、`93d578d`（#15。この節を追加する直前の版）、`fdd418a`（#16） |
| 差分の範囲 | `git rev-list --objects <7 heads> --not <前回の7 heads> origin/main` = **14 commit / 20 blob / 450,577 bytes**（うち9件はスタック伝播のmerge commit）＋ 14件のcommit message ＋ 更新後のPR #17〜#23の本文（30,945 chars） |
| 既存履歴 | 変化なし（`origin/main` `3ea388e`、`origin/fix/3-docker-compose-baselines` `c70af28`、タグ0件）。再検査していません |
| ツール | §3.1と同じ（`detect-secrets` 1.5.0 ＋ 自作の正規表現スキャナ） |

結果（値は非掲載）:

| 分類 | 結果 |
|---|---|
| `detect-secrets`（差分blob） | 検出0件 |
| 認証情報クラス（blob / commit message / PR本文） | いずれも**0件** |
| バイナリ・非UTF-8 blob | 0件 |
| 64桁hex・IPv4形式 | `cuda/uv.lock`（再生成分）のwheel checksumとバージョン文字列のみ |
| メールアドレス | **blob: 2種類**（`docs/generated-artifacts.md`に検査結果として記載した`git@github.com`と`noreply@anthropic.com`。いずれもサービス用アドレスで、個人のアドレスではない）／**commit message: 2種類**（コミット作者と`noreply@anthropic.com`。gitのcommitter情報）／**PR本文: 2種類**（blobと同じサービス用アドレス） |
| `/home/<user>`形式の絶対パス | blob・commit message・PR本文いずれも0件 |
| 40桁hex | blob 0件。commit message中の自コミットSHA（14件、`%H`書式）のみ |

この節自身の追加差分についても、push前に同じパターンで検査し、認証情報クラス・絶対パス・新規の個人アドレスが含まれないことを確認しています。

### 3.6 検査できなかったもの（「問題なし」に含めていない）

- `run-*.wandb` 18ファイル（非UTF-8のprotobuf/LevelDB形式）とモデル重み（`.pt`／`.pth`）、`.npy`: バイト列としてのパターン検査と、`.pt`についてはzipメンバー単位（55メンバー、うち26がUTF-8デコード可能）の検査まで実施し、認証情報クラスの検出は0件。ただし、**レコード境界をまたいで分割された文字列、非UTF-8エンコード、数値配列そのものに埋め込まれた値は、テキストパターンでは検出できません**。これらを「検査済みで問題なし」とは扱えません。
- 到達不能（dangling）オブジェクト、およびGitHub側にのみ残りうる過去のref・PRスナップショット: クローン側から列挙できないため未検査です。
- 高エントロピー検出は`detect-secrets`のUTF-8テキスト解釈に依存するため、上記バイナリ群には適用されていません（そのため自作スキャナで補完しています）。

## 4. 今後の運用

- W&Bのローカルログ（`wandb/`）は生成物として扱い、Gitでは追跡しない。実行の証跡は`<output-dir>/artifacts/<run_id>/`のrun artifacts（`metadata.json`ほか）が担う。
- ダミー重みは必要なときに生成する。既定値は `input_dim=5000, hidden_dim=128, num_blocks=3, pc_dim=200, seed=42` で、同じPyTorch版・同じ引数なら同じ重みが得られる（seed固定）。検証済みベースラインの実行には不要。
- テスト用のfixtureは実行中に一時ディレクトリへ生成する方針で、リポジトリへ追加しない（`tests/test_cpu_smoke.py`のsynthetic fixtureが該当）。

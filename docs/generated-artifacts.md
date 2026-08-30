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
| 40桁hex文字列 | 4種類。うち3種類は本リポジトリのgitオブジェクト（W&Bが記録したgit commit）として解決した。残り1種類はバイナリのrun記録内で、W&Bがgit commitを保持する位置に現れるが、現在のリポジトリのオブジェクトとしては解決しない（履歴から失われたcommitと考えられる）。認証情報を示すラベルは付いていない |
| メールアドレス形式 | 3種類。`wandb-metadata.json`と`config.yaml`の`email`（gitのauthor email）と`remote`（gitのremote URL）フィールドに出現 |
| `/home/<user>`形式の絶対パス | 多数（実行時の作業ディレクトリ・依存パス） |
| IPv4形式 | `requirements.txt`のみ。バージョン文字列の誤検出と判断 |

### 判断と限界

- 認証情報として断定できる値は検出されなかった。ただし**これは追跡ツリー内のファイルに対する機械的検査であり、全履歴・全ログの秘密情報検査を完了したものではない**。過去のcommitに含まれる内容は、追跡停止では消えない。
- 実行環境のユーザー名を含む絶対パスとgitのauthor emailが公開履歴に残っている。これはコミット作者情報として既に公開されている情報と同種であり、失効・ローテーション対象ではない。
- もし今後、実際の認証情報が見つかった場合は、**まず失効・ローテーションを行う**。履歴の書き換えは影響範囲と承認を別途確認する（本対応では履歴書換え・force pushは行っていない）。

## 3. 今後の運用

- W&Bのローカルログ（`wandb/`）は生成物として扱い、Gitでは追跡しない。実行の証跡は`<output-dir>/artifacts/<run_id>/`のrun artifacts（`metadata.json`ほか）が担う。
- ダミー重みは必要なときに生成する。既定値は `input_dim=5000, hidden_dim=128, num_blocks=3, pc_dim=200, seed=42` で、同じPyTorch版・同じ引数なら同じ重みが得られる（seed固定）。検証済みベースラインの実行には不要。
- テスト用のfixtureは実行中に一時ディレクトリへ生成する方針で、リポジトリへ追加しない（`tests/test_cpu_smoke.py`のsynthetic fixtureが該当）。

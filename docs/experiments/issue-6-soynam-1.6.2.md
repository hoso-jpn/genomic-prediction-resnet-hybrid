# GBLUP / ResNet 実測比較レポート（Issue #6・CRAN SoyNAM 1.6.2）

本書は、RTX 5090実機で完走したGBLUP・ResNetの本実験について、保存済み成果物のみを根拠に集計した実測レポートです。
手順・事前定義・計測範囲の規定は[比較実験の手順と未検証範囲](../comparison-experiment.md)、データ出典は[data-provenance.md](../data-provenance.md)を参照してください。

数値はすべて保存済みOOF予測・`metadata.json`・`comparison.json`から再計算して照合したものです。個体別データは本書に含みません。

## 1. 実験目的と事前定義

目的は、**同一のleave-one-family-out（LOFO）分割の上で**GBLUPとResNetの予測精度と計算コストを比較し、深層学習側の優位性を前提に置かずに実測値を記録することです。

探索範囲は事前定義した**1候補のみ**です（各ハイパーパラメータの候補は1値）。外側test結果を見た設定変更は行っていません。seedは候補選択に使わず、宣言した3 seedすべてを報告します。内側family検証はepochの選択だけに使用します。

| 事前定義項目 | 値 |
|---|---|
| dataset_id | `cran-soynam-1.6.2-yield-blup-false` |
| data_kind / device | real / cuda |
| seeds | 42, 43, 44 |
| max_epochs / patience | 200 / 20 |
| batch_size / pca_components | 64 / 64 |
| max_sample_missing_rate | 1.0 |
| gblup / resnet marker observed rate | 0.1 / 0.9 |
| threads | 1 |
| bootstrap | family単位・2,000反復・seed 42 |
| config_hash | `sha256:30bfab1f7bd3b9bedb047bdafae314ef419c2fe14f015c3af59674d9539f96d1` |
| split_plan_hash | `sha256:a9ae6cdebaff0f88d42ca07d8afb5ab5876a1502e3640088133d7dded33dd29a` |
| outer_split_hash | `sha256:575f86a2d1d867eaa3192c9c774475a7901524715e2f94e65fbdb3228a7a74bd` |
| git_commit | `1b59f3b15ce97496a7ecbd509a1eabf222addb49` |

実行は2026-09-18T20:20:43Z開始・20:47:52Z終了（wall 1,629秒、終了コード0）。各runの成果物書き出し時刻は GBLUP 20:27:05Z、seed 42 20:33:44Z、seed 43 20:40:05Z、seed 44 20:47:04Z。

全体wall 1,629秒と4 runの内部計測合計 約1,521.8秒の差（約107秒）は、**各runの内部計測範囲外で生じたオーバーヘッドであり、内訳は未計測**です。`compare_baselines.py run`は4モデルを同一コンテナ内の逐次Python子プロセスとして起動しており（`subprocess.run`）、wrapperの記録上コンテナ生成は**1回**です。したがってこの差をコンテナ起動回数に帰属させることはできません。候補にはPythonインタプリタ起動とimport、計測開始前の入力読み込み、計測終了後のartifact直列化、report生成が含まれますが、**処理別の時間は測定していません**。

## 2. データ・形質・除外基準・出典

| 項目 | 値 |
|---|---|
| 出典 | CRAN SoyNAM 1.6.2（GPL-3）、tarball SHA-256 `0bd87f7b101456006a42a11679809349f7545b95dee0b43d954aa5b642995aaa` |
| 形質 | `Yld (kg/ha)` |
| 表現型モデル | `lmer(yield ~ (1|environ) + (1|strain))`、REML、`use.check=FALSE` |
| 除外 | 欠測yieldの354行のみ（source 60,744行 → model frame 60,390行、18環境） |
| 規模 | 5,142 sample / 39 family / 4,312 marker |
| genotype | `gen.qa`の0/1/2 dosageをそのまま使用（A/H/Bへ呼び替えない）、欠測はNA |
| genotype欠測率 | 0.25647185786375654 |

`--max-sample-missing-rate 1.0`のため、**sample QCによる個体除外は行われていません**（5,142個体すべてが評価対象）。marker側の閾値はモデルごとに異なります（§3）。

## 3. 共通LOFOとモデル別前処理

外側分割は39 foldのLOFOで、GBLUPと全ResNet seedが**同一の`split-plan.json`を消費**します。5,142個体は各1回だけouter testへ出現し、重複・欠落はありません（4 runのOOF個体集合は完全一致）。

前処理はpipeline由来の差が残るため、**アーキテクチャ単独の比較にはなりません**。

| 条件 | GBLUP | ResNet |
|---|---|---|
| marker観測率閾値 | > 0.1 | >= 0.9 |
| MAF閾値 | 0.05 | 0.01 |
| 特徴変換 | 学習平均imputation、VanRaden-1 | 学習平均imputation、標準化、PCA 64成分 |
| 選択 | 学習foldのREML | 内側family検証でepoch選択、外側学習データで再fit |
| 実測の採用marker数（39 fold） | min 4310 / median 4312 / max 4312（4,312中、平均 4311.8 = 100.0%） | min 549 / median 561 / max 641（4,312中、平均 570.5 = 13.2%） |

genotype欠測率が25.6%であるため、`>= 0.9`の観測率閾値はmarkerの大半を除外します。**実測でGBLUPは4,312 markerのほぼ全量、ResNetは約13%（平均570.5本）しか使っていません。** これは事前定義した`--gblup-marker-rate 0.1` / `--resnet-marker-rate 0.9`から生じる既知の非対称で、本実験の設計に含まれます。

ResNetはさらにPCAで64成分へ圧縮します（採用markerの分散の平均98.4%を保持）。

## 4. 全seedの精度と不確実性

### 4.1 pooled指標とmacro指標

**pooled相関**は5,142個体を1本のベクトルとして計算したPearson相関、**macro相関**は39 familyそれぞれのPearson相関の非加重平均です。両者は別物で、本実験では大きく食い違います（§8・§9）。

| Run | pooled Pearson r | pooled RMSE (kg/ha) | macro family r | macro family RMSE (kg/ha) |
|---|---:|---:|---:|---:|
| GBLUP | 0.185615 | 268.0577 | 0.209370 | 252.3619 |
| ResNet seed 42 | -0.073657 | 299.2253 | 0.090372 | 272.3952 |
| ResNet seed 43 | 0.072782 | 274.5149 | 0.101164 | 256.2210 |
| ResNet seed 44 | -0.069815 | 284.1216 | 0.094902 | 261.9485 |

相関が未定義となったfamilyはありません（全4 runで有効family数 39/39、除外なし）。各runのOOFは5,142行・重複0・39 family。

上表のpooled値は`comparison.json`の保存値とOOFからの再計算が一致します（r: |差| < 1e-12、RMSE: |差| < 1e-9）。

### 4.2 seed間変動（ResNet、n=3）

| 指標 | 平均 | 標本SD | 最小 | 最大 | 範囲 |
|---|---:|---:|---:|---:|---:|
| pooled r | -0.023563 | 0.083460 | -0.073657 | 0.072782 | 0.146439 |
| pooled RMSE (kg/ha) | 285.953952 | 12.456682 | 274.514926 | 299.225315 | 24.710390 |
| macro family r | 0.095479 | 0.005419 | 0.090372 | 0.101164 | 0.010792 |
| macro family RMSE (kg/ha) | 263.521596 | 8.201014 | 256.221049 | 272.395197 | 16.174147 |

**pooled rはseedによって符号が反転します**（−0.0737 / +0.0728 / −0.0698）。一方macro family rは3 seedとも正で、変動もはるかに小さい（SD 0.0054）。この不一致の由来は§9で扱います。

**seed間変動に含まれるもの**: 外側LOFO splitは`split-plan.json`で固定され3 seedで共通ですが、seedは重みの初期値だけでなく次にも影響します。

- **内側validation familyの選択** — `select_validation_family()`は`families[(seed + fold_index) % families.size]`を返すため、seedごとに別のfamilyが検証に使われます。実測で**39 fold全てにおいて3 seedの検証familyが互いに異なります**。
- **選択段階の前処理** — 検証familyが変わると学習側の集合が変わり、marker maskとPCAの当てはめが変わります（実測で選択段階の採用marker数が seedごとに異なる）。
- **学習の確率的要素** — `fold_seed = seed + fold_index * 100`、最終refitは`fold_seed + 1`で初期化・シャッフルされます。
- **`best_epoch`** — 上記の結果としてseedごとに変わり、最終refitの学習epoch数も変わります。

したがってseed間変動は「同一条件で初期値だけを変えた反復」ではありません。

### 4.3 paired family bootstrap

`comparison.json`のbootstrapは、**familyごとにResNet 3 seedの指標を平均し、そこからGBLUPの同family指標を引いた差**をfamily単位でリサンプリングしたものです。seed間で予測値を平均したensembleの評価ではありません（両者の区別は§9）。

| 指標 | family数 | 平均差 | 95%区間 下限 | 95%区間 上限 | 反復 | seed | 単位 |
|---|---:|---:|---:|---:|---:|---:|---|
| r | 39 | -0.113890438 | -0.151011197 | -0.078717521 | 2000 | 42 | family |
| RMSE (kg/ha) | 39 | 11.159692122 | 2.491718891 | 20.121442136 | 2000 | 42 | family |

符号の約束（[comparison-experiment.md](../comparison-experiment.md)）は「RMSE差は負、相関差は正がResNet側の改善」です。**実測は相関差が負・RMSE差が正で、いずれもResNetがGBLUPに及ばない向き**を示します。両区間とも0を含みません。

ただしこれは記述的な区間です。LOFOの学習集合は互いに大きく重複し、family数は39、3 seedは同一splitを共有します。独立反復に基づく有意性検定ではありません。

### 4.4 family別指標

`n`はouter testの個体数。ResNet列は各seedのfamily相関、`seed平均`はbootstrapが用いる3 seedの平均です。

| Family | n | GBLUP r | GBLUP RMSE | seed42 r | seed43 r | seed44 r | ResNet seed平均 r | 差 (ResNet−GBLUP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NAM02 | 98 | 0.0850 | 187.21 | -0.1234 | -0.1229 | -0.0638 | -0.1034 | -0.1883 |
| NAM03 | 137 | 0.1216 | 286.83 | 0.0668 | -0.0606 | -0.0170 | -0.0036 | -0.1252 |
| NAM04 | 136 | 0.3887 | 193.86 | 0.1139 | 0.0948 | 0.0811 | 0.0966 | -0.2921 |
| NAM05 | 139 | 0.1898 | 238.61 | 0.2826 | 0.1709 | -0.0068 | 0.1489 | -0.0409 |
| NAM06 | 140 | 0.2606 | 307.07 | -0.0002 | 0.0222 | 0.0063 | 0.0094 | -0.2512 |
| NAM08 | 138 | 0.1632 | 251.21 | 0.1239 | 0.2015 | 0.0618 | 0.1291 | -0.0342 |
| NAM09 | 137 | 0.2700 | 164.30 | 0.0826 | 0.0356 | 0.0436 | 0.0539 | -0.2161 |
| NAM10 | 139 | 0.2690 | 205.37 | 0.1656 | 0.1905 | 0.2349 | 0.1970 | -0.0720 |
| NAM11 | 123 | 0.2733 | 223.40 | 0.2453 | 0.1885 | 0.2438 | 0.2259 | -0.0474 |
| NAM12 | 138 | 0.2741 | 302.07 | 0.1139 | 0.1411 | 0.0959 | 0.1170 | -0.1571 |
| NAM13 | 137 | 0.1850 | 184.93 | 0.2710 | 0.2621 | 0.2919 | 0.2750 | +0.0900 |
| NAM14 | 137 | 0.1400 | 222.90 | 0.0778 | -0.0569 | 0.0329 | 0.0179 | -0.1221 |
| NAM15 | 139 | 0.0575 | 203.93 | -0.0460 | -0.0830 | -0.0910 | -0.0733 | -0.1308 |
| NAM17 | 135 | 0.2298 | 190.48 | 0.1310 | 0.1114 | 0.1030 | 0.1151 | -0.1146 |
| NAM18 | 136 | 0.1133 | 242.14 | 0.0846 | 0.0964 | 0.1367 | 0.1059 | -0.0075 |
| NAM22 | 138 | 0.1220 | 218.61 | 0.2381 | 0.2411 | 0.1418 | 0.2070 | +0.0850 |
| NAM23 | 140 | -0.0102 | 231.44 | -0.0714 | -0.0110 | -0.1321 | -0.0715 | -0.0613 |
| NAM24 | 140 | 0.3792 | 197.79 | 0.2779 | 0.3000 | 0.3333 | 0.3037 | -0.0755 |
| NAM25 | 121 | 0.3014 | 219.27 | 0.0822 | 0.0908 | 0.1238 | 0.0989 | -0.2025 |
| NAM26 | 108 | 0.1953 | 193.61 | 0.2122 | 0.1803 | 0.2097 | 0.2008 | +0.0055 |
| NAM27 | 131 | 0.2285 | 283.98 | 0.1157 | 0.0879 | 0.1764 | 0.1267 | -0.1019 |
| NAM28 | 112 | 0.2185 | 184.11 | 0.1693 | 0.1001 | 0.0499 | 0.1064 | -0.1120 |
| NAM29 | 138 | 0.1525 | 159.66 | -0.0458 | 0.0309 | -0.0959 | -0.0369 | -0.1894 |
| NAM30 | 138 | 0.2665 | 209.60 | 0.1448 | 0.2273 | 0.1569 | 0.1763 | -0.0902 |
| NAM31 | 127 | 0.2331 | 165.13 | 0.0388 | 0.1189 | 0.0824 | 0.0800 | -0.1531 |
| NAM32 | 138 | 0.3315 | 160.37 | -0.1046 | -0.1779 | -0.1518 | -0.1448 | -0.4763 |
| NAM33 | 134 | 0.3470 | 317.48 | 0.1664 | 0.2157 | 0.2094 | 0.1971 | -0.1498 |
| NAM34 | 133 | 0.2846 | 205.85 | 0.1369 | 0.1342 | 0.0393 | 0.1035 | -0.1811 |
| NAM36 | 136 | 0.2996 | 312.43 | -0.2068 | 0.1655 | 0.0743 | 0.0110 | -0.2887 |
| NAM37 | 129 | 0.2376 | 219.26 | 0.2039 | 0.1800 | 0.2503 | 0.2114 | -0.0262 |
| NAM38 | 132 | 0.1744 | 189.52 | 0.2261 | 0.1549 | 0.1692 | 0.1834 | +0.0090 |
| NAM39 | 134 | 0.0553 | 314.48 | -0.0465 | 0.1714 | 0.2015 | 0.1088 | +0.0535 |
| NAM40 | 137 | 0.1599 | 345.69 | 0.0507 | -0.0225 | -0.0005 | 0.0092 | -0.1507 |
| NAM41 | 99 | 0.3092 | 177.10 | 0.0208 | 0.1519 | 0.2645 | 0.1457 | -0.1634 |
| NAM42 | 133 | 0.1634 | 556.17 | 0.0193 | 0.1380 | 0.0695 | 0.0756 | -0.0878 |
| NAM48 | 139 | 0.2742 | 375.12 | 0.2604 | 0.2024 | 0.2804 | 0.2477 | -0.0265 |
| NAM50 | 132 | 0.2148 | 434.40 | 0.0569 | 0.0676 | 0.0965 | 0.0737 | -0.1411 |
| NAM54 | 140 | 0.3222 | 409.95 | 0.0166 | 0.0069 | 0.0577 | 0.0271 | -0.2951 |
| NAM64 | 124 | -0.1159 | 356.77 | -0.0264 | -0.0006 | -0.0589 | -0.0286 | +0.0873 |

集計: GBLUPのfamily相関が正のfamilyは 37/39、ResNet seed平均が正のfamilyは 32/39。ResNet seed平均がGBLUPを上回ったfamilyは 6/39 です。

## 5. 計算コストと測定範囲

原値（秒・bytes）を成果物に保存されているとおりに示し、換算値を別表で併記します。`null`は非該当を意味し、0ではありません（未計測でもありません）。

| Run | wall_seconds | cpu_user_seconds | cpu_system_seconds | process_peak_rss_bytes | cuda_peak_allocated_bytes | cuda_peak_reserved_bytes |
|---|---:|---:|---:|---:|---:|---:|
| GBLUP | 337.6427506520413 | 329.414352 | 8.211834 | 2027925504 | null | null |
| ResNet seed 42 | 394.3279246839229 | 389.872122 | 4.041563 | 2584838144 | 303004160 | 545259520 |
| ResNet seed 43 | 376.43762843392324 | 371.893594 | 3.9625559999999997 | 2562420736 | 317366272 | 555745280 |
| ResNet seed 44 | 413.43497422500513 | 408.850374 | 4.09647 | 2570784768 | 310255616 | 564133888 |

換算:

| Run | wall | CPU user | peak RSS | CUDA allocated | CUDA reserved |
|---|---:|---:|---:|---:|---:|
| GBLUP | 337.6 s | 329.4 s | 1.89 GiB | 該当なし | 該当なし |
| ResNet seed 42 | 394.3 s | 389.9 s | 2.41 GiB | 289.0 MiB | 520.0 MiB |
| ResNet seed 43 | 376.4 s | 371.9 s | 2.39 GiB | 302.7 MiB | 530.0 MiB |
| ResNet seed 44 | 413.4 s | 408.9 s | 2.39 GiB | 295.9 MiB | 538.0 MiB |

### 測定範囲の区別

`run_measurements.py`の実装に基づき、次はそれぞれ別の量です。混同・合算しないでください。

| 量 | 取得方法 | 範囲 |
|---|---|---|
| wall_seconds | `time.perf_counter()`の差分 | CLI検証後〜全fold評価終了まで。import・artifact直列化は**範囲外** |
| cpu_user/system_seconds | `getrusage(RUSAGE_SELF)`の**差分** | 同上の区間。**自プロセスのみ**で子プロセスを含まない |
| process_peak_rss_bytes | `getrusage`の`ru_maxrss` | **プロセス生涯の絶対peak**（差分ではない）。importを含む。run間で合算不可 |
| cuda_peak_allocated/reserved_bytes | `torch.cuda.max_memory_allocated/reserved` | **PyTorch allocatorのみ**。driver確保・他プロセス・contextを含まない |
| ホストGPU観測値（§下表） | `nvidia-smi` 5秒間隔 | **device全体**の値。実験プロセス単独の値ではない |
| Docker CLI側 `/usr/bin/time -v` | wrapperプロセスの計測 | docker clientのみ（User 0.14 s）。**コンテナ内の処理を含まない** |

CPU時間は区間差分、peak RSSは生涯peakという非対称がある点に注意してください。GBLUPはCPU実装のため`cuda_*`が`null`です（未測定ではなく非該当）。

**未取得**: ホスト全体のCPU時間・RSS、コンテナcgroup単位の資源使用量、`cuda_driver_api_version`（`null`）、ResNetのfold単位wall時間、全体wallと内部計測の差（約107秒）の処理別内訳、ResNet選択段階の停止epochと停止理由、ResNet artifactの`external_logging`項目。これらは既知の制約として残します。**これらの取得だけを理由に本実験を再実行しません。**

### ホストGPU観測（nvidia-smi）

| 項目 | 値 |
|---|---|
| 計測期間 (UTC) | 2026-09-18T20:20:43Z 〜 2026-09-18T20:47:49Z |
| サンプル数 / 公称間隔 | 323 / 5 秒 |
| 実測間隔 | min 5 / median 5 / max 6 秒（平均 5.05） |
| 欠測 | 10秒を超える欠落なし。期待326サンプルに対し323（99.0%）で、差は`nvidia-smi`実行時間による位相ずれ |
| 最大GPU使用率 | 89 % |
| 最大GPUメモリ使用 | 1270 MiB / 32607 MiB |
| 最大電力 | 348.98 W |
| 最大温度 | 62 ℃ |
| 観測されたcompute process | 異なるPID 3件（ResNet 3 seedに対応）。同時刻に2件以上を観測したサンプルなし |

**測定対象はdevice全体**で、実験プロセス単独の値ではありません。デスクトップsessionの常駐分（約80 MiB）を含みます。
また、5秒間隔で観測された最大値であり、**それより短いスパイクは捕捉を保証しません**。監視ログは各サンプル時点でのcompute processの有無を示すのみで、サンプル間にGPUを占有した他プロセスが無かったことまでは証明しません。

## 6. provenance差分

canonical datasetのmanifestについて、**データの同一性**と**生成履歴の同一性**を分けて記録します。

### 6.1 参照build（先行commit）と本実験buildの比較

| 項目 | 本実験 | 先行参照build |
|---|---|---|
| 生成commit | `1b59f3b15ce97496a7ecbd509a1eabf222addb49` | `0057315dbcdefa554289001c75f258b0d1804f12` |
| builder `soynam_cran.py` SHA-256 | `eec573c290df84911e5425b8ca483115bac6f43355b9d5aa6c418407ffbbf95f` | `b7be7077ce895996a91ef76b5cbbbee1648d085fc514637b8d5ba972e5276bfa` |
| builder `scripts/build_soynam_canonical.R` SHA-256 | `778d4122bc93df555a8e81b122af46e3391b438bfe2870131e701395811f4e38` | `778d4122bc93df555a8e81b122af46e3391b438bfe2870131e701395811f4e38`（同一） |
| tarball SHA-256 | `0bd87f7b101456006a42a11679809349f7545b95dee0b43d954aa5b642995aaa` | 同一 |
| データファイル78件のSHA-256 | 78/78一致 | — |
| `joined_phenotype_sha256` | `22a632e2f8a1cea001bf5c4466c472d23df0331253e1ef7c97ca228664663c75` | 同一 |
| `marker_id_list_sha256` | `2f24bd7524cd3b7cf003fc8f5c99dffd540a03707d0f64e0f04c07dad001d7d4` | 同一 |
| `marker_id_file_sha256` | `07e7d257e524e06b69a504643b764aa3526cf0cb0f29f3b2dacedb03bd51c467` | 同一 |
| `sample_id_list_sha256` | `d1a43e4ac11abdd70fbbd99b96cb06e468ff130c5d02cba2e51e3245e3dcdfaf` | 同一 |
| `content_hash` | `4369cbcfe32da98b14183ee6fe3b3277fb87e93f7c7f29d605e9e5f22f2045c4` | `7bf9e4afff586537323b94f12aefe0e22f862f05ab47cc602914a0ff026307af` |
| `created_at` | `2026-09-18T04:47:59Z` | `2026-09-17T06:23:30Z` |

manifestをfield単位で展開すると**116 field中113が完全一致**し、相違は次の3つだけです。

1. `checksums.builder_sources.soynam_cran.py` — builderスクリプト自身のSHA-256
2. `content_hash` — 1に従属（`content_hash`は`created_at`を除くmanifest全体のcanonical JSON hash）
3. `created_at` — 生成時刻。`content_hash`の計算対象外

### 6.2 判定

- **データの同一性: 一致**。78ファイルのSHA-256、marker/sample/phenotypeの各identity hash、tarball、genotype欠測率、sample/family/marker数がすべて一致します。生成されたデータはbit単位で参照buildと同一です。
- **生成履歴の同一性: 不一致**。builder `soynam_cran.py`が異なるため`content_hash`が異なります。

`content_hash`はbuilder自身のSHA-256を入力に含むため、**builderを変更すればデータが同一でも必ず変化します**。参照値`7bf9e4af…`は先行commit `0057315`のbuilderに紐づく値であり、本実験commit `1b59f3b`からは再現できません。旧期待値は削除せず、上表に併記して保存します。

補強証拠として、先行commitでの2回のbuild（`created_at` `2026-09-17T06:23:30Z` と `2026-09-17T06:24:32Z`）は116 field中`created_at`のみが異なり、`content_hash`は両方とも`7bf9e4af…`でした。builderが同一ならcontent_hashが再現することを示します。

### 6.3 差分を生んだ変更の範囲

`0057315..1b59f3b`で本番コードのうち変更されたのは`soynam_cran.py`のみです（他はREADME・docs・tests・lock）。その差分は3 hunkで、実際の処理順では**出力先の事前条件チェック**と**manifest確定後のファイル公開処理**に分かれます。いずれも**データの計算内容ではありません**。

1. **build開始時（データ生成前）** — `build()`の出力先事前条件チェックに`not output_dir.is_dir()`を追加。出力先が空ディレクトリかどうかの検証条件であり、生成されるデータには影響しません。
2. **関数追加** — `publish_dataset()`の追加（同一filesystem上へcopy後、単一のdirectory renameで公開）。
3. **build末尾（manifest書き出し後）** — move loopを`publish_dataset()`呼び出しへ置換。

tarball読み込み、R builder実行、intermediates解析、genotype/phenotype生成、hash計算のいずれの関数も変更されていません。R builderもbyte単位で同一です。したがって**データ計算内容の変更ではなく、出力先検証と公開処理の変更**と判定します。

なお、manifest schemaとhash計算方式の変更は本書の対象外です。

## 7. 再実行に必要な環境と設定

| 項目 | 値 |
|---|---|
| commit | `1b59f3b15ce97496a7ecbd509a1eabf222addb49` |
| CUDA image ID | `sha256:b1d1fbde97042dd0832b98fb6fc74cd48dae4122b93ebded1486ebac6f44dc0f` |
| `Dockerfile.cuda` SHA-256 | `1ce6a4a37d77ac0e581f54d27e90600e7d5bca620349bd28f82fe3777f8e6c47` |
| `cuda/pyproject.toml` SHA-256 | `506649b1893b460e5efc3626187b39261f3b7cc671b2b066006b043b886feabb` |
| `cuda/uv.lock` SHA-256 | `ae3efbecd0517e1528dd85cfbadd1f30ae8ec881da6f8a9570fcf59c9c110de1` |
| environment_label | `cuda-13.0-torch-2.12.1` |
| torch / CUDA / cuDNN | 2.12.1+cu130 / 13.0 / 9.20.0（`92000`） |
| PyG / numpy / pandas / scikit-learn / scipy | 2.7.0 / 1.26.4 / 3.0.3 / 1.8.0 / 1.17.1 |
| Python | 3.11.16 |
| GPU / compute capability / driver | NVIDIA GeForce RTX 5090 / 12.0 / 595.84 |
| host OS / kernel / Docker | Ubuntu 24.04.5 LTS / 6.8.0-139-generic / 29.8.0 |

実行コマンドは[comparison-experiment.md](../comparison-experiment.md)のplan/run/reportに従い、`--experiment-dir`を新規ディレクトリにしたうえで、`GIT_COMMIT_SHA`と`WANDB_MODE=disabled`を渡します。W&Bは全runで無効です。

`content_hash`を再現したい場合は、canonical datasetを**同じcommitのbuilder**で生成してください（§6）。

## 8. 科学的な制約

- **事前定義した1候補の実験であり、広範な探索ではありません。** ResNetのハイパーパラメータは最適化していません。本結果はこの設定でのResNetの実測値であり、アーキテクチャの性能上限ではありません。
- **GBLUPとResNetには前処理差があり、アーキテクチャ単独の比較ではありません。** 実測でGBLUPは4,312 markerのほぼ全量、ResNetは約13%しか使用していません（§3）。
- **調整済み表現型の生成に全familyの観測が関与しています。** held-out familyの観測も分散成分・環境効果の推定に寄与するため、**完全に独立した外部検証や未知環境への予測ではありません**。比較できるのは、同じ調整済み表現型・同じsample集合・同じLOFO splitの上でのGBLUPとResNetの差だけです。
- **family bootstrapには制約があります。** LOFOの学習集合は互いに大きく重複し、family数は39、3 seedは同一splitを共有します。独立反復に基づく有意性検定ではなく、記述的な区間です。
- **本結果を小豆や他の形質へ一般化しません。** 対象はSoyNAMのyieldのみです。
- **実測から裏付けられない優位性を主張しません。** 本実験が支持するのは「この設定・この前処理条件下でResNetがGBLUPに及ばなかった」ことに限られ、ResNetが原理的に劣ることを示すものではありません。逆に、GBLUPの優位を他のデータ・形質へ拡張して主張することもしません。
- pooled相関は絶対値として小さく（GBLUP 0.186）、実用的な選抜性能を主張できる水準ではありません。

## 9. 補助診断（pooled相関とfamily内相関の乖離）

§4で見たとおり、ResNetはmacro family rが正（平均0.0955）でありながらpooled rはほぼ0〜負です。既存成果物から確認できた事実と、そこから立てた仮説を分けて記します。

### 9.1 確認できた事実

pooled相関を次の2つへ分けます。算出方法を明記します。

- **between-family r** — 39 familyそれぞれの実測平均と予測平均を1点ずつとし、**39点・family非加重**でPearson相関を取ったもの。
- **within-family r** — 各個体の実測値・予測値から**その個体が属するfamilyの平均を引いて中心化**し、中心化後の5,142個体を**1本のベクトルにまとめて**Pearson相関を取ったもの（個体単位の重み付け。familyごとに相関を出して平均する`macro family r`とは別物）。

これらは**保存済みOOFを使った診断であり、予測器の補正ではありません**。test familyの実測平均を予測へ差し戻すような操作は行っておらず、上表のpooled r列は§4.1と同一の値です。

| Run | between-family r | within-family r | pooled r |
|---|---:|---:|---:|
| GBLUP | 0.395730 | 0.207833 | 0.185615 |
| ResNet seed 42 | -0.184492 | 0.067239 | -0.073657 |
| ResNet seed 43 | 0.037340 | 0.104551 | 0.072782 |
| ResNet seed 44 | -0.286283 | 0.091036 | -0.069815 |

family平均値の散らばり（実測値の標準偏差は全run共通で193.20 kg/ha）と予測値の散らばりは次のとおりです。

| Run | 実測family平均のSD | 予測family平均のSD | 全体bias |
|---|---:|---:|---:|
| GBLUP | 193.20 | 14.31 | -6.39 |
| ResNet seed 42 | 193.20 | 86.06 | +2.77 |
| ResNet seed 43 | 193.20 | 43.73 | +6.55 |
| ResNet seed 44 | 193.20 | 42.80 | +10.28 |

また、内側family検証で選ばれた`best_epoch`は次のとおりです。

`resnet_baseline.py`の`_select_epoch()`では次の3つが別物です。混同しないでください。

- **`best_epoch`** — 選択段階で検証損失が最良だったepoch。artifactに記録される唯一のepoch値。
- **選択段階の停止epoch** — 連続非改善が`patience`（20）に達した時点、または`max_epochs`（200）到達時のepoch。**artifactに記録されていません**。`patience`は「連続非改善回数の上限」であり、epoch番号の上限ではありません。
- **最終refitのepoch数** — 外側学習データでの再fitは`best_epoch`回だけ学習します（`for _ in range(best_epoch)`）。

| Run | best_epoch min | median | mean | max |
|---|---:|---:|---:|---:|
| ResNet seed 42 | 1 | 16.0 | 15.97 | 46 |
| ResNet seed 43 | 2 | 10.0 | 14.51 | 80 |
| ResNet seed 44 | 2 | 16.0 | 17.33 | 48 |

事実として確認できるのは次の点です。

- ResNetのwithin-family相関は3 seedとも正（0.067〜0.105）で、family内の順位付けには弱い信号がある。
- ResNetのbetween-family相関は負または0近傍（−0.286〜+0.037）で、seedにより符号が変わる。
- GBLUPはbetween 0.396・within 0.208といずれも正。
- GBLUPの予測family平均のSDは14.31 kg/haで、実測の193.20 kg/haに対し強く縮小している。ResNetは42.80〜86.06 kg/haと大きい。
- `best_epoch`の中央値は10〜16。**選択段階の停止epochと停止理由はartifactに記録がなく未記録**のため、早期終了が発火したか`max_epochs`に到達したかは本成果物からは判定できない。この確認のための再学習・本番コード変更は行っていない。

### 9.2 仮説（未検証）

上記と整合する説明として、LOFOではhold-outしたfamilyの平均値を学習側から推定する手掛かりが乏しく、GBLUPは family平均をほぼ定数へ縮小することでpooled相関の悪化を避けている一方、ResNetは大きなfamily単位のオフセットを出力しており、その順序が実測と合わないためpooled相関が押し下げられている——という機序が考えられます。**これは仮説であり、本実験では検証していません。**

### 9.3 探索的追加集計: seed ensembleとの区別（事前定義外）

以下は**本実験のOOFを使った事前定義外の探索的追加集計**であり、主結果とは分離して扱います。`experiment.json`が宣言した出力ではなく、`comparison.json`にも含まれません。

2つの量を区別してください。

- **seed別指標の平均**（§4.3のbootstrapが使う量）— familyごとに各seedの相関・RMSEを計算し、その**指標を**3 seedで平均したもの。
- **予測値のensemble**（本節）— 個体ごとに3 seedの**予測値を**平均し、その平均予測で改めて相関・RMSEを計算したもの。

探索的に後者を計算すると pooled r = −0.038320、pooled RMSE = 280.13 kg/ha、macro family r = 0.103025 でした。**主結果ではなく、2つが別物であることを示すための数値です。** ensembleを本実験の成績として報告しません。

### 9.4 やっていないこと

test familyの実測値を使った補正（family平均の後付け調整など）は行っていません。そのような補正後の数値を改善した予測性能として報告することはしません。

PCA次元・marker閾値・学習率・モデル構造の変更による原因調査は本書の対象外で、後続Issue候補として分離します。

## 10. Issue #6 受入項目との対応

### 完了条件

| 受入項目 | 判定 | 根拠 |
|---|---|---|
| GBLUPとResNetを同一splitで比較できる | 充足 | 4 runが同一`split-plan.json`（`plan_hash sha256:a9ae6cde…`）を消費。`input_files`とsplit参照が4 runで一致。OOF個体集合も完全一致 |
| 全foldのOOF予測が保存される | 充足 | 39 fold・5,142個体分のOOFを4 run分保存。重複0・欠落0・39 family |
| 精度指標と計算コストが報告される | 充足 | §4（pooled/macro/family別/seed変動/bootstrap）と§5（wall・CPU・RSS・CUDA・ホストGPU） |
| 再実行に必要な設定と成果物が保存される | 充足（別媒体バックアップは未実施） | commit・image ID・lock hash・`experiment.json`・`split-plan.json`・run artifacts 6種に加え、実験成果物・canonical dataset・evidence・参照build・tarball・実験commitのsource・実行用CUDA imageを実験単位のアーカイブへ複製し、全215ファイルのSHA-256照合に成功（§11）。**同一ホスト上の保全**であり、別媒体・別ホストへのバックアップは未実施 |
| 結果から裏付けられない優位性を主張していない | 充足 | §8で制約を明示。bootstrapの符号規約に従い、ResNetが及ばない向きであることのみ記述 |

### 対応内容

| 項目 | 判定 | 根拠 |
|---|---|---|
| 使用データ、対象形質、除外基準を確定する | 充足 | §2 |
| 共通のLOFO splitを固定する | 充足 | §3、`outer_split_hash sha256:575f86a2…` |
| GBLUPを全foldで実行する | 充足 | 39/39 fold成功 |
| ResNetをGPU環境で全fold実行する | 充足 | 3 seed × 39 fold。`device_requested`/`device_resolved`ともcuda、`gpu_name` RTX 5090 |
| ハイパーパラメータ探索範囲を事前定義する | 充足 | `experiment.json`の`resnet_search_space`（各1候補）、`config_hash`で固定 |
| Pearson相関、RMSEなどの評価指標を算出する | 充足 | §4 |
| family別および全体の結果を集計する | 充足 | §4.1・§4.4 |
| 実行時間、GPU・CPU使用量、メモリ使用量を記録する | 充足 | §5 |
| seed違いによる変動を確認する | 充足 | §4.2 |
| 結果と制約をレポート化する | 充足 | 本書 |

判定の根拠となる成果物の所在はレビュー用のローカルinventoryに記録しています（公開文書には絶対パスを含めません）。

## 11. 成果物の保管

### 保管方針

- Git管理外・一時領域外に、**実験単位**で保管する。
- 明示的な削除指示があるまで自動削除しない。
- 原本は残し、**新規ディレクトリへの複製のみ**を行う。
- **同一ホスト上の保全**と**別媒体バックアップ**を区別する。

### 実施内容

| 項目 | 値 |
|---|---|
| 実験識別子 | `20260918T202043Z-30bfab1f`（run開始UTC ＋ config_hash先頭8桁） |
| 対象範囲 | 実験成果物36 / canonical dataset 79 / evidence 12 / 参照build・tarball 3 / 実験commitのsource 82 / 再計算スクリプト1 / CUDA imageアーカイブ1 |
| 総容量 | 約2.9 GB（3,027,354,097 bytes）、216ファイル |
| 検証方式 | 複製後に原本と相対パス・SHA-256で全件照合し、`MANIFEST.sha256`（215エントリ、自分自身を含まない）で再検証 |
| 検証結果 | 全件一致 |
| CUDA image | `docker image save` 終了コード0。archive SHA-256 `a9e065dc31cbb524b4e82748fa39834dacffa678ea8dd79777846ad1861114b1`。archive内`index.json`のdigestが保存元image ID `sha256:b1d1fbde9704…` と一致 |
| 別媒体バックアップ | **未実施** |

保管先の絶対パスは非公開のローカルinventoryに分離しています。アーカイブは個体別データとimageを含むため公開しません。

### 未実施

`docker image load`による**復元試験は行っていません**。復元手順はアーカイブ内の`RESTORE.md`に記載していますが、動作を検証した記録ではありません。復元したimageでの再実行・再学習も行っていません。

### 保管前のリスクと現状

保管前、canonical datasetの参照buildとSoyNAM tarballは一時領域（`/tmp`）にのみ存在していました。確認時点（2026-09-19 UTC）で`/tmp`には`systemd-tmpfiles`の`D /tmp 1777 root root 30d`が適用されており、清掃タイマーはactiveでした。**具体的な削除日時は確定していません**（実際の削除は清掃実行時のファイル状態に依存し、本作業では清掃の設定変更も実行も行っていません）。一時領域に依存している状態自体がリスクであったため、上記のとおり一時領域外へ複製し、照合済みです。原本および`/tmp`の参照buildは移動・削除していません。


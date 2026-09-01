# -*- coding: utf-8 -*-
"""転移学習の配線確認用に、CNN部分のランダム初期重みを生成するスクリプト。

生成物（既定で `pretrained_models/dummy_cnn_weights.pt`）は事前学習済み
モデルではなく、ランダム初期化された重みである。予測精度の証跡には
ならないため、Gitでは追跡せず（.gitignoreの`*.pt`）、必要なときに
このスクリプトで生成する。検証済みベースライン（gblup_baseline.py /
resnet_baseline.py）の実行には不要。

再生成の条件（既定値）:
  input_dim=5000, hidden_dim=128, num_blocks=3, pc_dim=200, seed=42

seedを固定しているため、同じPyTorch版・同じ引数であれば同じ重みが
得られる。
"""
import argparse
import os

import torch

from model import GatedGenomicResNet

DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "./pretrained_models"
DEFAULT_FILENAME = "dummy_cnn_weights.pt"
ARCHITECTURE = {
    "input_dim": 5000,
    "hidden_dim": 128,
    "num_blocks": 3,
    "pc_dim": 200,
}

def create_dummy_weights(output_dir=DEFAULT_OUTPUT_DIR, seed=DEFAULT_SEED):
    """ランダム初期化したCNN部分のstate_dictを保存する。"""
    print("擬似的な事前学習済み重みを作成中（ランダム初期重み。事前学習の成果ではない）...")
    torch.manual_seed(seed)

    # モデルを一度初期化して、アーキテクチャを構築
    # パラメータは実際の学習と一致させる必要はない
    dummy_model = GatedGenomicResNet(**ARCHITECTURE)

    # CNN部分のstate_dictのみを抽出
    cnn_weights = dummy_model.cnn_path.state_dict()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, DEFAULT_FILENAME)
    torch.save(cnn_weights, save_path)

    print(f"重みを {save_path} に保存しました（seed={seed}, {ARCHITECTURE}）。")
    print("保存されたレイヤー:")
    for key in cnn_weights.keys():
        print(f"- {key}")
    return save_path

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    create_dummy_weights(output_dir=args.output_dir, seed=args.seed)

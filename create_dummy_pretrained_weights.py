# -*- coding: utf-8 -*-
import torch
from model import GatedGenomicResNet
import os

def create_dummy_weights():
    """
    転移学習のテスト用に、モデルのCNN部分の初期重みを保存するスクリプト。
    実際には、このファイルは大規模データセットでの事前学習の結果として生成される。
    """
    print("擬似的な事前学習済み重みを作成中...")

    # モデルを一度初期化して、アーキテクチャを構築
    # パラメータは実際の学習と一致させる必要はない
    dummy_model = GatedGenomicResNet(
        input_dim=5000,   # 適当な値
        hidden_dim=128,   # スイープで見つかった最適な値に近いもの
        num_blocks=3,
        pc_dim=200
    )

    # CNN部分のstate_dictのみを抽出
    cnn_weights = dummy_model.cnn_path.state_dict()

    # 保存
    output_dir = "./pretrained_models"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "dummy_cnn_weights.pt")
    torch.save(cnn_weights, save_path)

    print(f"重みを {save_path} に保存しました。")
    print("保存されたレイヤー:")
    for key in cnn_weights.keys():
        print(f"- {key}")

if __name__ == "__main__":
    create_dummy_weights()

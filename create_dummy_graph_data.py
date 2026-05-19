# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

def create_dummy_graph_data():
    """
    GNNへの入力として使用する、擬似的なグラフデータを作成する。
    実際には、このデータはバイオインフォマティクスのパイプラインによって生成される。
    """
    print("擬似的なグラフデータを作成中...")

    # --- パラメータ設定 ---
    # プロジェクトの実際のデータに合わせて調整する必要がある
    num_snps = 4312  # preprocess.pyの出力に合わせる
    num_genes = 500   # 仮の遺伝子数
    avg_degree = 10 # 遺伝子あたりの平均的な接続数

    # --- 1. SNPと遺伝子のマッピングを作成 --- 
    # 各SNPがランダムにいずれかの遺伝子に属するように割り当てる
    snp_gene_mapping = np.random.randint(0, num_genes, size=num_snps)
    snp_to_gene_df = pd.DataFrame({
        'snp_id': range(num_snps),
        'gene_id': snp_gene_mapping
    })

    # --- 2. 遺伝子間の隣接行列（エッジリスト）を作成 ---
    # ランダムなグラフ（Erdos-Renyiモデル）を生成
    num_edges = num_genes * avg_degree
    source_nodes = np.random.randint(0, num_genes, size=num_edges)
    target_nodes = np.random.randint(0, num_genes, size=num_edges)
    
    # 自己ループと重複エッジを除去
    edge_list = set()
    for i in range(num_edges):
        u, v = source_nodes[i], target_nodes[i]
        if u != v:
            edge_list.add(tuple(sorted((u, v))))

    # GCNConvは無向グラフを前提とするため、両方向のエッジを保存する
    forward_edges = list(edge_list)
    reverse_edges = [(v, u) for u, v in forward_edges]
    adj_df = pd.DataFrame(forward_edges + reverse_edges, columns=['source', 'target'])

    # --- 保存 --- 
    output_dir = "./processed_data_hy"
    os.makedirs(output_dir, exist_ok=True)
    
    map_path = os.path.join(output_dir, "snp_to_gene_map.csv")
    adj_path = os.path.join(output_dir, "gene_adj.csv")

    snp_to_gene_df.to_csv(map_path, index=False)
    adj_df.to_csv(adj_path, index=False)

    print(f"SNP-遺伝子マッピングを {map_path} に保存しました。 (Shape: {snp_to_gene_df.shape})")
    print(f"遺伝子隣接リストを {adj_path} に保存しました。 (Shape: {adj_df.shape})")

if __name__ == "__main__":
    create_dummy_graph_data()

# -*- coding: utf-8 -*-
"""GNNの配線確認用に、擬似的な遺伝子グラフとSNP→遺伝子対応を生成する。

生成物（`processed_data_hy/snp_to_gene_map.csv`・`gene_adj.csv`）は
ランダムグラフであり、実際の遺伝子ネットワークではない。train_gnn.py は
experimentalであり、検証済みベースラインの実行には不要。

エッジは `gene_graph.to_bidirectional_edges` で「重複のない双方向表現」
（各方向1本）に正規化して保存する。loader（train_gnn.load_data）は同じ
契約を検証して読み込むため、生成側と読込側の表現が一致する。
"""
import argparse
import os

import numpy as np
import pandas as pd

import gene_graph

DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "./processed_data_hy"
DEFAULT_NUM_SNPS = 4312   # preprocess.pyの出力に合わせる
DEFAULT_NUM_GENES = 500   # 仮の遺伝子数
DEFAULT_AVG_DEGREE = 10   # 遺伝子あたりの平均的な接続数

def build_dummy_graph(num_snps, num_genes, avg_degree, seed):
    """SNP→遺伝子対応と、双方向エッジのDataFrameを生成する。"""
    rng = np.random.default_rng(seed)

    # 各SNPがランダムにいずれかの遺伝子に属するように割り当てる
    snp_gene_mapping = rng.integers(0, num_genes, size=num_snps)
    snp_to_gene_df = pd.DataFrame({
        'snp_id': range(num_snps),
        'gene_id': snp_gene_mapping
    })

    # ランダムなグラフ（Erdos-Renyiモデル）を生成し、自己ループを除く
    num_edges = num_genes * avg_degree
    source_nodes = rng.integers(0, num_genes, size=num_edges)
    target_nodes = rng.integers(0, num_genes, size=num_edges)
    undirected_pairs = [
        (int(u), int(v)) for u, v in zip(source_nodes, target_nodes) if u != v
    ]

    # 重複除去と双方向化は共通実装に任せる（loaderが検証する契約と同一）
    adj_df = gene_graph.edge_frame(
        gene_graph.to_bidirectional_edges(undirected_pairs)
    )
    return snp_to_gene_df, adj_df

def create_dummy_graph_data(
    output_dir=DEFAULT_OUTPUT_DIR,
    num_snps=DEFAULT_NUM_SNPS,
    num_genes=DEFAULT_NUM_GENES,
    avg_degree=DEFAULT_AVG_DEGREE,
    seed=DEFAULT_SEED,
):
    print("擬似的なグラフデータを作成中（experimental。実際の遺伝子ネットワークではない）...")
    snp_to_gene_df, adj_df = build_dummy_graph(num_snps, num_genes, avg_degree, seed)

    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "snp_to_gene_map.csv")
    adj_path = os.path.join(output_dir, "gene_adj.csv")

    snp_to_gene_df.to_csv(map_path, index=False)
    adj_df.to_csv(adj_path, index=False)

    print(f"SNP-遺伝子マッピングを {map_path} に保存しました。 (Shape: {snp_to_gene_df.shape})")
    print(f"遺伝子隣接リストを {adj_path} に保存しました。 (Shape: {adj_df.shape}, seed={seed})")
    return map_path, adj_path

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-snps", type=int, default=DEFAULT_NUM_SNPS)
    parser.add_argument("--num-genes", type=int, default=DEFAULT_NUM_GENES)
    parser.add_argument("--avg-degree", type=int, default=DEFAULT_AVG_DEGREE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    create_dummy_graph_data(
        output_dir=args.output_dir,
        num_snps=args.num_snps,
        num_genes=args.num_genes,
        avg_degree=args.avg_degree,
        seed=args.seed,
    )

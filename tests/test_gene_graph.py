"""Regression tests for the gene-graph edge contract and the shared loss."""

import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

import gene_graph
from create_dummy_graph_data import build_dummy_graph
from losses import CorrelationLoss

REPO_ROOT = Path(__file__).resolve().parent.parent


class EdgeContractTest(unittest.TestCase):
    def test_generator_output_has_each_direction_once(self) -> None:
        _, adj_frame = build_dummy_graph(num_snps=40, num_genes=8, avg_degree=4, seed=7)
        edge_index = gene_graph.load_edge_index(adj_frame, num_genes=8)

        edges = [tuple(pair) for pair in edge_index.t().tolist()]
        self.assertEqual(len(edges), len(set(edges)))
        # Every undirected pair contributes exactly two rows: (u, v) and
        # (v, u). The loader adds no reversed copy of its own.
        undirected = {tuple(sorted(edge)) for edge in edges}
        self.assertEqual(len(edges), 2 * len(undirected))
        for source, target in edges:
            self.assertIn((target, source), set(edges))
            self.assertNotEqual(source, target)

    def test_generator_is_deterministic_for_a_seed(self) -> None:
        first_map, first_adj = build_dummy_graph(20, 6, 3, 11)
        second_map, second_adj = build_dummy_graph(20, 6, 3, 11)

        pd.testing.assert_frame_equal(first_map, second_map)
        pd.testing.assert_frame_equal(first_adj, second_adj)

    def test_bidirectional_normalization_is_idempotent(self) -> None:
        once = gene_graph.to_bidirectional_edges([(0, 1), (2, 1), (1, 0)])
        twice = gene_graph.to_bidirectional_edges(once)

        self.assertEqual(once, twice)
        self.assertEqual(once, [(0, 1), (1, 0), (1, 2), (2, 1)])

    def test_normalization_rejects_self_loops_and_negative_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "self-loops"):
            gene_graph.to_bidirectional_edges([(1, 1)])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            gene_graph.to_bidirectional_edges([(-1, 2)])
        with self.assertRaisesRegex(ValueError, "at least one undirected edge"):
            gene_graph.to_bidirectional_edges([])

    def test_loader_rejects_a_one_directional_edge_list(self) -> None:
        frame = gene_graph.edge_frame([(0, 1), (1, 2), (2, 1)])
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            gene_graph.load_edge_index(frame, num_genes=3)

    def test_loader_rejects_duplicated_rows(self) -> None:
        frame = gene_graph.edge_frame([(0, 1), (1, 0), (0, 1), (1, 0)])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            gene_graph.load_edge_index(frame, num_genes=2)

    def test_loader_rejects_self_loops(self) -> None:
        frame = gene_graph.edge_frame([(0, 1), (1, 0), (1, 1)])
        with self.assertRaisesRegex(ValueError, "self-loops"):
            gene_graph.load_edge_index(frame, num_genes=2)

    def test_loader_rejects_out_of_range_and_empty_input(self) -> None:
        frame = gene_graph.edge_frame([(0, 5), (5, 0)])
        with self.assertRaisesRegex(ValueError, r"\[0, 3\)"):
            gene_graph.load_edge_index(frame, num_genes=3)

        with self.assertRaisesRegex(ValueError, "no edges"):
            gene_graph.load_edge_index(gene_graph.edge_frame([]), num_genes=3)

    def test_loader_rejects_wrong_columns_and_non_integer_ids(self) -> None:
        renamed = pd.DataFrame({"from": [0, 1], "to": [1, 0]})
        with self.assertRaisesRegex(ValueError, "columns"):
            gene_graph.load_edge_index(renamed, num_genes=2)

        floats = pd.DataFrame({"source": [0.0, 1.0], "target": [1.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "integer gene IDs"):
            gene_graph.load_edge_index(floats, num_genes=2)


class SnpToGeneMapTest(unittest.TestCase):
    def test_gene_count_is_derived_from_the_map(self) -> None:
        values = torch.tensor([0, 2, 1, 2], dtype=torch.long)
        self.assertEqual(gene_graph.validate_snp_to_gene_map(values, snp_count=4), 3)

    def test_length_mismatch_and_bad_values_are_rejected(self) -> None:
        values = torch.tensor([0, 1], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "does not match"):
            gene_graph.validate_snp_to_gene_map(values, snp_count=3)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            gene_graph.validate_snp_to_gene_map(
                torch.tensor([-1, 0], dtype=torch.long), snp_count=2
            )

        with self.assertRaisesRegex(ValueError, "integer gene IDs"):
            gene_graph.validate_snp_to_gene_map(torch.tensor([0.0, 1.0]), snp_count=2)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            gene_graph.validate_snp_to_gene_map(
                torch.zeros((2, 2), dtype=torch.long), snp_count=4
            )


class CorrelationLossTest(unittest.TestCase):
    def _loss_and_grad(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        predictions = predictions.clone().requires_grad_(True)
        value = CorrelationLoss()(predictions, targets)
        value.backward()
        return float(value), predictions.grad

    def test_perfect_correlation_gives_minus_one(self) -> None:
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        value, grad = self._loss_and_grad(predictions, predictions.clone())
        self.assertAlmostEqual(value, -1.0, places=5)
        self.assertTrue(torch.isfinite(grad).all())

    def test_constant_input_is_finite(self) -> None:
        value, grad = self._loss_and_grad(
            torch.ones(4), torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertTrue(torch.isfinite(grad).all())

    def test_single_element_input_is_finite(self) -> None:
        value, grad = self._loss_and_grad(torch.tensor([2.0]), torch.tensor([5.0]))
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertTrue(torch.isfinite(grad).all())

    def test_importing_the_module_starts_nothing(self) -> None:
        # The shared module must not drag in W&B or an entry point: a
        # subprocess import is the only way to see the real import graph.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import losses, sys; print('wandb' in sys.modules, 'main' in sys.modules)",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()

"""Regression tests for GraphGenomicNet's per-individual batching.

The model builds one big graph out of N per-individual copies, so a bug
in the node offsets would let one individual's SNPs reach another's
prediction. These tests pin that down on a synthetic graph small enough
to reason about.
"""

import unittest

import torch

import gene_graph
from losses import CorrelationLoss
from model import GraphGenomicNet

NUM_GENES = 4
SNP_COUNT = 8
HIDDEN_DIM = 6


def _fixture_edge_index() -> torch.Tensor:
    frame = gene_graph.edge_frame(
        gene_graph.to_bidirectional_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    )
    return gene_graph.load_edge_index(frame, num_genes=NUM_GENES)


def _snp_to_gene_map() -> torch.Tensor:
    # Two SNPs per gene, in gene order.
    return torch.arange(SNP_COUNT, dtype=torch.long) % NUM_GENES


def _batch_inputs(
    genotypes: torch.Tensor, snp_to_gene_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the flattened batch inputs the training loop passes in."""
    count = genotypes.shape[0]
    flat_genotypes = genotypes.reshape(-1)
    global_map = (
        snp_to_gene_map.repeat(count)
        + torch.arange(count).repeat_interleave(snp_to_gene_map.numel()) * NUM_GENES
    )
    batch_mapping = torch.arange(count).repeat_interleave(NUM_GENES)
    return flat_genotypes, global_map, batch_mapping


def _new_model(seed: int = 0) -> GraphGenomicNet:
    torch.manual_seed(seed)
    model = GraphGenomicNet(
        num_genes=NUM_GENES, hidden_dim=HIDDEN_DIM, num_layers=2, dropout_rate=0.4
    )
    model.eval()
    return model


def _predict(model: GraphGenomicNet, genotypes: torch.Tensor) -> torch.Tensor:
    snp_to_gene_map = _snp_to_gene_map()
    flat, global_map, batch_mapping = _batch_inputs(genotypes, snp_to_gene_map)
    with torch.no_grad():
        return model(flat, global_map, _fixture_edge_index(), batch_mapping)


class GraphGenomicNetBatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.genotypes = torch.randint(-1, 2, (3, SNP_COUNT), dtype=torch.float32)

    def test_output_shape_matches_the_individual_count(self) -> None:
        predictions = _predict(_new_model(), self.genotypes)
        self.assertEqual(tuple(predictions.shape), (3, 1))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_single_and_batched_predictions_agree(self) -> None:
        model = _new_model()
        batched = _predict(model, self.genotypes)
        for index in range(self.genotypes.shape[0]):
            alone = _predict(model, self.genotypes[index : index + 1])
            torch.testing.assert_close(alone[0], batched[index], rtol=1e-5, atol=1e-6)

    def test_other_individuals_do_not_change_a_prediction(self) -> None:
        model = _new_model()
        original = _predict(model, self.genotypes)

        changed = self.genotypes.clone()
        changed[1] = changed[1] + 10.0
        after = _predict(model, changed)

        # The changed individual moves; the others must not.
        torch.testing.assert_close(after[0], original[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(after[2], original[2], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(after[1], original[1]))

    def test_forward_and_backward_are_finite(self) -> None:
        torch.manual_seed(0)
        model = GraphGenomicNet(
            num_genes=NUM_GENES, hidden_dim=HIDDEN_DIM, num_layers=2, dropout_rate=0.0
        )
        model.train()
        snp_to_gene_map = _snp_to_gene_map()
        flat, global_map, batch_mapping = _batch_inputs(self.genotypes, snp_to_gene_map)
        predictions = model(flat, global_map, _fixture_edge_index(), batch_mapping)
        targets = torch.tensor([[1.0], [2.0], [3.0]])
        loss = CorrelationLoss()(predictions, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(predictions).all())
        self.assertTrue(torch.isfinite(loss))
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()

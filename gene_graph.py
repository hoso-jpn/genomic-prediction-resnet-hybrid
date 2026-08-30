"""Gene-graph edge contract shared by the generator and the GNN loader.

The on-disk edge list is a **deduplicated bidirectional** edge set: for
every undirected gene pair {u, v} the file contains ``(u, v)`` and
``(v, u)`` exactly once each. The loader therefore uses the file as-is.
Concatenating reversed edges after loading such a file (as the loader
previously did) stores every direction twice, i.e. four rows per
undirected pair, which changes GCN degree normalization.

Because the two sides must agree, the generator builds its output with
``to_bidirectional_edges`` and the loader validates the same contract
with ``load_edge_index``; a file that does not satisfy it is rejected
rather than silently repaired, so a one-directional edge list cannot be
accepted without anyone noticing.

Node ID rules (both directions of the contract):

- gene IDs are integers in ``[0, num_genes)``; ``num_genes`` is derived
  from the SNP-to-gene map, not from the edge file
- self-loops are rejected (GCNConv adds its own self-loops)
- duplicate rows are rejected
- an empty edge set is rejected
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import torch

EDGE_COLUMNS = ("source", "target")


def to_bidirectional_edges(pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Build the canonical deduplicated bidirectional edge list.

    Accepts undirected pairs in either orientation (and both orientations
    of the same pair) and returns each direction exactly once, sorted.
    Applying it to its own output returns the same list, so a generator
    may run it more than once without changing what it writes.
    """
    undirected: set[tuple[int, int]] = set()
    for pair in pairs:
        source, target = (int(pair[0]), int(pair[1]))
        if source < 0 or target < 0:
            raise ValueError(f"gene IDs must be non-negative: {(source, target)}")
        if source == target:
            raise ValueError(f"self-loops are not part of the edge contract: {source}")
        undirected.add((min(source, target), max(source, target)))
    if not undirected:
        raise ValueError("at least one undirected edge is required")
    edges: list[tuple[int, int]] = []
    for source, target in sorted(undirected):
        edges.append((source, target))
        edges.append((target, source))
    return edges


def edge_frame(edges: Iterable[tuple[int, int]]) -> pd.DataFrame:
    """Wrap an edge list in the on-disk frame layout."""
    return pd.DataFrame(list(edges), columns=list(EDGE_COLUMNS))


def _validate_edge_frame(frame: pd.DataFrame, num_genes: int) -> list[tuple[int, int]]:
    if list(frame.columns) != list(EDGE_COLUMNS):
        raise ValueError(
            f"edge file must have columns {list(EDGE_COLUMNS)}, "
            f"found {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("edge file contains no edges")
    for column in EDGE_COLUMNS:
        values = frame[column]
        if not pd.api.types.is_integer_dtype(values):
            raise ValueError(f"edge column '{column}' must contain integer gene IDs")

    edges = [
        (int(source), int(target))
        for source, target in zip(frame[EDGE_COLUMNS[0]], frame[EDGE_COLUMNS[1]])
    ]

    out_of_range = sorted(
        {node for edge in edges for node in edge if not 0 <= node < num_genes}
    )
    if out_of_range:
        raise ValueError(
            f"gene IDs must be in [0, {num_genes}); out-of-range IDs: {out_of_range}"
        )

    self_loops = sorted({source for source, target in edges if source == target})
    if self_loops:
        raise ValueError(f"self-loops are not allowed; gene IDs: {self_loops}")

    duplicates = sorted({edge for edge in edges if edges.count(edge) > 1})
    if duplicates:
        raise ValueError(
            "each direction must appear exactly once; duplicated edges: "
            f"{duplicates[:10]}"
        )

    directed = set(edges)
    missing_reverse = sorted(
        {edge for edge in edges if (edge[1], edge[0]) not in directed}
    )
    if missing_reverse:
        raise ValueError(
            "the edge list must be bidirectional; edges without their reverse: "
            f"{missing_reverse[:10]}"
        )
    return edges


def load_edge_index(frame: pd.DataFrame, num_genes: int) -> torch.Tensor:
    """Validate an on-disk edge frame and return its ``(2, E)`` edge index.

    No reversed copy is added here: the file already carries both
    directions, and adding them again would double every edge.
    """
    if num_genes <= 0:
        raise ValueError("num_genes must be positive")
    edges = _validate_edge_frame(frame, num_genes)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def validate_snp_to_gene_map(values: torch.Tensor, *, snp_count: int) -> int:
    """Validate a SNP-to-gene mapping and return the derived gene count.

    Gene IDs are contiguous from 0, so the number of genes is
    ``max(gene_id) + 1``; unused trailing gene IDs cannot be recovered
    from the mapping alone, which is why the edge file is validated
    against this value rather than the other way round.
    """
    if values.ndim != 1:
        raise ValueError("snp_to_gene_map must be one-dimensional")
    if values.numel() != snp_count:
        raise ValueError(
            f"snp_to_gene_map length ({values.numel()}) does not match the "
            f"SNP count ({snp_count})"
        )
    if values.numel() == 0:
        raise ValueError("snp_to_gene_map must not be empty")
    if values.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError(
            f"snp_to_gene_map must contain integer gene IDs, got {values.dtype}"
        )
    minimum = int(values.min().item())
    if minimum < 0:
        raise ValueError(f"gene IDs must be non-negative; found {minimum}")
    return int(values.max().item()) + 1

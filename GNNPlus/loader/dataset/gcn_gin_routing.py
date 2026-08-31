"""Synthetic GCN vs GIN operator-routing star graphs.

Each graph is a rooted star: central node ``r``, ``k`` signal neighbors, and
zero-feature dummy leaves attached to neighbors to vary ``d_u``.

Labels follow graph type ``tau``:
  tau=0 (GCN-type): y = 1[ sum_u x_u / sqrt((d_r+1)(d_u+1)) > 0 ]
  tau=1 (GIN-type): y = 1[ sum_u x_u > 0 ]

See ``Paper_gcn_gin_routing_synthetic.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

GraphType = Literal[0, 1]
Difficulty = Literal["easy", "medium", "hard", "opposite_sign"]

FEAT_SIGNAL: Final[int] = 0
FEAT_TYPE: Final[int] = 1
NUM_NODE_FEATURES: Final[int] = 2


@dataclass(frozen=True)
class NeighborSpec:
    """One signal neighbor of the root."""

    feature: int
    num_dummy_leaves: int

    @property
    def degree(self) -> int:
        """Graph degree of the neighbor (root edge + dummy leaves)."""
        return 1 + self.num_dummy_leaves


@dataclass(frozen=True)
class RoutingGraphSpec:
    """Fully specified star graph for the routing benchmark."""

    tau: GraphType
    neighbors: tuple[NeighborSpec, ...]
    pair_id: Optional[int] = None
    difficulty: Difficulty = "medium"

    @property
    def num_neighbors(self) -> int:
        """Number of signal neighbors attached to the root."""
        return len(self.neighbors)

    @property
    def root_degree(self) -> int:
        """Degree of the root node."""
        return self.num_neighbors

    def neighbor_features(self) -> list[int]:
        """Scalar features on signal neighbors."""
        return [n.feature for n in self.neighbors]

    def neighbor_degrees(self) -> list[int]:
        """Degrees of signal neighbors."""
        return [n.degree for n in self.neighbors]

    def gcn_score(self) -> float:
        """Degree-normalized aggregation score (GCN rule)."""
        return gcn_aggregate_score(
            self.root_degree,
            self.neighbor_features(),
            self.neighbor_degrees(),
        )

    def gin_score(self) -> float:
        """Unnormalized sum score (GIN rule)."""
        return float(sum(self.neighbor_features()))

    def label(self) -> int:
        """Binary label for this graph type."""
        score = self.gcn_score() if self.tau == 0 else self.gin_score()
        return int(score > 0.0)

    def scores_disagree(self) -> bool:
        """True when GCN and GIN rules would assign opposite classes."""
        return (self.gcn_score() > 0.0) != (self.gin_score() > 0.0)


def gcn_aggregate_score(
    root_degree: int,
    neighbor_features: Sequence[int],
    neighbor_degrees: Sequence[int],
) -> float:
    """Compute sum_u x_u / sqrt((d_r+1)(d_u+1))."""
    total = 0.0
    denom_root = math.sqrt(float(root_degree + 1))
    for x_u, d_u in zip(neighbor_features, neighbor_degrees, strict=True):
        total += float(x_u) / (denom_root * math.sqrt(float(d_u + 1)))
    return total


def gin_aggregate_score(neighbor_features: Sequence[int]) -> float:
    """Compute sum_u x_u."""
    return float(sum(neighbor_features))


def build_star_graph(spec: RoutingGraphSpec) -> Data:
    """Materialize a ``RoutingGraphSpec`` as a PyG ``Data`` object."""
    if not spec.neighbors:
        raise ValueError("star graph requires at least one neighbor")

    node_roles: list[int] = []
    signals: list[float] = []
    type_channel: list[float] = []

    # Root node (index 0).
    node_roles.append(0)
    signals.append(0.0)
    type_channel.append(float(spec.tau))

    edge_src: list[int] = []
    edge_dst: list[int] = []

    next_idx = 1
    for neigh in spec.neighbors:
        u_idx = next_idx
        next_idx += 1
        node_roles.append(1)
        signals.append(float(neigh.feature))
        type_channel.append(0.0)
        edge_src.extend([0, u_idx])
        edge_dst.extend([u_idx, 0])

        for _ in range(neigh.num_dummy_leaves):
            leaf_idx = next_idx
            next_idx += 1
            node_roles.append(2)
            signals.append(0.0)
            type_channel.append(0.0)
            edge_src.extend([u_idx, leaf_idx])
            edge_dst.extend([leaf_idx, u_idx])

    x = torch.tensor(
        [signals, type_channel],
        dtype=torch.float32,
    ).t().contiguous()
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor([spec.label()], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=next_idx,
    )
    data.tau = torch.tensor([spec.tau], dtype=torch.long)
    data.gcn_score = torch.tensor([spec.gcn_score()], dtype=torch.float32)
    data.gin_score = torch.tensor([spec.gin_score()], dtype=torch.float32)
    data.opposite_sign = torch.tensor([spec.scores_disagree()], dtype=torch.bool)
    data.root_index = torch.tensor([0], dtype=torch.long)
    data.node_role = torch.tensor(node_roles, dtype=torch.long)
    data.difficulty = spec.difficulty
    pair_id_value = -1 if spec.pair_id is None else spec.pair_id
    data.pair_id = torch.tensor([pair_id_value], dtype=torch.long)
    return data


def spec_to_metadata(spec: RoutingGraphSpec) -> dict[str, Any]:
    """JSON-serializable summary for plots and logs."""
    return {
        **asdict(spec),
        "gcn_score": spec.gcn_score(),
        "gin_score": spec.gin_score(),
        "label": spec.label(),
        "opposite_sign": spec.scores_disagree(),
    }


def iter_random_specs(
    n_graphs: int,
    *,
    rng: np.random.Generator,
    min_neighbors: int = 4,
    max_neighbors: int = 8,
    max_dummy_leaves: int = 24,
    opposite_sign_fraction: float = 0.2,
) -> Iterator[RoutingGraphSpec]:
    """Yield random star-graph specs with optional opposite-sign pairs."""
    if not 0.0 <= opposite_sign_fraction <= 1.0:
        raise ValueError("opposite_sign_fraction must be in [0, 1]")

    n_opposite_target = int(round(n_graphs * opposite_sign_fraction))
    n_opposite = 0
    pair_id = 0

    produced = 0
    attempts = 0
    max_attempts = n_graphs * 200

    while produced < n_graphs and attempts < max_attempts:
        attempts += 1
        k = int(rng.integers(min_neighbors, max_neighbors + 1))
        features = rng.choice([-1, 1], size=k).astype(int).tolist()
        dummy_counts = rng.integers(0, max_dummy_leaves + 1, size=k).astype(int).tolist()
        neighbors = tuple(
            NeighborSpec(feature=int(f), num_dummy_leaves=int(d))
            for f, d in zip(features, dummy_counts, strict=True)
        )
        base = RoutingGraphSpec(tau=0, neighbors=neighbors, difficulty="medium")
        disagree = base.scores_disagree()

        want_pair = (
            n_opposite < n_opposite_target
            and disagree
            and produced + 1 < n_graphs
        )
        if want_pair:
            gcn_spec = RoutingGraphSpec(
                tau=0,
                neighbors=neighbors,
                pair_id=pair_id,
                difficulty="opposite_sign",
            )
            gin_spec = RoutingGraphSpec(
                tau=1,
                neighbors=neighbors,
                pair_id=pair_id,
                difficulty="opposite_sign",
            )
            yield gcn_spec
            yield gin_spec
            produced += 2
            n_opposite += 1
            pair_id += 1
            continue

        tau = int(rng.integers(0, 2))
        difficulty: Difficulty = "medium"
        if not disagree:
            difficulty = "easy"
        elif abs(base.gcn_score()) < 0.15 or abs(base.gin_score()) < 0.5:
            difficulty = "hard"

        yield RoutingGraphSpec(
            tau=tau,  # type: ignore[arg-type]
            neighbors=neighbors,
            difficulty=difficulty,
        )
        produced += 1


def curated_example_specs() -> list[tuple[RoutingGraphSpec, str]]:
    """Hand-picked examples for documentation figures (spec, caption)."""
    examples: list[tuple[RoutingGraphSpec, str]] = []

    ex1 = RoutingGraphSpec(
        tau=0,
        neighbors=(
            NeighborSpec(+1, 0),
            NeighborSpec(+1, 0),
            NeighborSpec(+1, 0),
            NeighborSpec(+1, 0),
        ),
        difficulty="easy",
    )
    examples.append(
        (
            ex1,
            "Easy (aligned rules). Four neighbors with x=+1 and equal degree "
            f"(d_u=1). GCN score={ex1.gcn_score():.2f}, GIN score={ex1.gin_score():.0f} "
            "→ both rules predict y=1. A single-head GCN or GIN model succeeds "
            "without routing; gates should still prefer the GCN head on τ=0 graphs.",
        ),
    )

    ex2 = RoutingGraphSpec(
        tau=1,
        neighbors=(
            NeighborSpec(-1, 0),
            NeighborSpec(-1, 0),
            NeighborSpec(-1, 0),
            NeighborSpec(+1, 0),
            NeighborSpec(+1, 0),
        ),
        difficulty="easy",
    )
    examples.append(
        (
            ex2,
            "Easy (GIN-type, unanimous vote). Five neighbors: three −1 and two +1 "
            f"→ GIN sum={ex2.gin_score():.0f} (y=0). GCN score={ex2.gcn_score():.2f} "
            "agrees because degrees are equal. Routing is optional for accuracy but "
            "the GIN head should receive higher gate mass at the root.",
        ),
    )

    ex3 = RoutingGraphSpec(
        tau=0,
        neighbors=(
            NeighborSpec(+1, 0),
            NeighborSpec(+1, 19),
            NeighborSpec(-1, 0),
            NeighborSpec(-1, 0),
        ),
        difficulty="medium",
    )
    examples.append(
        (
            ex3,
            "Medium (degree imbalance, GCN-type). The +1 neighbor with 19 dummy "
            "leaves has large degree (d_u=20) and is down-weighted in the GCN norm; "
            f"two low-degree −1 neighbors matter more in GIN (sum={ex3.gin_score():.0f}) "
            f"than in GCN (score={ex3.gcn_score():.2f}). Only the normalized rule "
            f"gives y={ex3.label()}. A GIN-only baseline misclassifies.",
        ),
    )

    shared_neighbors = (
        NeighborSpec(+1, 0),
        NeighborSpec(+1, 0),
        NeighborSpec(-1, 19),
        NeighborSpec(-1, 19),
    )
    ex4a = RoutingGraphSpec(
        tau=0,
        neighbors=shared_neighbors,
        pair_id=0,
        difficulty="opposite_sign",
    )
    ex4b = RoutingGraphSpec(
        tau=1,
        neighbors=shared_neighbors,
        pair_id=0,
        difficulty="opposite_sign",
    )
    examples.append(
        (
            ex4a,
            "Hard (opposite-sign pair, GCN-type). Identical features and degrees as "
            "the next panel. GCN score="
            f"{ex4a.gcn_score():.2f} → y=1, but GIN sum={ex4a.gin_score():.0f} → y=0. "
            "Local structure alone cannot determine the label; the root type bit τ=0 "
            "selects the GCN rule. SiGMA must route to the GCN head.",
        ),
    )
    examples.append(
        (
            ex4b,
            "Hard (opposite-sign pair, GIN-type). Same graph as left, but τ=1. "
            f"GIN sum={ex4b.gin_score():.0f} → y=0 while GCN score={ex4b.gcn_score():.2f} "
            "→ y=1. This is the critical reviewer test: one model, two graphs, "
            "different correct operators.",
        ),
    )

    ex5 = RoutingGraphSpec(
        tau=0,
        neighbors=(
            NeighborSpec(+1, 2),
            NeighborSpec(+1, 0),
            NeighborSpec(-1, 1),
            NeighborSpec(-1, 0),
        ),
        difficulty="hard",
    )
    examples.append(
        (
            ex5,
            "Hard (near threshold, GCN-type). Mixed ±1 neighbors with heterogeneous "
            f"degrees; GCN score={ex5.gcn_score():.3f} (|·| small) while GIN "
            f"sum={ex5.gin_score():.0f}. A tiny change in dummy-leaf counts flips "
            "the GCN label; the model must implement normalization accurately.",
        ),
    )

    ex6 = RoutingGraphSpec(
        tau=1,
        neighbors=tuple(
            NeighborSpec(
                int(f),
                int(d),
            )
            for f, d in zip(
                [+1, +1, -1, +1, -1, -1, +1, -1],
                [0, 5, 12, 1, 8, 0, 15, 2],
                strict=True,
            )
        ),
        difficulty="medium",
    )
    examples.append(
        (
            ex6,
            "Medium (many neighbors, k=8). Larger stars increase the action space "
            f"for gating and make degree heterogeneity more pronounced (d_u ∈ "
            f"[1,16]). GIN sum={ex6.gin_score():.0f} vs GCN score={ex6.gcn_score():.2f}; "
            "rules agree here, but the pattern is representative of training graphs.",
        ),
    )

    return examples


class GcnGinRoutingDataset(InMemoryDataset):
    """In-memory dataset of synthetic routing stars."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        *,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
        regenerate: bool = False,
    ) -> None:
        self.split = split
        self.regenerate = regenerate
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[0],
            weights_only=False,
        )

    @property
    def raw_file_names(self) -> list[str]:
        """Placeholder raw file (generation is programmatic)."""
        return ["spec.json"]

    @property
    def processed_file_names(self) -> list[str]:
        """One processed tensor file per split."""
        return [f"{self.split}.pt"]

    def download(self) -> None:
        """Write split sizes to raw spec (no external download)."""
        raw_path = Path(self.raw_dir)
        raw_path.mkdir(parents=True, exist_ok=True)
        spec = {
            "train": 10_000,
            "val": 2_000,
            "test": 2_000,
            "seed": 42,
            "opposite_sign_fraction": 0.2,
        }
        with (raw_path / "spec.json").open("w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)

    def process(self) -> None:
        """Generate and save graphs for this split."""
        raw_path = Path(self.raw_dir) / "spec.json"
        with raw_path.open(encoding="utf-8") as fh:
            spec = json.load(fh)

        sizes = {
            "train": int(spec["train"]),
            "val": int(spec["val"]),
            "test": int(spec["test"]),
        }
        seed = int(spec.get("seed", 42))
        opposite_frac = float(spec.get("opposite_sign_fraction", 0.2))
        n = sizes[self.split]
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[self.split]
        rng = np.random.default_rng(split_seed)

        graphs = [
            build_star_graph(s)
            for s in iter_random_specs(
                n,
                rng=rng,
                opposite_sign_fraction=opposite_frac,
            )
        ]
        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]
        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

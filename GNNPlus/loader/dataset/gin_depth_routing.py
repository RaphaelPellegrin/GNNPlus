"""Synthetic GIN depth-routing trees (1-layer vs 2-layer GIN).

Each graph is a rooted depth-2 tree: root ``r``, hop-1 mid nodes with signals
``x_u ∈ {-1,+1}``, and hop-2 leaves with signals ``z_v ∈ {-1,+1}``.

Under neighbor-sum GIN (no self-loops, identity MLP):

* ``S1 = h_r^(1) = sum_u x_u``  (one GIN update at the root)
* ``S2 = h_r^(2) = sum_v z_v``  (GIN ∘ GIN at the root)

Labels follow graph type ``tau``:
  tau=0 (shallow / 1-GIN): y = 1[ S1 > 0 ]
  tau=1 (deep / 2-GIN):    y = 1[ S2 > 0 ]
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

GraphType = Literal[0, 1]
Difficulty = Literal["easy", "medium", "hard", "opposite_sign"]

FEAT_SIGNAL: Final[int] = 0
FEAT_TYPE: Final[int] = 1
NUM_NODE_FEATURES: Final[int] = 2

ROLE_ROOT: Final[int] = 0
ROLE_MID: Final[int] = 1
ROLE_LEAF: Final[int] = 2


@dataclass(frozen=True)
class MidBranchSpec:
    """One hop-1 mid node and its hop-2 leaf signals."""

    feature: int
    leaf_features: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate mid / leaf signal values."""
        if self.feature not in (-1, 1):
            raise ValueError(f"mid feature must be ±1, got {self.feature}")
        for z in self.leaf_features:
            if z not in (-1, 1):
                raise ValueError(f"leaf feature must be ±1, got {z}")


@dataclass(frozen=True)
class DepthRoutingGraphSpec:
    """Fully specified depth-2 tree for the GIN depth-routing benchmark."""

    tau: GraphType
    branches: tuple[MidBranchSpec, ...]
    pair_id: Optional[int] = None
    difficulty: Difficulty = "medium"

    @property
    def num_mids(self) -> int:
        """Number of hop-1 mid nodes."""
        return len(self.branches)

    @property
    def num_leaves(self) -> int:
        """Number of hop-2 leaf nodes."""
        return sum(len(b.leaf_features) for b in self.branches)

    def mid_features(self) -> list[int]:
        """Scalar signals on hop-1 mids."""
        return [b.feature for b in self.branches]

    def leaf_features(self) -> list[int]:
        """Scalar signals on hop-2 leaves (flattened)."""
        out: list[int] = []
        for branch in self.branches:
            out.extend(branch.leaf_features)
        return out

    def s1_score(self) -> float:
        """One-layer GIN score at the root: sum of hop-1 features."""
        return gin1_score(self.mid_features())

    def s2_score(self) -> float:
        """Two-layer GIN score at the root: sum of hop-2 features."""
        return gin2_score(self.leaf_features())

    def label(self) -> int:
        """Binary label for this depth type."""
        score = self.s1_score() if self.tau == 0 else self.s2_score()
        return int(score > 0.0)

    def scores_disagree(self) -> bool:
        """True when 1-GIN and 2-GIN rules assign opposite classes."""
        return (self.s1_score() > 0.0) != (self.s2_score() > 0.0)


def gin1_score(mid_features: Sequence[int]) -> float:
    """Compute S1 = sum_u x_u (1× GIN at root)."""
    return float(sum(mid_features))


def gin2_score(leaf_features: Sequence[int]) -> float:
    """Compute S2 = sum_v z_v (GIN∘GIN at root under no-self SumConv)."""
    return float(sum(leaf_features))


def build_depth2_tree(spec: DepthRoutingGraphSpec) -> Data:
    """Materialize a ``DepthRoutingGraphSpec`` as a PyG ``Data`` object."""
    if not spec.branches:
        raise ValueError("depth-2 tree requires at least one mid branch")

    node_roles: list[int] = []
    signals: list[float] = []
    type_channel: list[float] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []

    # Root node (index 0).
    node_roles.append(ROLE_ROOT)
    signals.append(0.0)
    type_channel.append(float(spec.tau))

    next_idx = 1
    for branch in spec.branches:
        mid_idx = next_idx
        next_idx += 1
        node_roles.append(ROLE_MID)
        signals.append(float(branch.feature))
        type_channel.append(0.0)
        edge_src.extend([0, mid_idx])
        edge_dst.extend([mid_idx, 0])

        for z in branch.leaf_features:
            leaf_idx = next_idx
            next_idx += 1
            node_roles.append(ROLE_LEAF)
            signals.append(float(z))
            type_channel.append(0.0)
            edge_src.extend([mid_idx, leaf_idx])
            edge_dst.extend([leaf_idx, mid_idx])

    x = torch.tensor([signals, type_channel], dtype=torch.float32).t().contiguous()
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor([spec.label()], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=next_idx,
    )
    data.tau = torch.tensor([spec.tau], dtype=torch.long)
    data.s1_score = torch.tensor([spec.s1_score()], dtype=torch.float32)
    data.s2_score = torch.tensor([spec.s2_score()], dtype=torch.float32)
    data.opposite_sign = torch.tensor([spec.scores_disagree()], dtype=torch.bool)
    data.root_index = torch.tensor([0], dtype=torch.long)
    data.node_role = torch.tensor(node_roles, dtype=torch.long)
    data.difficulty = spec.difficulty
    pair_id_value = -1 if spec.pair_id is None else spec.pair_id
    data.pair_id = torch.tensor([pair_id_value], dtype=torch.long)
    return data


def spec_to_metadata(spec: DepthRoutingGraphSpec) -> dict[str, Any]:
    """JSON-serializable summary for plots and logs."""
    return {
        **asdict(spec),
        "s1_score": spec.s1_score(),
        "s2_score": spec.s2_score(),
        "label": spec.label(),
        "opposite_sign": spec.scores_disagree(),
        "num_mids": spec.num_mids,
        "num_leaves": spec.num_leaves,
    }


def _sample_branches(
    rng: np.random.Generator,
    *,
    min_mids: int,
    max_mids: int,
    max_leaves_per_mid: int,
    min_total_leaves: int,
) -> tuple[MidBranchSpec, ...]:
    """Sample a random depth-2 branch structure with at least ``min_total_leaves``."""
    for _ in range(200):
        k = int(rng.integers(min_mids, max_mids + 1))
        branches: list[MidBranchSpec] = []
        for _mid in range(k):
            feat = int(rng.choice([-1, 1]))
            n_leaves = int(rng.integers(0, max_leaves_per_mid + 1))
            leaves = tuple(int(x) for x in rng.choice([-1, 1], size=n_leaves))
            branches.append(MidBranchSpec(feature=feat, leaf_features=leaves))
        if sum(len(b.leaf_features) for b in branches) >= min_total_leaves:
            return tuple(branches)
    raise RuntimeError("failed to sample depth-2 branches with enough leaves")


def iter_random_specs(
    n_graphs: int,
    *,
    rng: np.random.Generator,
    min_mids: int = 3,
    max_mids: int = 6,
    max_leaves_per_mid: int = 4,
    min_total_leaves: int = 2,
    opposite_sign_fraction: float = 0.25,
) -> Iterator[DepthRoutingGraphSpec]:
    """Yield random depth-2 specs with optional opposite-sign pairs."""
    if not 0.0 <= opposite_sign_fraction <= 1.0:
        raise ValueError("opposite_sign_fraction must be in [0, 1]")

    n_opposite_target = int(round(n_graphs * opposite_sign_fraction))
    n_opposite = 0
    pair_id = 0
    produced = 0
    attempts = 0
    max_attempts = n_graphs * 400

    while produced < n_graphs and attempts < max_attempts:
        attempts += 1
        branches = _sample_branches(
            rng,
            min_mids=min_mids,
            max_mids=max_mids,
            max_leaves_per_mid=max_leaves_per_mid,
            min_total_leaves=min_total_leaves,
        )
        base = DepthRoutingGraphSpec(tau=0, branches=branches, difficulty="medium")
        disagree = base.scores_disagree()

        want_pair = (
            n_opposite < n_opposite_target
            and disagree
            and produced + 1 < n_graphs
        )
        if want_pair:
            yield DepthRoutingGraphSpec(
                tau=0,
                branches=branches,
                pair_id=pair_id,
                difficulty="opposite_sign",
            )
            yield DepthRoutingGraphSpec(
                tau=1,
                branches=branches,
                pair_id=pair_id,
                difficulty="opposite_sign",
            )
            produced += 2
            n_opposite += 1
            pair_id += 1
            continue

        tau = int(rng.integers(0, 2))
        difficulty: Difficulty = "medium"
        if not disagree:
            difficulty = "easy"
        elif abs(base.s1_score()) < 1.5 or abs(base.s2_score()) < 1.5:
            difficulty = "hard"

        yield DepthRoutingGraphSpec(
            tau=tau,  # type: ignore[arg-type]
            branches=branches,
            difficulty=difficulty,
        )
        produced += 1


def curated_example_specs() -> list[tuple[DepthRoutingGraphSpec, str]]:
    """Hand-picked examples for documentation figures (spec, caption)."""
    examples: list[tuple[DepthRoutingGraphSpec, str]] = []

    # 1. Easy aligned (both depths say y=1), τ=0
    ex1 = DepthRoutingGraphSpec(
        tau=0,
        branches=(
            MidBranchSpec(+1, (+1,)),
            MidBranchSpec(+1, (+1,)),
            MidBranchSpec(+1, (+1,)),
        ),
        difficulty="easy",
    )
    examples.append(
        (
            ex1,
            "Easy (aligned depths, shallow). Three mids with x=+1 and one +1 leaf "
            f"each. S1={ex1.s1_score():.0f}, S2={ex1.s2_score():.0f} → both 1-GIN and "
            "2-GIN predict y=1. τ=0 selects the shallow rule; a deep-only model still "
            "succeeds here.",
        ),
    )

    # 2. Easy aligned (both say y=0), τ=1
    ex2 = DepthRoutingGraphSpec(
        tau=1,
        branches=(
            MidBranchSpec(-1, (-1,)),
            MidBranchSpec(-1, (-1,)),
            MidBranchSpec(+1, (-1, -1)),
        ),
        difficulty="easy",
    )
    examples.append(
        (
            ex2,
            "Easy (aligned depths, deep). S1="
            f"{ex2.s1_score():.0f} and S2={ex2.s2_score():.0f} both negative → y=0 "
            "under either rule. τ=1 asks for 2-GIN; gates should prefer the deep head "
            "even though a shallow specialist would also be correct.",
        ),
    )

    # 3–4. Opposite-sign pair from the design discussion
    shared_opp = (
        MidBranchSpec(+1, (-1, -1)),
        MidBranchSpec(+1, (-1,)),
        MidBranchSpec(-1, ()),
    )
    ex3a = DepthRoutingGraphSpec(
        tau=0,
        branches=shared_opp,
        pair_id=0,
        difficulty="opposite_sign",
    )
    ex3b = DepthRoutingGraphSpec(
        tau=1,
        branches=shared_opp,
        pair_id=0,
        difficulty="opposite_sign",
    )
    examples.append(
        (
            ex3a,
            "Hard (opposite-sign pair, 1-GIN / τ=0). Identical tree as the next panel. "
            f"S1={ex3a.s1_score():.0f} → y=1, but S2={ex3a.s2_score():.0f} → y=0. "
            "A forced 2-layer GIN misclassifies; SiGMA must open the shallow head.",
        ),
    )
    examples.append(
        (
            ex3b,
            "Hard (opposite-sign pair, 2-GIN / τ=1). Same features/topology, τ=1. "
            f"S2={ex3b.s2_score():.0f} → y=0 while S1={ex3b.s1_score():.0f} → y=1. "
            "Critical depth test: one model, two graphs, different correct depths.",
        ),
    )

    # 5–6. Mirrored opposite-sign (S1<0, S2>0)
    shared_mir = (
        MidBranchSpec(-1, (+1, +1)),
        MidBranchSpec(-1, (+1,)),
        MidBranchSpec(+1, ()),
    )
    ex4a = DepthRoutingGraphSpec(
        tau=0,
        branches=shared_mir,
        pair_id=1,
        difficulty="opposite_sign",
    )
    ex4b = DepthRoutingGraphSpec(
        tau=1,
        branches=shared_mir,
        pair_id=1,
        difficulty="opposite_sign",
    )
    examples.append(
        (
            ex4a,
            "Hard (mirrored opposite-sign, τ=0). "
            f"S1={ex4a.s1_score():.0f} → y=0, S2={ex4a.s2_score():.0f} → y=1. "
            "Shallow rule wins; deep-only baseline fails.",
        ),
    )
    examples.append(
        (
            ex4b,
            "Hard (mirrored opposite-sign, τ=1). Same graph, deep rule: "
            f"S2={ex4b.s2_score():.0f} → y=1. Leaf majority is positive while mid "
            "majority is negative — depth choice flips the label.",
        ),
    )

    # 7. Near-threshold / sparse leaves
    ex5 = DepthRoutingGraphSpec(
        tau=1,
        branches=(
            MidBranchSpec(+1, (+1,)),
            MidBranchSpec(+1, (-1,)),
            MidBranchSpec(-1, (-1,)),
            MidBranchSpec(-1, ()),
        ),
        difficulty="hard",
    )
    examples.append(
        (
            ex5,
            "Hard (near threshold, deep). "
            f"S1={ex5.s1_score():.0f}, S2={ex5.s2_score():.0f}. One leaf flip changes "
            "the 2-GIN class; mids alone (S1) already disagree with leaves.",
        ),
    )

    # 8. Larger tree
    ex6 = DepthRoutingGraphSpec(
        tau=0,
        branches=(
            MidBranchSpec(+1, (+1, -1)),
            MidBranchSpec(+1, (+1, +1, -1)),
            MidBranchSpec(-1, (-1, +1)),
            MidBranchSpec(+1, (-1,)),
            MidBranchSpec(-1, (+1, +1)),
        ),
        difficulty="medium",
    )
    examples.append(
        (
            ex6,
            "Medium (larger tree, k=5 mids). "
            f"S1={ex6.s1_score():.0f}, S2={ex6.s2_score():.0f}, opposite_sign="
            f"{ex6.scores_disagree()}. Representative of training graphs with mixed "
            "branching.",
        ),
    )

    return examples


class GinDepthRoutingDataset(InMemoryDataset):
    """In-memory dataset of synthetic GIN depth-routing trees."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        *,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
    ) -> None:
        self.split = split
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
        """Write default split sizes to raw spec."""
        raw_path = Path(self.raw_dir)
        raw_path.mkdir(parents=True, exist_ok=True)
        spec = {
            "train": 10_000,
            "val": 2_000,
            "test": 2_000,
            "seed": 42,
            "opposite_sign_fraction": 0.25,
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
        opposite_frac = float(spec.get("opposite_sign_fraction", 0.25))
        n = sizes[self.split]
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[self.split]
        rng = np.random.default_rng(split_seed)

        graphs = [
            build_depth2_tree(s)
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

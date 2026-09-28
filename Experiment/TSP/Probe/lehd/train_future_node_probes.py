import os
import re
import sys
import csv
import glob
import time
import pickle
import random
import argparse
import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Distribution metadata
# ============================================================

DISTRIBUTION_ORDER = [
    "Clustered",
    "Expansion",
    "Explosion",
    "Grid",
    "Implosion",
    "Mixed",
    "Uniform",
]

DISTRIBUTION_CODE = {name: idx for idx, name in enumerate(DISTRIBUTION_ORDER)}


# ============================================================
# NumPy pickle compatibility
# ============================================================

def install_numpy_pickle_compatibility_patch() -> None:
    """Allow loading pickle files saved with different NumPy versions."""
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core._internal": "numpy.core._internal",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core.arrayprint": "numpy.core.arrayprint",
        "numpy._core.records": "numpy.core.records",
    }

    for new_name, old_name in aliases.items():
        if new_name in sys.modules:
            continue
        try:
            sys.modules[new_name] = importlib.import_module(old_name)
        except Exception:
            pass


install_numpy_pickle_compatibility_patch()


# ============================================================
# Data containers
# ============================================================

@dataclass
class InstanceRecord:
    """One TSP instance and its metadata."""
    coords: np.ndarray
    n: int
    distribution: str
    distribution_code: int
    instance_id: int
    split: str
    source_file: str


@dataclass
class MetricState:
    """Streaming accumulator for ranking metrics."""
    count: int = 0
    top1_sum: float = 0.0
    top5_sum: float = 0.0
    top10_sum: float = 0.0
    mrr_sum: float = 0.0
    rank_sum: float = 0.0
    rank_percentile_sum: float = 0.0

    def update_ranks(self, ranks: torch.Tensor, num_candidates: int) -> None:
        """Update metrics from a tensor of 1-based ranks."""
        if ranks.numel() == 0:
            return

        r = ranks.detach().float().cpu().numpy()
        a = max(int(num_candidates), 1)
        c = int(r.shape[0])

        self.count += c
        self.top1_sum += float(np.sum(r <= 1))
        self.top5_sum += float(np.sum(r <= min(5, a)))
        self.top10_sum += float(np.sum(r <= min(10, a)))
        self.mrr_sum += float(np.sum(1.0 / r))
        self.rank_sum += float(np.sum(r))

        if a <= 1:
            self.rank_percentile_sum += 0.0
        else:
            self.rank_percentile_sum += float(np.sum((r - 1.0) / float(a - 1)))

    def update_random_expected_many(self, num_candidates: int, count: int) -> None:
        """Update using expected metrics of a uniformly random ranking."""
        if count <= 0:
            return

        a = max(int(num_candidates), 1)
        c = int(count)

        self.count += c
        self.top1_sum += c * (1.0 / a)
        self.top5_sum += c * (min(5, a) / a)
        self.top10_sum += c * (min(10, a) / a)

        harmonic = sum(1.0 / r for r in range(1, a + 1))
        self.mrr_sum += c * (harmonic / a)
        self.rank_sum += c * ((a + 1) / 2.0)
        self.rank_percentile_sum += c * (0.0 if a <= 1 else 0.5)

    def to_row(self) -> Dict:
        """Convert accumulated values to averaged metrics."""
        if self.count == 0:
            return {
                "num_samples": 0,
                "top1": float("nan"),
                "top5": float("nan"),
                "top10": float("nan"),
                "mrr": float("nan"),
                "mean_rank": float("nan"),
                "mean_rank_percentile": float("nan"),
            }

        c = float(self.count)
        return {
            "num_samples": self.count,
            "top1": self.top1_sum / c,
            "top5": self.top5_sum / c,
            "top10": self.top10_sum / c,
            "mrr": self.mrr_sum / c,
            "mean_rank": self.rank_sum / c,
            "mean_rank_percentile": self.rank_percentile_sum / c,
        }


# ============================================================
# CLI arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train batched future-node planning probes for LEHD, horizons 5..100.")

    parser.add_argument(
        "--inference_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing LEHD plot-friendly PKL files.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory for checkpoints and CSV files.",
    )

    parser.add_argument(
        "--run_tag",
        type=str,
        default="horizons_5_100",
        help=(
            "Tag added to horizon-5..100 CSV files and combined checkpoints. "
            "This prevents overwriting the existing horizon-1..4 or horizon-5..15 files in the selected output directory."
        ),
    )

    parser.add_argument(
        "--append_to_global_csv",
        action="store_true",
        help=(
            "If set, also append rows to all_validation_metrics.csv and all_test_metrics.csv. "
            "By default, this script writes separate tagged global CSV files only."
        ),
    )

    parser.add_argument(
        "--lehd_root",
        type=str,
        required=True,
        help="Root directory containing the LEHD package.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the pretrained LEHD checkpoint.",
    )

    parser.add_argument("--nodes", type=str, default="20,50,100,200,500,1000")
    parser.add_argument("--layers", type=str, default="-1,0,1,2,3,4,5")
    parser.add_argument("--horizons", type=str, default="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100")

    parser.add_argument(
        "--max_instances_per_distribution_by_n",
        type=str,
        default="20:2000,50:1000,100:1000,200:500,500:200,1000:100",
        help="Example: 20:2000,50:1000. Use -1 for all instances.",
    )

    parser.add_argument(
        "--batch_size_by_n",
        type=str,
        default="20:128,50:128,100:64,200:32,500:16,1000:4",
        help="Batch size per N. Lower this if CUDA OOM occurs.",
    )

    parser.add_argument("--fallback_batch_size", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--cuda_device_num", type=int, default=0)

    parser.add_argument("--train_step_stride", type=int, default=1)
    parser.add_argument("--eval_step_stride", type=int, default=1)
    parser.add_argument("--max_train_steps_per_instance", type=int, default=-1)
    parser.add_argument("--max_eval_steps_per_instance", type=int, default=-1)

    parser.add_argument("--eval_train_split", action="store_true")
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--skip_test", action="store_true")

    args = parser.parse_args()
    if not args.run_tag.strip():
        parser.error("--run_tag cannot be empty.")
    if args.fallback_batch_size <= 0:
        parser.error("--fallback_batch_size must be positive.")
    if not 0 < args.train_ratio < 1:
        parser.error("--train_ratio must be strictly between 0 and 1.")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val_ratio must be in [0, 1).")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("--train_ratio + --val_ratio must be less than 1.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative.")
    if args.grad_clip_norm <= 0:
        parser.error("--grad_clip_norm must be positive.")
    if args.cuda_device_num < 0:
        parser.error("--cuda_device_num cannot be negative.")
    if args.train_step_stride <= 0 or args.eval_step_stride <= 0:
        parser.error("Step strides must be positive.")
    for field_name in (
        "max_train_steps_per_instance",
        "max_eval_steps_per_instance",
    ):
        value = getattr(args, field_name)
        if value == 0 or value < -1:
            parser.error(f"--{field_name} must be -1 or a positive integer.")

    def parse_map_for_validation(text: str, option_name: str):
        result = {}
        try:
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                key_text, value_text = item.split(":", maxsplit=1)
                result[int(key_text)] = int(value_text)
        except (TypeError, ValueError) as exc:
            parser.error(f"--{option_name} is invalid: {exc}")
        if not result:
            parser.error(f"--{option_name} cannot be empty.")
        return result

    max_instances = parse_map_for_validation(
        args.max_instances_per_distribution_by_n,
        "max_instances_per_distribution_by_n",
    )
    batch_sizes = parse_map_for_validation(args.batch_size_by_n, "batch_size_by_n")
    if any(key <= 1 for key in max_instances) or any(key <= 1 for key in batch_sizes):
        parser.error("Map keys must be problem sizes greater than 1.")
    if any(value == 0 or value < -1 for value in max_instances.values()):
        parser.error(
            "--max_instances_per_distribution_by_n values must be -1 or positive."
        )
    if any(value <= 0 for value in batch_sizes.values()):
        parser.error("--batch_size_by_n values must be positive.")
    try:
        nodes = [int(item.strip()) for item in args.nodes.split(",") if item.strip()]
        layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
        horizons = [
            int(item.strip()) for item in args.horizons.split(",") if item.strip()
        ]
    except ValueError as exc:
        parser.error(f"Invalid integer list: {exc}")
    if not nodes or any(value <= 1 for value in nodes):
        parser.error("--nodes must contain integers greater than 1.")
    if not layers or any(value < -1 for value in layers):
        parser.error("--layers must contain -1 or non-negative integers.")
    if not horizons or any(value <= 0 for value in horizons):
        parser.error("--horizons must contain positive integers.")
    return args


# ============================================================
# General utilities
# ============================================================

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_map(text: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not text.strip():
        return out
    for part in text.split(","):
        k, v = part.split(":")
        out[int(k.strip())] = int(v.strip())
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_pickle_load(path: str):
    install_numpy_pickle_compatibility_patch()
    with open(path, "rb") as f:
        return pickle.load(f)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_coords(coords) -> np.ndarray:
    """Return coordinates as float32 array with shape [N, 2]."""
    arr = to_numpy(coords).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T
    raise ValueError(f"Invalid coordinate shape: {arr.shape}")


def first_existing_key(d: dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in d:
            return key
    return None


def canonical_distribution_name(name: str) -> Optional[str]:
    key = str(name).strip().lower()
    mapping = {
        "cluster": "Clustered",
        "clustered": "Clustered",
        "expansion": "Expansion",
        "explosion": "Explosion",
        "grid": "Grid",
        "implosion": "Implosion",
        "mixed": "Mixed",
        "uniform": "Uniform",
    }
    return mapping.get(key)


def parse_distribution_and_n_from_filename(path: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse distribution and N from common TSP result filenames."""
    stem = os.path.splitext(os.path.basename(path))[0]
    pattern = re.compile(
        r"(clustered|cluster|expansion|explosion|grid|implosion|mixed|uniform)(\d+)",
        re.IGNORECASE,
    )
    match = pattern.search(stem)
    if match is None:
        return None, None
    dist = canonical_distribution_name(match.group(1))
    n = int(match.group(2))
    return dist, n


def select_step_indices(n: int, stride: int, max_steps: int) -> set:
    """Select decoding steps. Valid steps are 0..n-2."""
    steps = list(range(0, n - 1, max(1, stride)))
    if max_steps is not None and max_steps > 0 and len(steps) > max_steps:
        chosen = np.linspace(0, len(steps) - 1, num=max_steps)
        chosen = np.unique(np.round(chosen).astype(int))
        steps = [steps[i] for i in chosen]
    return set(steps)


def make_batches(records: List[InstanceRecord], batch_size: int, shuffle: bool, seed: int) -> Iterable[List[InstanceRecord]]:
    """Yield mini-batches of InstanceRecord objects."""
    items = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def records_to_coords_batch(records: List[InstanceRecord], device: torch.device) -> torch.Tensor:
    """Convert a list of records with the same N into a tensor [B, N, 2]."""
    coords = np.stack([r.coords for r in records], axis=0).astype(np.float32)
    return torch.tensor(coords, dtype=torch.float32, device=device)


def compute_ranks_batched(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return descending 1-based rank of each label in a batched score matrix.

    scores: [B, A]
    labels: [B]

    Rank 1 means the target candidate has the highest score.
    """
    target_scores = scores.gather(dim=1, index=labels[:, None]).squeeze(1)
    ranks = (scores > target_scores[:, None]).sum(dim=1) + 1
    return ranks.long()


def dense_ranks_from_scores_batched(scores: torch.Tensor, descending: bool = True) -> torch.Tensor:
    """Return a 1-based rank for every candidate in each row.

    Args:
        scores:
            Tensor with shape [B, A].
        descending:
            If True, larger scores get better ranks.
            If False, smaller scores get better ranks.

    Returns:
        ranks:
            Long tensor with shape [B, A].
            ranks[b, j] is the 1-based rank of candidate j in row b.

    Note:
        This function uses argsort, so ties are broken by PyTorch's sorting order.
        For our geometric/logit baselines this is acceptable because ties are rare.
    """
    batch_size, num_candidates = scores.shape
    order = torch.argsort(scores, dim=1, descending=descending)
    ranks = torch.empty_like(order, dtype=torch.long)
    values = torch.arange(1, num_candidates + 1, device=scores.device, dtype=torch.long)
    ranks.scatter_(dim=1, index=order, src=values[None, :].expand(batch_size, -1))
    return ranks


def score_candidates_by_rank_distance(candidate_ranks: torch.Tensor, horizon: int) -> torch.Tensor:
    """Score candidates by closeness of their ranking position to the horizon.

    This is the key horizon-aware baseline conversion.

    Example:
        For horizon=4, the candidate with rank/position 4 receives the best score.
        Candidates with rank 3 or 5 receive the next-best score, and so on.

    Args:
        candidate_ranks:
            Tensor [B, A] with 1-based ranks or rollout positions.
        horizon:
            Future horizon h.

    Returns:
        scores:
            Tensor [B, A], where higher is better.
    """
    return -torch.abs(candidate_ranks.float() - float(horizon))


def nearest_neighbor_rollout_positions_batched(
    coords_batch: torch.Tensor,
    available_nodes: torch.Tensor,
    current_nodes: torch.Tensor,
) -> torch.Tensor:
    """Compute each candidate's position in a greedy nearest-neighbor rollout.

    Starting from the current node, this baseline repeatedly selects the nearest
    remaining available node. It returns, for every candidate, at which rollout
    position it would be selected.

    Args:
        coords_batch:
            Tensor [B, N, 2].
        available_nodes:
            Tensor [B, A] containing available node ids at the current decoding step.
        current_nodes:
            Tensor [B] containing the current node id for each instance.

    Returns:
        positions:
            Tensor [B, A].
            positions[b, j] = 1 means candidate j is selected first by NN rollout.
            positions[b, j] = 4 means candidate j is selected fourth by NN rollout.

    Why this is fairer for horizon h:
        For horizon=4, the candidate selected fourth by the NN rollout becomes
        the top prediction of this baseline.
    """
    device = coords_batch.device
    batch_size, num_candidates = available_nodes.shape

    batch_arange = torch.arange(batch_size, device=device)
    available_xy = coords_batch[batch_arange[:, None], available_nodes.long()]  # [B, A, 2]
    current_xy = coords_batch[batch_arange, current_nodes.long()]               # [B, 2]

    remaining_mask = torch.ones((batch_size, num_candidates), dtype=torch.bool, device=device)
    positions = torch.empty((batch_size, num_candidates), dtype=torch.long, device=device)

    for pos in range(1, num_candidates + 1):
        distances = torch.norm(available_xy - current_xy[:, None, :], dim=2)
        distances = distances.masked_fill(~remaining_mask, float("inf"))

        chosen_col = distances.argmin(dim=1)  # [B]

        positions.scatter_(
            dim=1,
            index=chosen_col[:, None],
            src=torch.full((batch_size, 1), pos, dtype=torch.long, device=device),
        )

        remaining_mask.scatter_(dim=1, index=chosen_col[:, None], value=False)
        current_xy = available_xy[batch_arange, chosen_col]

    return positions


def write_csv(path: str, rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: str, rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    fieldnames = sorted(rows[0].keys())
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def csv_with_tag(filename: str, run_tag: str) -> str:
    """Return a CSV filename with the run tag inserted before .csv.

    Example:
        split_manifest.csv + horizons_5_100
        -> split_manifest_horizons_5_100.csv
    """
    if not filename.endswith(".csv"):
        return f"{filename}_{run_tag}"
    return filename[:-4] + f"_{run_tag}.csv"


def tagged_csv_path(directory: str, filename: str, run_tag: str) -> str:
    """Build a tagged CSV path inside a directory."""
    return os.path.join(directory, csv_with_tag(filename, run_tag))


# ============================================================
# Dataset loading
# ============================================================

def load_coords_from_plotfriendly_pkl(path: str) -> List[np.ndarray]:
    """Load coordinates from plot-friendly PKL result files."""
    data = safe_pickle_load(path)

    if isinstance(data, list):
        coords_list = []
        for item in data:
            if isinstance(item, dict):
                key = first_existing_key(item, ["coords", "coordinates", "problem", "problems", "nodes", "data"])
                if key is None:
                    raise ValueError(f"No coordinate key found in item from {path}")
                coords_list.append(normalize_coords(item[key]))
            elif isinstance(item, (tuple, list)) and len(item) >= 1:
                coords_list.append(normalize_coords(item[0]))
            else:
                coords_list.append(normalize_coords(item))
        return coords_list

    if isinstance(data, dict):
        key = first_existing_key(data, ["coords", "coordinates", "problems", "nodes", "data"])
        if key is None:
            raise ValueError(f"No coordinate key found in dict from {path}")
        arr = to_numpy(data[key])
        if arr.ndim == 3:
            return [normalize_coords(arr[i]) for i in range(arr.shape[0])]
        return [normalize_coords(arr)]

    if isinstance(data, tuple):
        arr = to_numpy(data[0])
        if arr.ndim == 3:
            return [normalize_coords(arr[i]) for i in range(arr.shape[0])]
        return [normalize_coords(arr)]

    arr = to_numpy(data)
    if arr.ndim == 3:
        return [normalize_coords(arr[i]) for i in range(arr.shape[0])]
    return [normalize_coords(arr)]


def split_indices(num_items: int, train_ratio: float, val_ratio: float, seed: int):
    """Create sequential train/validation/test indices.

    Important:
        We intentionally do NOT shuffle here.

        The split is:
            first train_ratio      -> train
            next val_ratio         -> validation
            remaining instances    -> test

        Example for 1000 instances:
            0..699   -> train
            700..849 -> validation
            850..999 -> test

    The seed argument is kept only for API compatibility with older scripts.
    """
    del seed  # The split is deterministic and sequential.

    indices = np.arange(num_items)

    n_train = int(num_items * train_ratio)
    n_val = int(num_items * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def build_instance_records(
    inference_dirs: List[str],
    target_nodes: List[int],
    max_instances_by_n: Dict[int, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> List[InstanceRecord]:
    """Load all instances and assign split labels."""
    pkl_files: List[str] = []
    for directory in inference_dirs:
        pkl_files.extend(sorted(glob.glob(os.path.join(directory, "*.pkl"))))

    records: List[InstanceRecord] = []

    for path in pkl_files:
        distribution, n = parse_distribution_and_n_from_filename(path)
        if distribution is None or n is None:
            print(f"WARNING: could not parse distribution/N from: {path}")
            continue
        if n not in target_nodes:
            continue
        if distribution not in DISTRIBUTION_CODE:
            continue

        coords_list = load_coords_from_plotfriendly_pkl(path)
        max_instances = max_instances_by_n.get(n, -1)
        if max_instances >= 0:
            coords_list = coords_list[:max_instances]

        dist_code = DISTRIBUTION_CODE[distribution]
        train_idx, val_idx, test_idx = split_indices(
            num_items=len(coords_list),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + n + dist_code,
        )

        split_map: Dict[int, str] = {}
        for idx in train_idx:
            split_map[int(idx)] = "train"
        for idx in val_idx:
            split_map[int(idx)] = "validation"
        for idx in test_idx:
            split_map[int(idx)] = "test"

        for instance_id, coords in enumerate(coords_list):
            records.append(
                InstanceRecord(
                    coords=coords,
                    n=n,
                    distribution=distribution,
                    distribution_code=dist_code,
                    instance_id=instance_id,
                    split=split_map[instance_id],
                    source_file=path,
                )
            )

    return records


def records_to_manifest_rows(records: List[InstanceRecord]) -> List[Dict]:
    """Convert records to CSV rows so the exact split is auditable.

    This makes it easy to know exactly which instances are train/validation/test.
    In particular, the test set is the last 15% of each loaded source file
    after applying max_instances_per_distribution_by_n.
    """
    rows = []
    for r in records:
        rows.append(
            {
                "n": r.n,
                "distribution": r.distribution,
                "distribution_code": r.distribution_code,
                "instance_id": r.instance_id,
                "split": r.split,
                "source_file": r.source_file,
                "split_rule": "sequential_first_70_next_15_last_15_within_each_source_file",
            }
        )
    return rows


# ============================================================
# LEHD model loading
# ============================================================

def import_lehd_model(lehd_root: str):
    """Import original LEHD TSPModel."""
    sys.path.insert(0, lehd_root)
    from LEHD.TSP.TSPModel import TSPModel as Model
    return Model


def load_torch_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_state_dict_from_checkpoint(checkpoint):
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def infer_decoder_layer_num(state_dict: dict, fallback: int = 6) -> int:
    """Infer decoder layer count from checkpoint keys."""
    layer_indices = []
    patterns = [re.compile(r"decoder\.layers\.(\d+)\."), re.compile(r"layers\.(\d+)\.")]
    for key in state_dict.keys():
        for pattern in patterns:
            match = pattern.search(key)
            if match is not None:
                layer_indices.append(int(match.group(1)))
    if len(layer_indices) == 0:
        print(f"WARNING: could not infer decoder_layer_num. Using fallback={fallback}")
        return fallback
    return max(layer_indices) + 1


def build_lehd_model(args, device: torch.device):
    """Build and load frozen LEHD model."""
    Model = import_lehd_model(args.lehd_root)
    checkpoint = load_torch_checkpoint(args.checkpoint_path, device)
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    decoder_layer_num = infer_decoder_layer_num(state_dict, fallback=6)

    model_params = {
        "mode": "test",
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "decoder_layer_num": decoder_layer_num,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
    }

    model = Model(**model_params)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.mode = "test"

    # Freeze LEHD. Only probes are trained.
    for p in model.parameters():
        p.requires_grad_(False)

    return model, decoder_layer_num


# ============================================================
# Manual LEHD decoder forward
# ============================================================

def gather_encoded_nodes(encoded_nodes: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    """Gather encoded node vectors by node indices.

    encoded_nodes: [B, N, D]
    node_indices  : [B, K]
    return        : [B, K, D]
    """
    batch_size = node_indices.size(0)
    pick_size = node_indices.size(1)
    embedding_dim = encoded_nodes.size(2)
    gather_idx = node_indices[:, :, None].expand(batch_size, pick_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gather_idx.long())


def get_available_nodes(selected_node_list: torch.Tensor, problem_size: int) -> torch.Tensor:
    """Return nodes that have not been selected yet.

    selected_node_list: [B, selected_len]
    return            : [B, problem_size - selected_len]
    """
    device = selected_node_list.device
    batch_size = selected_node_list.size(0)
    selected_len = selected_node_list.size(1)

    all_nodes = torch.arange(problem_size, dtype=torch.long, device=device)[None, :].repeat(batch_size, 1)
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=device)[:, None].expand(batch_size, selected_len)
    all_nodes[batch_idx, selected_node_list.long()] = -1
    return all_nodes[all_nodes >= 0].view(batch_size, problem_size - selected_len)


def manual_decoder_forward_with_reps_and_logits(
    model,
    encoded_nodes: torch.Tensor,
    selected_node_list: torch.Tensor,
    layers_to_collect: List[int],
):
    """
    Run LEHD decoder manually in batched mode.

    Layer convention:
        layer -1:
            Raw encoder output of the available candidate nodes.
            Shape: [B, A, D]

        layer k for k >= 0:
            Output of decoder layer k for the available candidate nodes.
            Shape: [B, A, D]

    Returns:
        selected_next    : [B]
        available_nodes  : [B, A]
        layer_reps       : dict[layer] -> [B, A, D]
        available_logits : [B, A]
    """
    decoder = model.decoder
    _, problem_size, _ = encoded_nodes.shape

    selected_node_list = selected_node_list.long()
    available_nodes = get_available_nodes(selected_node_list, problem_size)

    # Raw encoder representations for currently available candidates.
    # This is exactly the representation used for layer -1.
    available_encoded = gather_encoded_nodes(encoded_nodes, available_nodes)

    layer_reps = {}

    if -1 in layers_to_collect:
        # layer -1 is before any decoder layer.
        # We clone only to make the semantic boundary explicit; gradients do not
        # flow into LEHD because the model is frozen and this function is called
        # under no_grad when extracting representations.
        layer_reps[-1] = available_encoded.clone()

    first_last = gather_encoded_nodes(encoded_nodes, selected_node_list[:, [0, -1]])
    first_encoded = first_last[:, 0]
    last_encoded = first_last[:, 1]

    # LEHD decoder special tokens: start and current.
    first_token = decoder.embedding_first_node(first_encoded).unsqueeze(1)
    last_token = decoder.embedding_last_node(last_encoded).unsqueeze(1)

    # Token layout: [start token] + [available candidates] + [current token]
    out = torch.cat((first_token, available_encoded, last_token), dim=1)

    for layer_idx, layer in enumerate(decoder.layers):
        out = layer(out)

        if layer_idx in layers_to_collect:
            # Keep only available candidate token representations.
            # The first token is start, and the last token is current.
            layer_reps[layer_idx] = out[:, 1:-1, :]

    logits = decoder.Linear_final(out).squeeze(-1)
    logits[:, [0, -1]] = float("-inf")
    available_logits = logits[:, 1:-1]

    selected_pos = available_logits.argmax(dim=1)
    selected_next = available_nodes.gather(dim=1, index=selected_pos[:, None]).squeeze(1)

    return selected_next, available_nodes, layer_reps, available_logits


def rollout_lehd_tour_batch(model, coords_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run greedy LEHD rollout for a batch of instances.

    coords_batch: [B, N, 2]

    Returns:
        tour_batch    : [B, N]
        encoded_nodes : [B, N, D]
    """
    batch_size, n, _ = coords_batch.shape

    with torch.no_grad():
        encoded_nodes = model.encoder(coords_batch)

    selected_node_list = torch.zeros((batch_size, 1), dtype=torch.long, device=coords_batch.device)

    for _ in range(n - 1):
        with torch.no_grad():
            selected_next, _, _, _ = manual_decoder_forward_with_reps_and_logits(
                model=model,
                encoded_nodes=encoded_nodes,
                selected_node_list=selected_node_list,
                layers_to_collect=[],
            )
        selected_node_list = torch.cat([selected_node_list, selected_next[:, None].long()], dim=1)

    return selected_node_list, encoded_nodes


# ============================================================
# Probe models
# ============================================================

class CandidateLinearProbe(nn.Module):
    """Linear candidate ranker.

    Input can be:
        [A, D]    -> output [A]
        [B, A, D] -> output [B, A]
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x).squeeze(-1)


class FutureProbeBank(nn.Module):
    """All embedding probes and coordinate baseline probes for one N."""

    def __init__(self, layers: List[int], horizons: List[int], embedding_dim: int, coord_dim: int):
        super().__init__()
        self.layers = list(layers)
        self.horizons = list(horizons)

        self.embedding_probes = nn.ModuleDict()
        for layer in self.layers:
            for horizon in self.horizons:
                self.embedding_probes[self.embedding_key(layer, horizon)] = CandidateLinearProbe(embedding_dim)

        self.coord_probes = nn.ModuleDict()
        for horizon in self.horizons:
            self.coord_probes[self.coord_key(horizon)] = CandidateLinearProbe(coord_dim)

    @staticmethod
    def embedding_key(layer: int, horizon: int) -> str:
        return f"layer_{layer}__horizon_{horizon}"

    @staticmethod
    def coord_key(horizon: int) -> str:
        return f"horizon_{horizon}"

    def get_embedding_probe(self, layer: int, horizon: int) -> CandidateLinearProbe:
        return self.embedding_probes[self.embedding_key(layer, horizon)]

    def get_coord_probe(self, horizon: int) -> CandidateLinearProbe:
        return self.coord_probes[self.coord_key(horizon)]


# ============================================================
# Feature and label helpers
# ============================================================

def build_coord_features_batch(
    coords_batch: torch.Tensor,
    available_nodes: torch.Tensor,
    start_nodes: torch.Tensor,
    current_nodes: torch.Tensor,
    step_idx: int,
    n: int,
) -> torch.Tensor:
    """Coordinate-only baseline features in batched mode.

    coords_batch   : [B, N, 2]
    available_nodes: [B, A]
    start_nodes    : [B]
    current_nodes  : [B]

    Returns:
        features: [B, A, 14]
    """
    batch_size, a = available_nodes.shape
    device = coords_batch.device
    batch_idx = torch.arange(batch_size, device=device)[:, None]

    candidate_xy = coords_batch[batch_idx, available_nodes.long()]  # [B, A, 2]
    current_xy = coords_batch[torch.arange(batch_size, device=device), current_nodes.long()][:, None, :].expand_as(candidate_xy)
    start_xy = coords_batch[torch.arange(batch_size, device=device), start_nodes.long()][:, None, :].expand_as(candidate_xy)

    cand_minus_current = candidate_xy - current_xy
    cand_minus_start = candidate_xy - start_xy

    dist_current = torch.norm(cand_minus_current, dim=2, keepdim=True)
    dist_start = torch.norm(cand_minus_start, dim=2, keepdim=True)

    step_fraction = torch.full(
        (batch_size, a, 1),
        float(step_idx) / float(max(n - 1, 1)),
        dtype=coords_batch.dtype,
        device=device,
    )
    remaining_fraction = torch.full(
        (batch_size, a, 1),
        float(a) / float(max(n, 1)),
        dtype=coords_batch.dtype,
        device=device,
    )

    return torch.cat(
        [
            candidate_xy,
            current_xy,
            start_xy,
            cand_minus_current,
            cand_minus_start,
            dist_current,
            dist_start,
            step_fraction,
            remaining_fraction,
        ],
        dim=2,
    )


def get_label_indices_batched(available_nodes: torch.Tensor, target_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find each target node inside each row of available_nodes.

    available_nodes: [B, A]
    target_nodes   : [B]

    Returns:
        labels: [B]
        valid : [B]
    """
    matches = available_nodes.long().eq(target_nodes.long()[:, None])
    valid = matches.any(dim=1)
    labels = matches.float().argmax(dim=1).long()
    return labels, valid


# ============================================================
# Training
# ============================================================

def train_one_epoch_for_n(
    n: int,
    epoch: int,
    model,
    probe_bank: FutureProbeBank,
    optimizer: torch.optim.Optimizer,
    train_records: List[InstanceRecord],
    layers: List[int],
    horizons: List[int],
    device: torch.device,
    batch_size: int,
    train_step_stride: int,
    max_train_steps_per_instance: int,
    grad_clip_norm: float,
    seed: int,
) -> Dict:
    """Train all probes for one N for one epoch using batched instances."""
    probe_bank.train()

    total_loss = 0.0
    total_updates = 0
    total_tasks = 0
    total_instances = 0

    selected_steps = select_step_indices(n, train_step_stride, max_train_steps_per_instance)

    for batch_idx, batch_records in enumerate(
        make_batches(train_records, batch_size=batch_size, shuffle=True, seed=seed + 1000 * epoch + n)
    ):
        coords_batch = records_to_coords_batch(batch_records, device=device)
        bsz = coords_batch.size(0)

        # LEHD future trajectory is the ground truth.
        with torch.no_grad():
            tour_batch, encoded_nodes = rollout_lehd_tour_batch(model, coords_batch)

        for step_idx in range(n - 1):
            # Follow the already-computed LEHD rollout.
            selected_node_list = tour_batch[:, :step_idx + 1]

            with torch.no_grad():
                _, available_nodes, layer_reps, _ = manual_decoder_forward_with_reps_and_logits(
                    model=model,
                    encoded_nodes=encoded_nodes,
                    selected_node_list=selected_node_list,
                    layers_to_collect=layers,
                )

            if step_idx not in selected_steps:
                continue

            start_nodes = tour_batch[:, 0]
            current_nodes = tour_batch[:, step_idx]

            coord_features = build_coord_features_batch(
                coords_batch=coords_batch,
                available_nodes=available_nodes,
                start_nodes=start_nodes,
                current_nodes=current_nodes,
                step_idx=step_idx,
                n=n,
            )

            step_losses = []

            for horizon in horizons:
                future_step = step_idx + horizon
                if future_step >= n:
                    continue

                target_nodes = tour_batch[:, future_step]
                labels, valid = get_label_indices_batched(available_nodes, target_nodes)
                if valid.sum().item() == 0:
                    continue

                labels_valid = labels[valid]

                # Coordinate-only trainable baseline.
                coord_scores = probe_bank.get_coord_probe(horizon)(coord_features)  # [B, A]
                step_losses.append(F.cross_entropy(coord_scores[valid], labels_valid))

                # Embedding probes for each decoder layer.
                for layer in layers:
                    x = layer_reps[layer]  # [B, A, D]
                    emb_scores = probe_bank.get_embedding_probe(layer, horizon)(x)  # [B, A]
                    step_losses.append(F.cross_entropy(emb_scores[valid], labels_valid))

            if len(step_losses) > 0:
                loss = torch.stack(step_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(probe_bank.parameters(), grad_clip_norm)
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_updates += 1
                total_tasks += len(step_losses) * int(bsz)

        total_instances += bsz

        if (batch_idx + 1) % 10 == 0:
            print(
                f"    [N={n} epoch={epoch}] batches={batch_idx + 1} | "
                f"instances={total_instances}/{len(train_records)} | "
                f"avg_loss={total_loss / max(total_updates, 1):.6f} | updates={total_updates}"
            )

    return {
        "n": n,
        "epoch": epoch,
        "batch_size": batch_size,
        "train_instances": total_instances,
        "train_updates": total_updates,
        "train_tasks": total_tasks,
        "train_avg_loss": total_loss / max(total_updates, 1),
    }


# ============================================================
# Evaluation
# ============================================================

def metric_key(model_type: str, layer: int, horizon: int):
    return model_type, int(layer), int(horizon)


def update_score_metric_batched(
    metrics: Dict,
    model_type: str,
    layer: int,
    horizon: int,
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Update ranking metrics from batched scores."""
    if valid.sum().item() == 0:
        return

    scores_valid = scores[valid]
    labels_valid = labels[valid]
    ranks = compute_ranks_batched(scores_valid, labels_valid)
    num_candidates = int(scores.size(1))

    key = metric_key(model_type, layer, horizon)
    if key not in metrics:
        metrics[key] = MetricState()
    metrics[key].update_ranks(ranks, num_candidates=num_candidates)


def update_random_metric_batched(metrics: Dict, horizon: int, num_candidates: int, valid_count: int) -> None:
    key = metric_key("random_expected", -1, horizon)
    if key not in metrics:
        metrics[key] = MetricState()
    metrics[key].update_random_expected_many(num_candidates=num_candidates, count=valid_count)


def evaluate_split_for_n(
    n: int,
    split_name: str,
    epoch: int,
    model,
    probe_bank: FutureProbeBank,
    records: List[InstanceRecord],
    layers: List[int],
    horizons: List[int],
    device: torch.device,
    batch_size: int,
    eval_step_stride: int,
    max_eval_steps_per_instance: int,
) -> List[Dict]:
    """Evaluate probes and baselines on one split using batched instances."""
    probe_bank.eval()
    metrics: Dict[Tuple[str, int, int], MetricState] = {}

    selected_steps = select_step_indices(n, eval_step_stride, max_eval_steps_per_instance)

    with torch.no_grad():
        for batch_idx, batch_records in enumerate(make_batches(records, batch_size=batch_size, shuffle=False, seed=0)):
            coords_batch = records_to_coords_batch(batch_records, device=device)
            bsz = coords_batch.size(0)

            tour_batch, encoded_nodes = rollout_lehd_tour_batch(model, coords_batch)

            for step_idx in range(n - 1):
                selected_node_list = tour_batch[:, :step_idx + 1]

                _, available_nodes, layer_reps, available_logits = manual_decoder_forward_with_reps_and_logits(
                    model=model,
                    encoded_nodes=encoded_nodes,
                    selected_node_list=selected_node_list,
                    layers_to_collect=layers,
                )

                if step_idx not in selected_steps:
                    continue

                start_nodes = tour_batch[:, 0]
                current_nodes = tour_batch[:, step_idx]

                coord_features = build_coord_features_batch(
                    coords_batch=coords_batch,
                    available_nodes=available_nodes,
                    start_nodes=start_nodes,
                    current_nodes=current_nodes,
                    step_idx=step_idx,
                    n=n,
                )

                batch_arange_col = torch.arange(bsz, device=device)[:, None]
                batch_arange = torch.arange(bsz, device=device)

                available_xy = coords_batch[batch_arange_col, available_nodes.long()]  # [B, A, 2]
                current_xy = coords_batch[batch_arange, current_nodes.long()][:, None, :]
                start_xy = coords_batch[batch_arange, start_nodes.long()][:, None, :]

                # ------------------------------------------------------------
                # Static diagnostic baselines.
                # These are NOT horizon-aware. They ask:
                #   "Where does the future node rank according to the current
                #    one-step distance/logit signal?"
                # ------------------------------------------------------------
                dist_current = torch.norm(available_xy - current_xy, dim=2)
                dist_start = torch.norm(available_xy - start_xy, dim=2)

                nearest_current_static_scores = -dist_current
                nearest_start_static_scores = -dist_start
                lehd_current_logits_static_scores = available_logits

                # ------------------------------------------------------------
                # Horizon-aware order baselines.
                # For horizon h, the candidate whose distance/logit rank is h
                # is treated as the top prediction.
                # ------------------------------------------------------------
                current_distance_ranks = dense_ranks_from_scores_batched(dist_current, descending=False)
                start_distance_ranks = dense_ranks_from_scores_batched(dist_start, descending=False)
                lehd_logit_ranks = dense_ranks_from_scores_batched(available_logits, descending=True)

                # ------------------------------------------------------------
                # Horizon-aware nearest-neighbor rollout baseline.
                # For horizon h, the candidate selected at rollout position h
                # is treated as the top prediction.
                # ------------------------------------------------------------
                nn_rollout_positions = nearest_neighbor_rollout_positions_batched(
                    coords_batch=coords_batch,
                    available_nodes=available_nodes,
                    current_nodes=current_nodes,
                )

                for horizon in horizons:
                    future_step = step_idx + horizon
                    if future_step >= n:
                        continue

                    target_nodes = tour_batch[:, future_step]
                    labels, valid = get_label_indices_batched(available_nodes, target_nodes)
                    valid_count = int(valid.sum().item())
                    if valid_count == 0:
                        continue

                    num_candidates = int(available_nodes.size(1))

                    update_random_metric_batched(metrics, horizon, num_candidates, valid_count)

                    # Static diagnostics, kept mainly for interpretability.
                    update_score_metric_batched(
                        metrics, "nearest_current_static", -1, horizon,
                        nearest_current_static_scores, labels, valid,
                    )
                    update_score_metric_batched(
                        metrics, "nearest_start_static", -1, horizon,
                        nearest_start_static_scores, labels, valid,
                    )
                    update_score_metric_batched(
                        metrics, "lehd_current_logits_static", -1, horizon,
                        lehd_current_logits_static_scores, labels, valid,
                    )

                    # Horizon-aware fairer baselines.
                    nearest_current_order_scores = score_candidates_by_rank_distance(
                        current_distance_ranks, horizon=horizon
                    )
                    nearest_start_order_scores = score_candidates_by_rank_distance(
                        start_distance_ranks, horizon=horizon
                    )
                    lehd_logit_order_scores = score_candidates_by_rank_distance(
                        lehd_logit_ranks, horizon=horizon
                    )
                    nn_rollout_scores = score_candidates_by_rank_distance(
                        nn_rollout_positions, horizon=horizon
                    )

                    update_score_metric_batched(
                        metrics, "nearest_current_order_h", -1, horizon,
                        nearest_current_order_scores, labels, valid,
                    )
                    update_score_metric_batched(
                        metrics, "nearest_start_order_h", -1, horizon,
                        nearest_start_order_scores, labels, valid,
                    )
                    update_score_metric_batched(
                        metrics, "lehd_logit_order_h", -1, horizon,
                        lehd_logit_order_scores, labels, valid,
                    )
                    update_score_metric_batched(
                        metrics, "nearest_neighbor_rollout_h", -1, horizon,
                        nn_rollout_scores, labels, valid,
                    )

                    # Trainable coordinate-only baseline.
                    coord_scores = probe_bank.get_coord_probe(horizon)(coord_features)
                    update_score_metric_batched(metrics, "coord_linear_probe", -1, horizon, coord_scores, labels, valid)

                    # Main embedding probes.
                    for layer in layers:
                        emb_scores = probe_bank.get_embedding_probe(layer, horizon)(layer_reps[layer])
                        update_score_metric_batched(metrics, "embedding_probe", layer, horizon, emb_scores, labels, valid)

            if (batch_idx + 1) % 10 == 0:
                done = min((batch_idx + 1) * batch_size, len(records))
                print(f"    [N={n} split={split_name}] batches={batch_idx + 1} | instances={done}/{len(records)}")

    rows = []
    for (model_type, layer, horizon), state in sorted(metrics.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        row = {
            "n": n,
            "split": split_name,
            "epoch": epoch,
            "model_type": model_type,
            "layer": layer,
            "horizon": horizon,
        }
        row.update(state.to_row())
        rows.append(row)

    return rows


# ============================================================
# Checkpoint saving
# ============================================================

def save_probe_checkpoints_for_n(
    output_root: str,
    n: int,
    epoch: int,
    probe_bank: FutureProbeBank,
    optimizer: torch.optim.Optimizer,
    layers: List[int],
    horizons: List[int],
    embedding_dim: int,
    coord_dim: int,
    run_tag: str = "horizons_5_100",
) -> None:
    """Save combined and separate probe checkpoints.

    This version is safe for running horizons 5..100 in an output directory
    that may already contain shorter-horizon runs.

    It does NOT overwrite:
        checkpoints/all_probes_latest.pt

    Instead, it writes:
        checkpoints/all_probes_<run_tag>_latest.pt
        checkpoints/all_probes_<run_tag>_epoch_XXX.pt

    Individual horizon folders are naturally separate:
        layer_0/horizon_5/
        layer_0/horizon_6/
        ...
        layer_0/horizon_100/

    Note:
        The horizons list passed here is already filtered for the current N.
        For example:
            N=20  -> horizons 5..19
            N=50  -> horizons 5..49
            N=100 -> horizons 5..99
            N>=101 -> horizons 5..100
    """
    n_root = os.path.join(output_root, f"N_{n}")
    combined_dir = os.path.join(n_root, "checkpoints")
    ensure_dir(combined_dir)

    safe_tag = str(run_tag).replace("/", "_").replace(" ", "_")

    payload = {
        "n": n,
        "epoch": epoch,
        "layers": layers,
        "horizons": horizons,
        "embedding_dim": embedding_dim,
        "coord_dim": coord_dim,
        "run_tag": safe_tag,
        "probe_bank_state_dict": probe_bank.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(payload, os.path.join(combined_dir, f"all_probes_{safe_tag}_epoch_{epoch:03d}.pt"))
    torch.save(payload, os.path.join(combined_dir, f"all_probes_{safe_tag}_latest.pt"))

    for layer in layers:
        for horizon in horizons:
            ckpt_dir = os.path.join(n_root, f"layer_{layer}", f"horizon_{horizon}", "checkpoints")
            ensure_dir(ckpt_dir)
            probe = probe_bank.get_embedding_probe(layer, horizon)
            torch.save(
                {
                    "n": n,
                    "epoch": epoch,
                    "layer": layer,
                    "horizon": horizon,
                    "input_dim": embedding_dim,
                    "run_tag": safe_tag,
                    "probe_type": "embedding_linear_candidate_ranker",
                    "model_state_dict": probe.state_dict(),
                },
                os.path.join(ckpt_dir, "future_node_probe.pt"),
            )

    for horizon in horizons:
        ckpt_dir = os.path.join(n_root, "coordinate_baseline", f"horizon_{horizon}", "checkpoints")
        ensure_dir(ckpt_dir)
        probe = probe_bank.get_coord_probe(horizon)
        torch.save(
            {
                "n": n,
                "epoch": epoch,
                "horizon": horizon,
                "input_dim": coord_dim,
                "run_tag": safe_tag,
                "probe_type": "coordinate_linear_candidate_ranker",
                "feature_names": [
                    "candidate_x", "candidate_y",
                    "current_x", "current_y",
                    "start_x", "start_y",
                    "candidate_minus_current_x", "candidate_minus_current_y",
                    "candidate_minus_start_x", "candidate_minus_start_y",
                    "dist_candidate_current", "dist_candidate_start",
                    "step_fraction", "remaining_fraction",
                ],
                "model_state_dict": probe.state_dict(),
            },
            os.path.join(ckpt_dir, "coord_linear_probe.pt"),
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    nodes = parse_int_list(args.nodes)
    layers = parse_int_list(args.layers)
    horizons = parse_int_list(args.horizons)
    max_instances_by_n = parse_int_map(args.max_instances_per_distribution_by_n)
    batch_size_by_n = parse_int_map(args.batch_size_by_n)
    inference_dirs = [os.path.abspath(os.path.expanduser(x)) for x in args.inference_dirs]

    ensure_dir(args.output_root)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device_num)
        device = torch.device(f"cuda:{args.cuda_device_num}")
    else:
        device = torch.device("cpu")

    print("=" * 90)
    print("Batched future-node planning probe experiment with layer -1 encoder-output probes, horizons 5..100")
    print("Output root:", args.output_root)
    print("Nodes:", nodes)
    print("Layers:", layers)
    print("Horizons:", horizons)
    print("Batch size by N:", batch_size_by_n)
    print("Device:", device)
    print("=" * 90)

    print("Loading coordinate records and reconstructing splits...")
    all_records = build_instance_records(
        inference_dirs=inference_dirs,
        target_nodes=nodes,
        max_instances_by_n=max_instances_by_n,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Total records loaded: {len(all_records)}")

    # Save an auditable manifest of the exact sequential split.
    # This is especially useful for knowing exactly which instances are in test.
    split_manifest_rows = records_to_manifest_rows(all_records)
    write_csv(tagged_csv_path(args.output_root, "split_manifest.csv", args.run_tag), split_manifest_rows)
    write_csv(
        tagged_csv_path(args.output_root, "test_manifest.csv", args.run_tag),
        [row for row in split_manifest_rows if row["split"] == "test"],
    )
    print("Saved split manifest:", tagged_csv_path(args.output_root, "split_manifest.csv", args.run_tag))
    print("Saved test manifest :", tagged_csv_path(args.output_root, "test_manifest.csv", args.run_tag))

    print("Loading frozen LEHD model...")
    lehd_model, decoder_layer_num = build_lehd_model(args, device)
    print(f"LEHD loaded. Decoder layers in checkpoint: {decoder_layer_num}")

    # Validate requested layers.
    # Valid values are:
    #   -1                        -> encoder output
    #    0..decoder_layer_num - 1 -> decoder layer outputs
    for layer in layers:
        if layer < -1 or layer >= decoder_layer_num:
            raise ValueError(
                f"Invalid layer={layer}. Valid range is -1..{decoder_layer_num - 1}."
            )

    embedding_dim = 128
    coord_dim = 14

    for n in nodes:
        print("\n" + "=" * 90)
        print(f"Starting N={n}")
        print("=" * 90)

        batch_size = batch_size_by_n.get(n, args.fallback_batch_size)
        print(f"Using batch_size={batch_size} for N={n}")

        # Only horizons h < N can be valid because target tour[t+h] must exist.
        # This is important for small problem sizes:
        #   N=20  -> active horizons 5..19
        #   N=50  -> active horizons 5..49
        #   N=100 -> active horizons 5..99
        #   N>=146 -> active horizons 5..100
        active_horizons = [h for h in horizons if h < n]

        if len(active_horizons) == 0:
            print(f"WARNING: no valid horizons for N={n}. Skipping this N.")
            continue

        print(f"Active horizons for N={n}: {active_horizons[0]}..{active_horizons[-1]} "
              f"({len(active_horizons)} horizons)")

        n_records = [r for r in all_records if r.n == n]
        train_records = [r for r in n_records if r.split == "train"]
        val_records = [r for r in n_records if r.split == "validation"]
        test_records = [r for r in n_records if r.split == "test"]

        print(f"N={n}: train={len(train_records)}, validation={len(val_records)}, test={len(test_records)}")

        n_root = os.path.join(args.output_root, f"N_{n}")
        ensure_dir(n_root)

        # Save per-N split manifests.
        n_manifest_rows = records_to_manifest_rows(n_records)
        write_csv(tagged_csv_path(n_root, "split_manifest.csv", args.run_tag), n_manifest_rows)
        write_csv(
            tagged_csv_path(n_root, "test_manifest.csv", args.run_tag),
            [row for row in n_manifest_rows if row["split"] == "test"],
        )

        if len(train_records) == 0:
            print(f"WARNING: no train records for N={n}. Skipping.")
            continue

        probe_bank = FutureProbeBank(
            layers=layers,
            horizons=active_horizons,
            embedding_dim=embedding_dim,
            coord_dim=coord_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(probe_bank.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_log_rows = []

        for epoch in range(1, args.epochs + 1):
            print("\n" + "-" * 90)
            print(f"Training N={n}, epoch={epoch}/{args.epochs}")
            t0 = time.time()

            train_log = train_one_epoch_for_n(
                n=n,
                epoch=epoch,
                model=lehd_model,
                probe_bank=probe_bank,
                optimizer=optimizer,
                train_records=train_records,
                layers=layers,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                train_step_stride=args.train_step_stride,
                max_train_steps_per_instance=args.max_train_steps_per_instance,
                grad_clip_norm=args.grad_clip_norm,
                seed=args.seed,
            )
            train_log["elapsed_seconds"] = time.time() - t0
            train_log["run_tag"] = args.run_tag
            train_log["active_horizons"] = ",".join(str(h) for h in active_horizons)
            train_log_rows.append(train_log)
            write_csv(tagged_csv_path(n_root, "train_log.csv", args.run_tag), train_log_rows)
            append_csv(tagged_csv_path(args.output_root, "all_train_log.csv", args.run_tag), [train_log])

            print(
                f"Finished epoch={epoch}: avg_loss={train_log['train_avg_loss']:.6f}, "
                f"updates={train_log['train_updates']}, elapsed={train_log['elapsed_seconds']:.1f}s"
            )

            print(f"Evaluating validation split for N={n}, epoch={epoch}...")
            val_rows = evaluate_split_for_n(
                n=n,
                split_name="validation",
                epoch=epoch,
                model=lehd_model,
                probe_bank=probe_bank,
                records=val_records,
                layers=layers,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                eval_step_stride=args.eval_step_stride,
                max_eval_steps_per_instance=args.max_eval_steps_per_instance,
            )
            # Save tagged validation metrics so existing files are not overwritten.
            write_csv(
                tagged_csv_path(n_root, f"validation_metrics_epoch_{epoch:03d}.csv", args.run_tag),
                val_rows,
            )
            append_csv(
                tagged_csv_path(args.output_root, "all_validation_metrics.csv", args.run_tag),
                val_rows,
            )

            if args.append_to_global_csv:
                append_csv(os.path.join(args.output_root, "all_validation_metrics.csv"), val_rows)

            if args.save_every_epoch or epoch == args.epochs:
                print(f"Saving checkpoints for N={n}, epoch={epoch}...")
                save_probe_checkpoints_for_n(
                    output_root=args.output_root,
                    n=n,
                    epoch=epoch,
                    probe_bank=probe_bank,
                    optimizer=optimizer,
                    layers=layers,
                    horizons=active_horizons,
                    embedding_dim=embedding_dim,
                    coord_dim=coord_dim,
                    run_tag=args.run_tag,
                )

        if args.eval_train_split:
            print(f"Evaluating train split for N={n} after final epoch...")
            train_eval_rows = evaluate_split_for_n(
                n=n,
                split_name="train",
                epoch=args.epochs,
                model=lehd_model,
                probe_bank=probe_bank,
                records=train_records,
                layers=layers,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                eval_step_stride=args.eval_step_stride,
                max_eval_steps_per_instance=args.max_eval_steps_per_instance,
            )
            write_csv(tagged_csv_path(n_root, "train_metrics_final.csv", args.run_tag), train_eval_rows)
            append_csv(tagged_csv_path(args.output_root, "all_train_metrics_final.csv", args.run_tag), train_eval_rows)

            if args.append_to_global_csv:
                append_csv(os.path.join(args.output_root, "all_train_metrics_final.csv"), train_eval_rows)

        if not args.skip_test:
            print(f"Evaluating test split for N={n} after final epoch...")
            test_rows = evaluate_split_for_n(
                n=n,
                split_name="test",
                epoch=args.epochs,
                model=lehd_model,
                probe_bank=probe_bank,
                records=test_records,
                layers=layers,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                eval_step_stride=args.eval_step_stride,
                max_eval_steps_per_instance=args.max_eval_steps_per_instance,
            )
            # Save tagged test metrics so existing files are not overwritten.
            write_csv(tagged_csv_path(n_root, "test_metrics.csv", args.run_tag), test_rows)
            append_csv(
                tagged_csv_path(args.output_root, "all_test_metrics.csv", args.run_tag),
                test_rows,
            )

            if args.append_to_global_csv:
                append_csv(os.path.join(args.output_root, "all_test_metrics.csv"), test_rows)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("Experiment finished.")
    print("Output root:", args.output_root)
    print("Tagged global validation CSV:", tagged_csv_path(args.output_root, "all_validation_metrics.csv", args.run_tag))
    print("Tagged global test CSV:", tagged_csv_path(args.output_root, "all_test_metrics.csv", args.run_tag))
    print("=" * 90)


if __name__ == "__main__":
    main()

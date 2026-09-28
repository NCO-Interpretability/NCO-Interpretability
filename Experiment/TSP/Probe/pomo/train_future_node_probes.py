import argparse
import csv
import glob
import importlib
import importlib.util
import math
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DISTRIBUTIONS = [
    "Clustered", "Expansion", "Explosion", "Grid",
    "Implosion", "Mixed", "Uniform",
]
DISTRIBUTION_CODE = {name: i for i, name in enumerate(DISTRIBUTIONS)}
LAYER_NAMES = [
    "encoder_layer_0", "encoder_layer_1", "encoder_layer_2",
    "encoder_layer_3", "encoder_layer_4", "encoder_layer_5",
    "decoder_layer_0",
]


def install_numpy_pickle_compatibility_patch() -> None:
    """Allow loading pickles written by different NumPy versions."""
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


@dataclass
class InstanceRecord:
    coords: np.ndarray
    n: int
    distribution: str
    distribution_code: int
    instance_id: int
    split: str
    source_file: str


@dataclass
class MetricState:
    count: int = 0
    top1_sum: float = 0.0
    top5_sum: float = 0.0
    top10_sum: float = 0.0
    mrr_sum: float = 0.0
    rank_sum: float = 0.0
    rank_percentile_sum: float = 0.0

    def update_ranks(self, ranks: torch.Tensor, num_candidates: int) -> None:
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
        if a > 1:
            self.rank_percentile_sum += float(np.sum((r - 1.0) / (a - 1.0)))

    def update_random_expected_many(self, num_candidates: int, count: int) -> None:
        if count <= 0:
            return
        a = max(int(num_candidates), 1)
        c = int(count)
        self.count += c
        self.top1_sum += c / a
        self.top5_sum += c * min(5, a) / a
        self.top10_sum += c * min(10, a) / a
        harmonic = sum(1.0 / r for r in range(1, a + 1))
        self.mrr_sum += c * harmonic / a
        self.rank_sum += c * (a + 1) / 2.0
        self.rank_percentile_sum += 0.0 if a <= 1 else c * 0.5

    def to_row(self) -> Dict:
        if self.count == 0:
            return {
                "num_samples": 0, "top1": float("nan"),
                "top5": float("nan"), "top10": float("nan"),
                "mrr": float("nan"), "mean_rank": float("nan"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train POMO future-node planning probes, horizons 1..10."
    )
    parser.add_argument(
        "--inference_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing plot-friendly POMO result files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory for probe checkpoints and metric files.",
    )
    parser.add_argument("--run_tag", type=str, default="horizons_1_10")
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to the instrumented POMO TSP model Python file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the pretrained POMO checkpoint.",
    )
    parser.add_argument("--nodes", type=str, default="20,50,100,200,500,1000")
    parser.add_argument("--layers", type=str, default=",".join(LAYER_NAMES))
    parser.add_argument("--horizons", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument(
        "--max_instances_per_distribution_by_n", type=str,
        default="20:2000,50:1000,100:1000,200:500,500:200,1000:100",
    )
    parser.add_argument(
        "--batch_size_by_n", type=str,
        default="20:128,50:128,100:64,200:16,500:4,1000:1",
    )
    parser.add_argument("--fallback_batch_size", type=int, default=8)
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
        horizons = [
            int(item.strip()) for item in args.horizons.split(",") if item.strip()
        ]
    except ValueError as exc:
        parser.error(f"Invalid integer list: {exc}")
    layers = [item.strip() for item in args.layers.split(",") if item.strip()]
    if not nodes or any(value <= 1 for value in nodes):
        parser.error("--nodes must contain integers greater than 1.")
    if not layers:
        parser.error("--layers cannot be empty.")
    unknown_layers = sorted(set(layers) - set(LAYER_NAMES))
    if unknown_layers:
        parser.error(f"Unknown layer names: {unknown_layers}")
    if not horizons or any(value <= 0 for value in horizons):
        parser.error("--horizons must contain positive integers.")
    return args


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_string_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_map(text: str) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split(":")
        result[int(key.strip())] = int(value.strip())
    return result


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


def to_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def normalize_coords(coords) -> np.ndarray:
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
    mapping = {
        "cluster": "Clustered", "clustered": "Clustered",
        "expansion": "Expansion", "explosion": "Explosion",
        "grid": "Grid", "implosion": "Implosion",
        "mixed": "Mixed", "uniform": "Uniform",
    }
    return mapping.get(str(name).strip().lower())


def parse_distribution_and_n_from_filename(path: str) -> Tuple[Optional[str], Optional[int]]:
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(
        r"(clustered|cluster|expansion|explosion|grid|implosion|mixed|uniform)[_-]?(\d+)",
        stem, re.IGNORECASE,
    )
    if match is None:
        return None, None
    return canonical_distribution_name(match.group(1)), int(match.group(2))


def select_step_indices(n: int, stride: int, max_steps: int) -> set:
    steps = list(range(0, n - 1, max(1, stride)))
    if max_steps > 0 and len(steps) > max_steps:
        indices = np.unique(np.round(np.linspace(0, len(steps) - 1, max_steps)).astype(int))
        steps = [steps[i] for i in indices]
    return set(steps)


def make_batches(records: List[InstanceRecord], batch_size: int, shuffle: bool, seed: int) -> Iterable[List[InstanceRecord]]:
    items = list(records)
    if shuffle:
        random.Random(seed).shuffle(items)
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def records_to_coords_batch(records: List[InstanceRecord], device: torch.device) -> torch.Tensor:
    coords = np.stack([r.coords for r in records], axis=0).astype(np.float32)
    return torch.tensor(coords, dtype=torch.float32, device=device)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def tagged_csv_path(directory: str, filename: str, run_tag: str) -> str:
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}_{run_tag}{ext}")


def layer_metadata(layer_name: str) -> Tuple[str, int]:
    if layer_name.startswith("encoder_layer_"):
        return "encoder", int(layer_name.rsplit("_", 1)[1])
    if layer_name.startswith("decoder_layer_"):
        return "decoder", int(layer_name.rsplit("_", 1)[1])
    return "baseline", -1


# ============================================================
# Dataset loading
# ============================================================

def load_coords_from_plotfriendly_pkl(path: str) -> List[np.ndarray]:
    data = safe_pickle_load(path)
    keys = ["coords", "coordinates", "problem", "problems", "nodes", "data"]

    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, dict):
                key = first_existing_key(item, keys)
                if key is None:
                    raise ValueError(f"No coordinate key in {path}")
                result.append(normalize_coords(item[key]))
            elif isinstance(item, (tuple, list)) and len(item) > 0:
                result.append(normalize_coords(item[0]))
            else:
                result.append(normalize_coords(item))
        return result

    if isinstance(data, dict):
        key = first_existing_key(data, keys)
        if key is None:
            raise ValueError(f"No coordinate key in {path}")
        arr = to_numpy(data[key])
    elif isinstance(data, tuple):
        arr = to_numpy(data[0])
    else:
        arr = to_numpy(data)

    if arr.ndim == 3:
        return [normalize_coords(arr[i]) for i in range(arr.shape[0])]
    return [normalize_coords(arr)]


def build_instance_records(
    inference_dirs: List[str], target_nodes: List[int], max_instances_by_n: Dict[int, int],
    train_ratio: float, val_ratio: float,
) -> List[InstanceRecord]:
    pkl_files: List[str] = []
    for directory in inference_dirs:
        pkl_files.extend(sorted(glob.glob(os.path.join(directory, "*.pkl"))))

    records: List[InstanceRecord] = []
    for path in pkl_files:
        distribution, n = parse_distribution_and_n_from_filename(path)
        if distribution is None or n is None:
            print(f"WARNING: could not parse distribution/N from {path}")
            continue
        if n not in target_nodes:
            continue

        coords_list = load_coords_from_plotfriendly_pkl(path)
        limit = max_instances_by_n.get(n, -1)
        if limit >= 0:
            coords_list = coords_list[:limit]

        num_items = len(coords_list)
        n_train = int(num_items * train_ratio)
        n_val = int(num_items * val_ratio)

        for instance_id, coords in enumerate(coords_list):
            if instance_id < n_train:
                split = "train"
            elif instance_id < n_train + n_val:
                split = "validation"
            else:
                split = "test"
            records.append(InstanceRecord(
                coords=coords, n=n, distribution=distribution,
                distribution_code=DISTRIBUTION_CODE[distribution],
                instance_id=instance_id, split=split, source_file=path,
            ))
    return records


def records_to_manifest_rows(records: List[InstanceRecord]) -> List[Dict]:
    return [{
        "n": r.n,
        "distribution": r.distribution,
        "distribution_code": r.distribution_code,
        "instance_id": r.instance_id,
        "split": r.split,
        "source_file": r.source_file,
        "split_rule": "sequential_first_70_next_15_last_15_within_each_source_file",
    } for r in records]


# ============================================================
# POMO model loading
# ============================================================

def import_pomo_model(model_file: str):
    """Import the instrumented POMO model from an explicit file path."""
    model_path = os.path.abspath(os.path.expanduser(model_file))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"POMO model file was not found: {model_path}")

    module_name = "instrumented_pomo_tsp_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from: {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "TSPModel"):
        raise AttributeError(f"TSPModel was not found in: {model_path}")
    return module.TSPModel


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint) -> Dict:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def build_pomo_model(args: argparse.Namespace, device: torch.device):
    Model = import_pomo_model(args.model_file)
    params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    model = Model(**params)
    model.load_state_dict(extract_state_dict(load_checkpoint(args.checkpoint_path, device)), strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

# ============================================================
# POMO representation extraction and greedy rollout
# ============================================================

def reshape_by_heads(tensor: torch.Tensor, head_num: int) -> torch.Tensor:
    """Convert [B, L, H*D] to [B, H, L, D]."""
    batch_size, length, _ = tensor.shape
    return tensor.reshape(batch_size, length, head_num, -1).transpose(1, 2)


def gather_node_representations(node_reps: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    """Gather node representations and return [B, K, D]."""
    batch_size, pick_count = node_indices.shape
    embedding_dim = node_reps.size(2)
    gather_indices = node_indices[:, :, None].expand(batch_size, pick_count, embedding_dim)
    return node_reps.gather(dim=1, index=gather_indices.long())


def get_available_nodes(selected_node_list: torch.Tensor, problem_size: int) -> torch.Tensor:
    """Return currently unvisited node IDs as [B, A]."""
    batch_size, selected_count = selected_node_list.shape
    device = selected_node_list.device
    all_nodes = torch.arange(problem_size, device=device)[None, :].expand(batch_size, -1)
    visited = torch.zeros((batch_size, problem_size), dtype=torch.bool, device=device)
    visited.scatter_(1, selected_node_list.long(), True)
    return all_nodes[~visited].view(batch_size, problem_size - selected_count)


def build_visited_ninf_mask(
    selected_node_list: torch.Tensor, problem_size: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B, N] mask with negative infinity at visited nodes."""
    mask = torch.zeros(
        (selected_node_list.size(0), problem_size),
        dtype=dtype,
        device=selected_node_list.device,
    )
    mask.scatter_(1, selected_node_list.long(), float("-inf"))
    return mask


def encode_pomo_with_layer_reps(
    model, coords_batch: torch.Tensor, layer_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Run the six-layer POMO encoder and collect requested layer outputs."""
    requested = {
        int(name.rsplit("_", 1)[1])
        for name in layer_names
        if name.startswith("encoder_layer_")
    }
    out = model.encoder.embedding(coords_batch)
    layer_reps: Dict[str, torch.Tensor] = {}
    for layer_index, layer in enumerate(model.encoder.layers):
        out = layer(out)
        if layer_index in requested:
            layer_reps[f"encoder_layer_{layer_index}"] = out
    return layer_reps, out


def initialize_pomo_decoder(model, final_encoded_nodes: torch.Tensor) -> None:
    """Cache decoder keys, values, and the first-node query for start node 0."""
    batch_size = final_encoded_nodes.size(0)
    start_nodes = torch.zeros(batch_size, 1, dtype=torch.long, device=final_encoded_nodes.device)
    model.decoder.set_kv(final_encoded_nodes)
    model.decoder.set_q1(gather_node_representations(final_encoded_nodes, start_nodes))


def manual_pomo_decoder_forward(
    model, final_encoded_nodes: torch.Tensor, selected_node_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one greedy POMO decoder step and expose decoder_layer_0.

    Returns:
        selected_next: [B]
        available_nodes: [B, A]
        decoder_candidate_reps: [B, A, 128]
        available_logits: [B, A]
    """
    decoder = model.decoder
    batch_size, problem_size, _ = final_encoded_nodes.shape
    available_nodes = get_available_nodes(selected_node_list, problem_size)

    current_nodes = selected_node_list[:, -1:]
    encoded_last_node = gather_node_representations(final_encoded_nodes, current_nodes)

    head_num = model.model_params["head_num"]
    qkv_dim = model.model_params["qkv_dim"]
    q_last = reshape_by_heads(decoder.Wq_last(encoded_last_node), head_num)
    q = decoder.q_first + q_last

    attention_scores = torch.matmul(q, decoder.k.transpose(2, 3)) / math.sqrt(qkv_dim)
    ninf_mask = build_visited_ninf_mask(
        selected_node_list, problem_size, attention_scores.dtype
    )
    attention_scores = attention_scores + ninf_mask[:, None, None, :]
    attention_weights = F.softmax(attention_scores, dim=3)

    head_outputs = torch.matmul(attention_weights, decoder.v)
    out_concat = head_outputs.transpose(1, 2).reshape(
        batch_size, 1, head_num * qkv_dim
    )
    decoder_context = decoder.multi_head_combine(out_concat).squeeze(1)

    raw_scores = torch.matmul(
        decoder_context[:, None, :], decoder.single_head_key
    ).squeeze(1)
    scaled_scores = raw_scores / model.model_params["sqrt_embedding_dim"]
    clipped_scores = model.model_params["logit_clipping"] * torch.tanh(scaled_scores)
    masked_logits = clipped_scores + ninf_mask
    selected_next = masked_logits.argmax(dim=1)

    available_final_reps = gather_node_representations(
        final_encoded_nodes, available_nodes
    )
    decoder_candidate_reps = available_final_reps * decoder_context[:, None, :]
    available_logits = masked_logits.gather(1, available_nodes.long())

    return selected_next, available_nodes, decoder_candidate_reps, available_logits


def rollout_pomo_tour_batch(
    model, coords_batch: torch.Tensor, layer_names: List[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Run the frozen greedy POMO rollout with pomo_size=1 and start node 0."""
    batch_size, n, _ = coords_batch.shape
    encoder_layer_reps, final_encoded_nodes = encode_pomo_with_layer_reps(
        model, coords_batch, layer_names
    )
    initialize_pomo_decoder(model, final_encoded_nodes)
    tour = torch.zeros((batch_size, 1), dtype=torch.long, device=coords_batch.device)

    for _ in range(n - 1):
        selected_next, _, _, _ = manual_pomo_decoder_forward(
            model, final_encoded_nodes, tour
        )
        tour = torch.cat([tour, selected_next[:, None]], dim=1)

    return tour, encoder_layer_reps, final_encoded_nodes


def collect_step_layer_reps(
    model,
    encoder_layer_reps: Dict[str, torch.Tensor],
    final_encoded_nodes: torch.Tensor,
    selected_node_list: torch.Tensor,
    layer_names: List[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Collect available-candidate representations at one decoding step."""
    _, available_nodes, decoder_reps, available_logits = manual_pomo_decoder_forward(
        model, final_encoded_nodes, selected_node_list
    )
    result: Dict[str, torch.Tensor] = {}
    for layer_name in layer_names:
        if layer_name.startswith("encoder_layer_"):
            result[layer_name] = gather_node_representations(
                encoder_layer_reps[layer_name], available_nodes
            )
        elif layer_name == "decoder_layer_0":
            result[layer_name] = decoder_reps
        else:
            raise ValueError(f"Unsupported layer name: {layer_name}")
    return available_nodes, result, available_logits


# ============================================================
# Probe models and features
# ============================================================

class CandidateLinearProbe(nn.Module):
    """Linear candidate ranker: [B, A, D] -> [B, A]."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x).squeeze(-1)


class FutureProbeBank(nn.Module):
    """All representation probes and coordinate probes for one problem size."""

    def __init__(
        self, layer_names: List[str], horizons: List[int],
        representation_dim: int, coord_dim: int,
    ):
        super().__init__()
        self.layer_names = list(layer_names)
        self.horizons = list(horizons)
        self.representation_probes = nn.ModuleDict({
            self.rep_key(layer_name, horizon): CandidateLinearProbe(representation_dim)
            for layer_name in layer_names
            for horizon in horizons
        })
        self.coord_probes = nn.ModuleDict({
            self.coord_key(horizon): CandidateLinearProbe(coord_dim)
            for horizon in horizons
        })

    @staticmethod
    def rep_key(layer_name: str, horizon: int) -> str:
        return f"{layer_name}__horizon_{horizon}"

    @staticmethod
    def coord_key(horizon: int) -> str:
        return f"horizon_{horizon}"

    def get_representation_probe(self, layer_name: str, horizon: int) -> CandidateLinearProbe:
        return self.representation_probes[self.rep_key(layer_name, horizon)]

    def get_coord_probe(self, horizon: int) -> CandidateLinearProbe:
        return self.coord_probes[self.coord_key(horizon)]


def build_coord_features_batch(
    coords_batch: torch.Tensor,
    available_nodes: torch.Tensor,
    start_nodes: torch.Tensor,
    current_nodes: torch.Tensor,
    step_idx: int,
    n: int,
) -> torch.Tensor:
    """Build the same 14-dimensional coordinate baseline used for LEHD."""
    batch_size, candidate_count = available_nodes.shape
    batch_indices = torch.arange(batch_size, device=coords_batch.device)
    candidate_xy = coords_batch[batch_indices[:, None], available_nodes.long()]
    current_xy = coords_batch[batch_indices, current_nodes.long()][:, None, :].expand_as(candidate_xy)
    start_xy = coords_batch[batch_indices, start_nodes.long()][:, None, :].expand_as(candidate_xy)
    cand_minus_current = candidate_xy - current_xy
    cand_minus_start = candidate_xy - start_xy
    dist_current = torch.norm(cand_minus_current, dim=2, keepdim=True)
    dist_start = torch.norm(cand_minus_start, dim=2, keepdim=True)
    step_fraction = torch.full(
        (batch_size, candidate_count, 1),
        float(step_idx) / float(max(n - 1, 1)),
        dtype=coords_batch.dtype,
        device=coords_batch.device,
    )
    remaining_fraction = torch.full(
        (batch_size, candidate_count, 1),
        float(candidate_count) / float(max(n, 1)),
        dtype=coords_batch.dtype,
        device=coords_batch.device,
    )
    return torch.cat([
        candidate_xy, current_xy, start_xy,
        cand_minus_current, cand_minus_start,
        dist_current, dist_start,
        step_fraction, remaining_fraction,
    ], dim=2)


def get_label_indices_batched(
    available_nodes: torch.Tensor, target_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locate each future target node inside the current candidate set."""
    matches = available_nodes.long().eq(target_nodes.long()[:, None])
    valid = matches.any(dim=1)
    labels = matches.float().argmax(dim=1).long()
    return labels, valid


def compute_ranks_batched(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target_scores = scores.gather(1, labels[:, None]).squeeze(1)
    return ((scores > target_scores[:, None]).sum(dim=1) + 1).long()


def dense_ranks_from_scores_batched(scores: torch.Tensor, descending: bool) -> torch.Tensor:
    batch_size, candidate_count = scores.shape
    order = torch.argsort(scores, dim=1, descending=descending)
    ranks = torch.empty_like(order)
    values = torch.arange(1, candidate_count + 1, device=scores.device)[None, :].expand(batch_size, -1)
    ranks.scatter_(1, order, values)
    return ranks.long()


def score_candidates_by_rank_distance(candidate_ranks: torch.Tensor, horizon: int) -> torch.Tensor:
    return -torch.abs(candidate_ranks.float() - float(horizon))


def nearest_neighbor_rollout_positions_batched(
    coords_batch: torch.Tensor,
    available_nodes: torch.Tensor,
    current_nodes: torch.Tensor,
) -> torch.Tensor:
    """Return each candidate's position in a greedy nearest-neighbor rollout."""
    device = coords_batch.device
    batch_size, candidate_count = available_nodes.shape
    batch_indices = torch.arange(batch_size, device=device)
    available_xy = coords_batch[batch_indices[:, None], available_nodes.long()]
    current_xy = coords_batch[batch_indices, current_nodes.long()]
    remaining = torch.ones((batch_size, candidate_count), dtype=torch.bool, device=device)
    positions = torch.empty((batch_size, candidate_count), dtype=torch.long, device=device)

    for position in range(1, candidate_count + 1):
        distances = torch.norm(available_xy - current_xy[:, None, :], dim=2)
        distances = distances.masked_fill(~remaining, float("inf"))
        chosen = distances.argmin(dim=1)
        positions.scatter_(1, chosen[:, None], torch.full(
            (batch_size, 1), position, dtype=torch.long, device=device
        ))
        remaining.scatter_(1, chosen[:, None], False)
        current_xy = available_xy[batch_indices, chosen]
    return positions

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
    layer_names: List[str],
    horizons: List[int],
    device: torch.device,
    batch_size: int,
    train_step_stride: int,
    max_train_steps_per_instance: int,
    grad_clip_norm: float,
    seed: int,
) -> Dict:
    """Train all POMO probes for one problem size and one epoch."""
    probe_bank.train()
    total_loss = 0.0
    total_updates = 0
    total_tasks = 0
    total_instances = 0
    selected_steps = select_step_indices(
        n, train_step_stride, max_train_steps_per_instance
    )

    for batch_index, batch_records in enumerate(make_batches(
        train_records, batch_size, True, seed + 1000 * epoch + n
    )):
        coords_batch = records_to_coords_batch(batch_records, device)
        current_batch_size = coords_batch.size(0)

        # POMO is frozen. Its own greedy rollout provides future-node labels.
        with torch.no_grad():
            tour_batch, encoder_reps, final_encoded = rollout_pomo_tour_batch(
                model, coords_batch, layer_names
            )

        for step_idx in range(n - 1):
            if step_idx not in selected_steps:
                continue

            selected_node_list = tour_batch[:, :step_idx + 1]
            with torch.no_grad():
                available_nodes, layer_reps, _ = collect_step_layer_reps(
                    model, encoder_reps, final_encoded,
                    selected_node_list, layer_names,
                )

            start_nodes = tour_batch[:, 0]
            current_nodes = tour_batch[:, step_idx]
            coord_features = build_coord_features_batch(
                coords_batch, available_nodes, start_nodes,
                current_nodes, step_idx, n,
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

                coord_scores = probe_bank.get_coord_probe(horizon)(coord_features)
                step_losses.append(F.cross_entropy(coord_scores[valid], labels_valid))

                for layer_name in layer_names:
                    scores = probe_bank.get_representation_probe(
                        layer_name, horizon
                    )(layer_reps[layer_name])
                    step_losses.append(F.cross_entropy(scores[valid], labels_valid))

            if step_losses:
                loss = torch.stack(step_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        probe_bank.parameters(), grad_clip_norm
                    )
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_updates += 1
                total_tasks += len(step_losses) * current_batch_size

        total_instances += current_batch_size
        if (batch_index + 1) % 10 == 0:
            print(
                f"    [N={n} epoch={epoch}] batches={batch_index + 1} | "
                f"instances={total_instances}/{len(train_records)} | "
                f"avg_loss={total_loss / max(total_updates, 1):.6f} | "
                f"updates={total_updates}"
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

def update_score_metric(
    metrics: Dict,
    model_type: str,
    layer_name: str,
    horizon: int,
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    if valid.sum().item() == 0:
        return
    key = (model_type, layer_name, int(horizon))
    if key not in metrics:
        metrics[key] = MetricState()
    ranks = compute_ranks_batched(scores[valid], labels[valid])
    metrics[key].update_ranks(ranks, scores.size(1))


def update_random_metric(
    metrics: Dict, horizon: int, num_candidates: int, valid_count: int,
) -> None:
    key = ("random_expected", "baseline", int(horizon))
    if key not in metrics:
        metrics[key] = MetricState()
    metrics[key].update_random_expected_many(num_candidates, valid_count)


def evaluate_split_for_n(
    n: int,
    split_name: str,
    epoch: int,
    model,
    probe_bank: FutureProbeBank,
    records: List[InstanceRecord],
    layer_names: List[str],
    horizons: List[int],
    device: torch.device,
    batch_size: int,
    eval_step_stride: int,
    max_eval_steps_per_instance: int,
) -> List[Dict]:
    """Evaluate probes and all baselines on one split."""
    if not records:
        return []

    probe_bank.eval()
    metrics: Dict[Tuple[str, str, int], MetricState] = {}
    selected_steps = select_step_indices(
        n, eval_step_stride, max_eval_steps_per_instance
    )

    with torch.no_grad():
        for batch_index, batch_records in enumerate(make_batches(
            records, batch_size, False, 0
        )):
            coords_batch = records_to_coords_batch(batch_records, device)
            bsz = coords_batch.size(0)
            tour_batch, encoder_reps, final_encoded = rollout_pomo_tour_batch(
                model, coords_batch, layer_names
            )

            for step_idx in range(n - 1):
                if step_idx not in selected_steps:
                    continue

                selected_node_list = tour_batch[:, :step_idx + 1]
                available_nodes, layer_reps, available_logits = collect_step_layer_reps(
                    model, encoder_reps, final_encoded,
                    selected_node_list, layer_names,
                )

                start_nodes = tour_batch[:, 0]
                current_nodes = tour_batch[:, step_idx]
                coord_features = build_coord_features_batch(
                    coords_batch, available_nodes, start_nodes,
                    current_nodes, step_idx, n,
                )

                batch_indices = torch.arange(bsz, device=device)
                available_xy = coords_batch[
                    batch_indices[:, None], available_nodes.long()
                ]
                current_xy = coords_batch[
                    batch_indices, current_nodes.long()
                ][:, None, :]
                start_xy = coords_batch[
                    batch_indices, start_nodes.long()
                ][:, None, :]

                dist_current = torch.norm(available_xy - current_xy, dim=2)
                dist_start = torch.norm(available_xy - start_xy, dim=2)
                nearest_current_static = -dist_current
                nearest_start_static = -dist_start
                pomo_current_logits_static = available_logits

                current_distance_ranks = dense_ranks_from_scores_batched(
                    dist_current, descending=False
                )
                start_distance_ranks = dense_ranks_from_scores_batched(
                    dist_start, descending=False
                )
                pomo_logit_ranks = dense_ranks_from_scores_batched(
                    available_logits, descending=True
                )
                nn_positions = nearest_neighbor_rollout_positions_batched(
                    coords_batch, available_nodes, current_nodes
                )

                for horizon in horizons:
                    future_step = step_idx + horizon
                    if future_step >= n:
                        continue

                    target_nodes = tour_batch[:, future_step]
                    labels, valid = get_label_indices_batched(
                        available_nodes, target_nodes
                    )
                    valid_count = int(valid.sum().item())
                    if valid_count == 0:
                        continue

                    candidate_count = available_nodes.size(1)
                    update_random_metric(
                        metrics, horizon, candidate_count, valid_count
                    )

                    update_score_metric(
                        metrics, "nearest_current_static", "baseline", horizon,
                        nearest_current_static, labels, valid,
                    )
                    update_score_metric(
                        metrics, "nearest_start_static", "baseline", horizon,
                        nearest_start_static, labels, valid,
                    )
                    update_score_metric(
                        metrics, "pomo_current_logits_static", "baseline", horizon,
                        pomo_current_logits_static, labels, valid,
                    )

                    update_score_metric(
                        metrics, "nearest_current_order_h", "baseline", horizon,
                        score_candidates_by_rank_distance(current_distance_ranks, horizon),
                        labels, valid,
                    )
                    update_score_metric(
                        metrics, "nearest_start_order_h", "baseline", horizon,
                        score_candidates_by_rank_distance(start_distance_ranks, horizon),
                        labels, valid,
                    )
                    update_score_metric(
                        metrics, "pomo_logit_order_h", "baseline", horizon,
                        score_candidates_by_rank_distance(pomo_logit_ranks, horizon),
                        labels, valid,
                    )
                    update_score_metric(
                        metrics, "nearest_neighbor_rollout_h", "baseline", horizon,
                        score_candidates_by_rank_distance(nn_positions, horizon),
                        labels, valid,
                    )

                    coord_scores = probe_bank.get_coord_probe(horizon)(coord_features)
                    update_score_metric(
                        metrics, "coord_linear_probe", "coordinate_baseline", horizon,
                        coord_scores, labels, valid,
                    )

                    for layer_name in layer_names:
                        scores = probe_bank.get_representation_probe(
                            layer_name, horizon
                        )(layer_reps[layer_name])
                        update_score_metric(
                            metrics, "representation_probe", layer_name, horizon,
                            scores, labels, valid,
                        )

            if (batch_index + 1) % 10 == 0:
                completed = min((batch_index + 1) * batch_size, len(records))
                print(
                    f"    [N={n} split={split_name}] batches={batch_index + 1} | "
                    f"instances={completed}/{len(records)}"
                )

    rows = []
    for (model_type, layer_name, horizon), state in sorted(
        metrics.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
    ):
        layer_type, layer_index = layer_metadata(layer_name)
        row = {
            "n": n,
            "split": split_name,
            "epoch": epoch,
            "model_type": model_type,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "layer_index": layer_index,
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
    layer_names: List[str],
    horizons: List[int],
    representation_dim: int,
    coord_dim: int,
    run_tag: str,
) -> None:
    """Save combined and per-layer/per-horizon probe checkpoints."""
    n_root = os.path.join(output_root, f"N_{n}")
    combined_dir = os.path.join(n_root, "checkpoints")
    ensure_dir(combined_dir)
    safe_tag = run_tag.replace("/", "_").replace(" ", "_")

    payload = {
        "architecture": "POMO",
        "n": n,
        "epoch": epoch,
        "layer_names": layer_names,
        "horizons": horizons,
        "representation_dim": representation_dim,
        "coord_dim": coord_dim,
        "run_tag": safe_tag,
        "decoder_layer_0_definition": (
            "elementwise_product_of_dynamic_decoder_context_"
            "and_final_encoder_candidate"
        ),
        "probe_bank_state_dict": probe_bank.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(
        payload,
        os.path.join(combined_dir, f"all_probes_{safe_tag}_epoch_{epoch:03d}.pt"),
    )
    torch.save(
        payload,
        os.path.join(combined_dir, f"all_probes_{safe_tag}_latest.pt"),
    )

    for layer_name in layer_names:
        layer_type, layer_index = layer_metadata(layer_name)
        for horizon in horizons:
            checkpoint_dir = os.path.join(
                n_root, layer_name, f"horizon_{horizon}", "checkpoints"
            )
            ensure_dir(checkpoint_dir)
            probe = probe_bank.get_representation_probe(layer_name, horizon)
            torch.save({
                "architecture": "POMO",
                "n": n,
                "epoch": epoch,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "layer_index": layer_index,
                "horizon": horizon,
                "input_dim": representation_dim,
                "probe_type": "linear_candidate_ranker",
                "decoder_layer_0_definition": (
                    "elementwise_product_of_dynamic_decoder_context_"
                    "and_final_encoder_candidate"
                ),
                "model_state_dict": probe.state_dict(),
            }, os.path.join(checkpoint_dir, "future_node_probe.pt"))

    feature_names = [
        "candidate_x", "candidate_y", "current_x", "current_y",
        "start_x", "start_y",
        "candidate_minus_current_x", "candidate_minus_current_y",
        "candidate_minus_start_x", "candidate_minus_start_y",
        "dist_candidate_current", "dist_candidate_start",
        "step_fraction", "remaining_fraction",
    ]
    for horizon in horizons:
        checkpoint_dir = os.path.join(
            n_root, "coordinate_baseline", f"horizon_{horizon}", "checkpoints"
        )
        ensure_dir(checkpoint_dir)
        probe = probe_bank.get_coord_probe(horizon)
        torch.save({
            "architecture": "POMO",
            "n": n,
            "epoch": epoch,
            "horizon": horizon,
            "input_dim": coord_dim,
            "probe_type": "coordinate_linear_candidate_ranker",
            "feature_names": feature_names,
            "model_state_dict": probe.state_dict(),
        }, os.path.join(checkpoint_dir, "coord_linear_probe.pt"))

# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    nodes = parse_int_list(args.nodes)
    layer_names = parse_string_list(args.layers)
    horizons = parse_int_list(args.horizons)
    max_instances_by_n = parse_int_map(args.max_instances_per_distribution_by_n)
    batch_size_by_n = parse_int_map(args.batch_size_by_n)
    inference_dirs = [os.path.abspath(os.path.expanduser(x)) for x in args.inference_dirs]

    invalid_layers = [name for name in layer_names if name not in set(LAYER_NAMES)]
    if invalid_layers:
        raise ValueError(
            f"Invalid layer names: {invalid_layers}. Valid names: {LAYER_NAMES}"
        )
    invalid_horizons = [h for h in horizons if h < 1 or h > 10]
    if invalid_horizons:
        raise ValueError(
            f"This script supports horizons 1..10 only. Invalid: {invalid_horizons}"
        )

    ensure_dir(args.output_root)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device_num)
        device = torch.device(f"cuda:{args.cuda_device_num}")
    else:
        device = torch.device("cpu")

    print("=" * 90)
    print("POMO future-node planning probes, horizons 1..10")
    print("Output root:", args.output_root)
    print("Inference directories:", inference_dirs)
    print("Nodes:", nodes)
    print("Layers:", layer_names)
    print("Horizons:", horizons)
    print("Batch sizes:", batch_size_by_n)
    print("Device:", device)
    print("=" * 90)

    print("Loading coordinates and constructing sequential splits...")
    all_records = build_instance_records(
        inference_dirs=inference_dirs,
        target_nodes=nodes,
        max_instances_by_n=max_instances_by_n,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Total records loaded: {len(all_records)}")
    if not all_records:
        raise RuntimeError(
            "No records were loaded. Check --inference_dirs and file naming."
        )

    manifest_rows = records_to_manifest_rows(all_records)
    write_csv(
        tagged_csv_path(args.output_root, "split_manifest.csv", args.run_tag),
        manifest_rows,
    )
    write_csv(
        tagged_csv_path(args.output_root, "test_manifest.csv", args.run_tag),
        [row for row in manifest_rows if row["split"] == "test"],
    )

    print("Loading frozen POMO checkpoint...")
    pomo_model = build_pomo_model(args, device)
    encoder_layer_count = len(pomo_model.encoder.layers)
    if encoder_layer_count != 6:
        raise ValueError(
            f"Expected six POMO encoder layers, found {encoder_layer_count}."
        )
    print("POMO loaded: encoder layers 0..5 and decoder layer 0.")

    representation_dim = int(pomo_model.model_params["embedding_dim"])
    coord_dim = 14

    # Remove old global tagged files to prevent duplicate rows after reruns.
    global_files = [
        tagged_csv_path(args.output_root, "all_train_log.csv", args.run_tag),
        tagged_csv_path(args.output_root, "all_validation_metrics.csv", args.run_tag),
        tagged_csv_path(args.output_root, "all_test_metrics.csv", args.run_tag),
        tagged_csv_path(args.output_root, "all_train_metrics_final.csv", args.run_tag),
    ]
    for path in global_files:
        if os.path.exists(path):
            os.remove(path)

    for n in nodes:
        print("\n" + "=" * 90)
        print(f"Starting N={n}")
        print("=" * 90)

        active_horizons = [h for h in horizons if h < n]
        if not active_horizons:
            print(f"WARNING: no valid horizons for N={n}; skipping.")
            continue

        batch_size = batch_size_by_n.get(n, args.fallback_batch_size)
        n_records = [r for r in all_records if r.n == n]
        train_records = [r for r in n_records if r.split == "train"]
        val_records = [r for r in n_records if r.split == "validation"]
        test_records = [r for r in n_records if r.split == "test"]

        print(
            f"batch_size={batch_size} | train={len(train_records)} | "
            f"validation={len(val_records)} | test={len(test_records)}"
        )
        if not train_records:
            print(f"WARNING: no train records for N={n}; skipping.")
            continue

        n_root = os.path.join(args.output_root, f"N_{n}")
        ensure_dir(n_root)
        n_manifest = records_to_manifest_rows(n_records)
        write_csv(
            tagged_csv_path(n_root, "split_manifest.csv", args.run_tag),
            n_manifest,
        )
        write_csv(
            tagged_csv_path(n_root, "test_manifest.csv", args.run_tag),
            [row for row in n_manifest if row["split"] == "test"],
        )

        probe_bank = FutureProbeBank(
            layer_names=layer_names,
            horizons=active_horizons,
            representation_dim=representation_dim,
            coord_dim=coord_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            probe_bank.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        train_log_rows = []
        for epoch in range(1, args.epochs + 1):
            print("\n" + "-" * 90)
            print(f"Training N={n}, epoch={epoch}/{args.epochs}")
            start_time = time.time()

            train_log = train_one_epoch_for_n(
                n=n,
                epoch=epoch,
                model=pomo_model,
                probe_bank=probe_bank,
                optimizer=optimizer,
                train_records=train_records,
                layer_names=layer_names,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                train_step_stride=args.train_step_stride,
                max_train_steps_per_instance=args.max_train_steps_per_instance,
                grad_clip_norm=args.grad_clip_norm,
                seed=args.seed,
            )
            train_log["elapsed_seconds"] = time.time() - start_time
            train_log["run_tag"] = args.run_tag
            train_log["active_horizons"] = ",".join(map(str, active_horizons))
            train_log["layer_names"] = ",".join(layer_names)
            train_log_rows.append(train_log)

            write_csv(
                tagged_csv_path(n_root, "train_log.csv", args.run_tag),
                train_log_rows,
            )
            append_csv(
                tagged_csv_path(args.output_root, "all_train_log.csv", args.run_tag),
                [train_log],
            )
            print(
                f"Finished epoch={epoch}: avg_loss={train_log['train_avg_loss']:.6f}, "
                f"updates={train_log['train_updates']}, "
                f"elapsed={train_log['elapsed_seconds']:.1f}s"
            )

            print(f"Evaluating validation split for N={n}, epoch={epoch}...")
            val_rows = evaluate_split_for_n(
                n=n,
                split_name="validation",
                epoch=epoch,
                model=pomo_model,
                probe_bank=probe_bank,
                records=val_records,
                layer_names=layer_names,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                eval_step_stride=args.eval_step_stride,
                max_eval_steps_per_instance=args.max_eval_steps_per_instance,
            )
            write_csv(
                tagged_csv_path(
                    n_root, f"validation_metrics_epoch_{epoch:03d}.csv", args.run_tag
                ),
                val_rows,
            )
            append_csv(
                tagged_csv_path(
                    args.output_root, "all_validation_metrics.csv", args.run_tag
                ),
                val_rows,
            )

            if args.save_every_epoch or epoch == args.epochs:
                print(f"Saving checkpoints for N={n}, epoch={epoch}...")
                save_probe_checkpoints_for_n(
                    output_root=args.output_root,
                    n=n,
                    epoch=epoch,
                    probe_bank=probe_bank,
                    optimizer=optimizer,
                    layer_names=layer_names,
                    horizons=active_horizons,
                    representation_dim=representation_dim,
                    coord_dim=coord_dim,
                    run_tag=args.run_tag,
                )

        if args.eval_train_split:
            print(f"Evaluating final train split for N={n}...")
            train_rows = evaluate_split_for_n(
                n=n,
                split_name="train",
                epoch=args.epochs,
                model=pomo_model,
                probe_bank=probe_bank,
                records=train_records,
                layer_names=layer_names,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                eval_step_stride=args.eval_step_stride,
                max_eval_steps_per_instance=args.max_eval_steps_per_instance,
            )
            write_csv(
                tagged_csv_path(n_root, "train_metrics_final.csv", args.run_tag),
                train_rows,
            )
            append_csv(
                tagged_csv_path(
                    args.output_root, "all_train_metrics_final.csv", args.run_tag
                ),
                train_rows,
            )

        if not args.skip_test:
            print(f"Evaluating final test split for N={n}...")
            test_rows = evaluate_split_for_n(
                n=n,
                split_name="test",
                epoch=args.epochs,
                model=pomo_model,
                probe_bank=probe_bank,
                records=test_records,
                layer_names=layer_names,
                horizons=active_horizons,
                device=device,
                batch_size=batch_size,
                eval_step_stride=args.eval_step_stride,
                max_eval_steps_per_instance=args.max_eval_steps_per_instance,
            )
            write_csv(
                tagged_csv_path(n_root, "test_metrics.csv", args.run_tag),
                test_rows,
            )
            append_csv(
                tagged_csv_path(args.output_root, "all_test_metrics.csv", args.run_tag),
                test_rows,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("Experiment finished.")
    print("Output root:", args.output_root)
    print(
        "Global validation CSV:",
        tagged_csv_path(args.output_root, "all_validation_metrics.csv", args.run_tag),
    )
    print(
        "Global test CSV:",
        tagged_csv_path(args.output_root, "all_test_metrics.csv", args.run_tag),
    )
    print("=" * 90)


if __name__ == "__main__":
    main()

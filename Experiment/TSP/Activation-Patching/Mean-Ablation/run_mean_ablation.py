from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


# ============================================================
# Experiment configuration
# ============================================================

DEFAULT_PROBLEM_SIZES = (20, 50, 100, 200, 500)

DEFAULT_DISTRIBUTIONS = (
    "cluster",
    "expansion",
    "explosion",
    "grid",
    "implosion",
    "mixed",
    "uniform",
)

DEFAULT_BATCH_SIZE_BY_PROBLEM = {
    20: 512,
    50: 256,
    100: 128,
    200: 32,
    500: 8,
}

DISTRIBUTION_ALIASES = {
    "cluster": ("cluster", "clustered"),
    "expansion": ("expansion",),
    "explosion": ("explosion",),
    "grid": ("grid",),
    "implosion": ("implosion",),
    "mixed": ("mixed",),
    "uniform": ("uniform", "random"),
}

METRIC_NAMES = (
    "matching",
    "change_rate",
    "tv_distance",
    "clean_next_probability",
    "patched_probability_on_clean_next",
    "next_probability_difference",
    "next_probability_drop",
    "absolute_next_probability_difference",
)


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure-greedy LEHD step-local Start/Current mean ablation."
        )
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained LEHD checkpoint.",
    )

    parser.add_argument(
        "--patched_model_path",
        type=str,
        required=True,
        help=(
            "Path to the activation-patching-compatible LEHD TSPModel.py."
        ),
    )

    parser.add_argument(
        "--lehd_root",
        type=str,
        required=True,
        help=(
            "Path to the LEHD root containing the LEHD package and TSPEnv."
        ),
    )

    parser.add_argument(
        "--utils_root",
        type=str,
        required=True,
        help="Directory containing util.py or utils/util.py.",
    )

    parser.add_argument(
        "--data_roots",
        type=str,
        nargs="+",
        required=True,
        help="Directories recursively searched for evaluation PKL files.",
    )

    parser.add_argument(
        "--mean_data_roots",
        type=str,
        nargs="+",
        help=(
            "Optional separate directories for estimating mean embeddings. "
            "When omitted, data_roots are used."
        ),
    )

    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root directory where results are saved.",
    )

    parser.add_argument(
        "--problem_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROBLEM_SIZES),
        help="Problem sizes to process.",
    )

    parser.add_argument(
        "--distributions",
        type=str,
        nargs="+",
        choices=list(DEFAULT_DISTRIBUTIONS),
        default=list(DEFAULT_DISTRIBUTIONS),
        help="Distribution folders to process.",
    )

    parser.add_argument(
        "--cuda_device_num",
        type=int,
        default=0,
        help="CUDA device index.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help=(
            "Override automatic problem-size-specific batch sizes. "
            "Use 0 to keep the defaults."
        ),
    )

    parser.add_argument(
        "--max_instances_per_combination",
        type=int,
        default=-1,
        help=(
            "Maximum number of instances per problem-size/distribution pair. "
            "Use -1 for all available instances."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="LEHD model and environment mode.",
    )

    parser.add_argument(
        "--exclude_forced_last_step",
        action="store_true",
        help="Exclude the final one-candidate decision from saved metrics.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite completed result folders.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first failed combination.",
    )

    args = parser.parse_args()
    if not args.problem_sizes or any(value <= 1 for value in args.problem_sizes):
        parser.error("--problem_sizes must contain integers greater than 1.")
    if args.cuda_device_num < 0:
        parser.error("--cuda_device_num cannot be negative.")
    if args.batch_size < 0:
        parser.error("--batch_size cannot be negative; use 0 for automatic sizing.")
    if (
        args.max_instances_per_combination == 0
        or args.max_instances_per_combination < -1
    ):
        parser.error(
            "--max_instances_per_combination must be -1 or a positive integer."
        )
    return args


# ============================================================
# Dynamic imports
# ============================================================

def import_shared_utils(utils_root: Path):
    """
    Import shared utility functions from either of these layouts:

        <utils_root>/util.py

    or:

        <utils_root>/utils/util.py

    The first layout is used when --utils_root points directly to the
    directory named "utils". The second layout is used when --utils_root
    points to its parent directory.
    """
    utils_root = utils_root.expanduser().resolve()

    if not utils_root.exists():
        raise FileNotFoundError(
            f"Utils path does not exist: {utils_root}"
        )

    direct_util_file = utils_root / "util.py"
    nested_util_file = utils_root / "utils" / "util.py"

    if direct_util_file.exists():
        # utils_root points directly to the package directory:
        import_parent = utils_root.parent
        util_file = direct_util_file

    elif nested_util_file.exists():
        # utils_root points to the package parent:
        import_parent = utils_root
        util_file = nested_util_file

    else:
        raise FileNotFoundError(
            "Could not find util.py. Expected one of:\n"
            f"  {direct_util_file}\n"
            f"  {nested_util_file}"
        )

    import_parent_string = str(import_parent)

    if import_parent_string not in sys.path:
        sys.path.insert(0, import_parent_string)

    try:
        from utils.util import (
            setup_torch_device,
            move_tensor_attrs_to_device,
            load_coordinate_list,
            load_checkpoint,
            get_state_dict_from_checkpoint,
            empty_cuda_cache_if_available,
        )

    except ImportError as package_error:
        # Fall back to loading util.py directly. This also works when the
        # utils directory does not contain an __init__.py file.
        spec = importlib.util.spec_from_file_location(
            "lehd_activation_patching_shared_util",
            str(util_file),
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to create an import specification for {util_file}."
            ) from package_error

        util_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(util_module)

        required_names = (
            "setup_torch_device",
            "move_tensor_attrs_to_device",
            "load_coordinate_list",
            "load_checkpoint",
            "get_state_dict_from_checkpoint",
            "empty_cuda_cache_if_available",
        )

        missing_names = [
            name
            for name in required_names
            if not hasattr(util_module, name)
        ]

        if missing_names:
            raise AttributeError(
                f"Utility module {util_file} is missing: {missing_names}"
            )

        setup_torch_device = util_module.setup_torch_device
        move_tensor_attrs_to_device = (
            util_module.move_tensor_attrs_to_device
        )
        load_coordinate_list = util_module.load_coordinate_list
        load_checkpoint = util_module.load_checkpoint
        get_state_dict_from_checkpoint = (
            util_module.get_state_dict_from_checkpoint
        )
        empty_cuda_cache_if_available = (
            util_module.empty_cuda_cache_if_available
        )

    print(f"Loaded shared utilities from: {util_file}")

    return {
        "setup_torch_device": setup_torch_device,
        "move_tensor_attrs_to_device": move_tensor_attrs_to_device,
        "load_coordinate_list": load_coordinate_list,
        "load_checkpoint": load_checkpoint,
        "get_state_dict_from_checkpoint": get_state_dict_from_checkpoint,
        "empty_cuda_cache_if_available": empty_cuda_cache_if_available,
    }


def import_lehd_env(lehd_root: Path):
    """
    Import the original LEHD TSPEnv while keeping the patched model separate.
    """
    if not lehd_root.exists():
        raise FileNotFoundError(
            f"LEHD root does not exist: {lehd_root}"
        )

    sys.path.insert(0, str(lehd_root))

    from LEHD.TSP.TSPEnv import TSPEnv

    return TSPEnv


def import_patched_model(model_path: Path):
    """
    Import the activation-patching-compatible TSPModel from an explicit path.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Patched model file does not exist: {model_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "lehd_step_local_patched_model",
        str(model_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to import the patched model from {model_path}."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "TSPModel"):
        raise AttributeError(
            f"TSPModel was not found in {model_path}."
        )

    return module.TSPModel


# ============================================================
# Dataset discovery
# ============================================================

@dataclass(frozen=True)
class DatasetRecord:
    path: Path
    distribution: str
    problem_size_counts: Dict[int, int]


def infer_distribution_from_filename(path: Path) -> Optional[str]:
    """
    Infer one of the seven canonical distribution names from a filename.
    """
    filename = path.stem.lower()

    for canonical_name, aliases in DISTRIBUTION_ALIASES.items():
        for alias in aliases:
            pattern = rf"(^|[^a-z]){re.escape(alias)}([^a-z]|$)"

            if re.search(pattern, filename):
                return canonical_name

    return None


def discover_dataset_records(
    data_roots: Sequence[Path],
    allowed_problem_sizes: Sequence[int],
    allowed_distributions: Sequence[str],
    load_coordinate_list,
) -> List[DatasetRecord]:
    """
    Inspect PKL files and record which requested problem sizes they contain.
    """
    records: List[DatasetRecord] = []
    seen_paths = set()

    for root in data_roots:
        if not root.exists():
            print(f"WARNING: data root does not exist: {root}")
            continue

        for path_string in sorted(
            glob.glob(str(root / "**" / "*.pkl"), recursive=True)
        ):
            path = Path(path_string).resolve()

            if path in seen_paths:
                continue

            seen_paths.add(path)

            distribution = infer_distribution_from_filename(path)

            if distribution is None:
                print(
                    "WARNING: distribution could not be inferred; "
                    f"skipping {path}"
                )
                continue

            if distribution not in allowed_distributions:
                continue

            try:
                coordinates = load_coordinate_list(str(path))
            except Exception as error:
                print(f"WARNING: failed to inspect {path}: {error}")
                continue

            size_counts: Dict[int, int] = {}

            for coords in coordinates:
                problem_size = int(np.asarray(coords).shape[0])

                if problem_size not in allowed_problem_sizes:
                    continue

                size_counts[problem_size] = (
                    size_counts.get(problem_size, 0) + 1
                )

            if size_counts:
                records.append(
                    DatasetRecord(
                        path=path,
                        distribution=distribution,
                        problem_size_counts=size_counts,
                    )
                )

    return records


def select_records(
    records: Sequence[DatasetRecord],
    problem_size: int,
    distribution: str,
) -> List[DatasetRecord]:
    return [
        record
        for record in records
        if (
            record.distribution == distribution
            and record.problem_size_counts.get(problem_size, 0) > 0
        )
    ]


def count_available_instances(
    records: Sequence[DatasetRecord],
    problem_size: int,
) -> int:
    return sum(
        record.problem_size_counts.get(problem_size, 0)
        for record in records
    )


def iterate_coordinate_batches(
    records: Sequence[DatasetRecord],
    problem_size: int,
    batch_size: int,
    max_instances: int,
    load_coordinate_list,
) -> Iterator[np.ndarray]:
    """
    Yield homogeneous coordinate batches from one combination.
    """
    buffer: List[np.ndarray] = []
    produced = 0

    for record in records:
        coordinates = load_coordinate_list(str(record.path))

        for coords in coordinates:
            coords = np.asarray(coords, dtype=np.float32)

            if int(coords.shape[0]) != problem_size:
                continue

            if max_instances >= 0 and produced >= max_instances:
                break

            buffer.append(coords)
            produced += 1

            if len(buffer) == batch_size:
                yield np.stack(buffer, axis=0)
                buffer = []

        if max_instances >= 0 and produced >= max_instances:
            break

    if buffer:
        yield np.stack(buffer, axis=0)


# ============================================================
# Model loading
# ============================================================

def infer_decoder_layer_num(
    state_dict: Mapping[str, torch.Tensor],
    fallback: int = 6,
) -> int:
    """
    Infer decoder_layer_num from checkpoint parameter names.
    """
    layer_indices: List[int] = []

    patterns = (
        re.compile(r"decoder\.layers\.(\d+)\."),
        re.compile(r"layers\.(\d+)\."),
    )

    for key in state_dict.keys():
        for pattern in patterns:
            match = pattern.search(key)

            if match is not None:
                layer_indices.append(int(match.group(1)))

    if not layer_indices:
        print(
            "WARNING: decoder_layer_num could not be inferred. "
            f"Using fallback={fallback}."
        )
        return fallback

    decoder_layer_num = max(layer_indices) + 1

    print(
        "Inferred decoder_layer_num from checkpoint:",
        decoder_layer_num,
    )

    return decoder_layer_num


def build_lehd_model(
    Model,
    checkpoint_path: Path,
    device: torch.device,
    mode: str,
    load_checkpoint,
    get_state_dict_from_checkpoint,
):
    """
    Build the patched LEHD model and load the original checkpoint.
    """
    checkpoint = load_checkpoint(
        path=str(checkpoint_path),
        device=device,
    )

    state_dict = get_state_dict_from_checkpoint(checkpoint)

    decoder_layer_num = infer_decoder_layer_num(
        state_dict=state_dict,
        fallback=6,
    )

    model_params = {
        "mode": mode,
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
    model.mode = mode

    required_methods = (
        "clear_cache",
        "pre_forward",
        "get_step_embeddings",
        "get_action_probs",
    )

    missing_methods = [
        name
        for name in required_methods
        if not hasattr(model, name)
    ]

    if missing_methods:
        raise AttributeError(
            "The imported TSPModel does not support activation patching. "
            f"Missing methods: {missing_methods}"
        )

    return model, model_params


# ============================================================
# Original LEHD environment helpers
# ============================================================

def build_lehd_env(Env, mode: str):
    """
    Build the original pure-greedy LEHD environment.

    RRC and repair are disabled.
    """
    env_params = {
        "mode": mode,
        "data_path": "",
        "sub_path": False,
        "RRC_budget": 0,
    }

    return Env(**env_params)


def inject_problems_into_env(
    env,
    problems: torch.Tensor,
    device: torch.device,
    move_tensor_attrs_to_device,
) -> None:
    """
    Inject external coordinates exactly as in the original inference script.

    A dummy identity solution is created only for interface compatibility.
    It is never used as the predicted tour.
    """
    batch_size, problem_size, _ = problems.shape

    env.problems = problems.to(device)
    env.batch_size = batch_size
    env.problem_size = problem_size

    env.solution = torch.arange(
        problem_size,
        dtype=torch.long,
        device=device,
    )[None, :].expand(
        batch_size,
        problem_size,
    ).clone()

    move_tensor_attrs_to_device(env, device)


def reset_and_pre_step(
    env,
    mode: str,
    device: torch.device,
    move_tensor_attrs_to_device,
):
    """
    Reset the original LEHD environment and obtain its initial state.
    """
    try:
        reset_state, _, _ = env.reset(mode)
    except TypeError:
        reset_state, _, _ = env.reset()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(reset_state, device)

    state, reward, reward_student, done = env.pre_step()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(state, device)

    return state, reward, reward_student, done


def take_environment_step(
    env,
    selected_node: torch.Tensor,
    device: torch.device,
    move_tensor_attrs_to_device,
):
    """
    Advance the original environment with the same clean teacher/student node.
    """
    selected_node = selected_node.to(
        device=device,
        dtype=torch.long,
    )

    move_tensor_attrs_to_device(env, device)

    state, reward, reward_student, done = env.step(
        selected_node,
        selected_node,
    )

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(state, device)

    return state, reward, reward_student, done


# ============================================================
# Statistical accumulators
# ============================================================

@dataclass
class VectorMean:
    total: torch.Tensor
    count: int = 0

    @classmethod
    def create(cls, embedding_dim: int) -> "VectorMean":
        return cls(
            total=torch.zeros(
                embedding_dim,
                dtype=torch.float64,
                device="cpu",
            ),
            count=0,
        )

    def update(self, embeddings: torch.Tensor) -> None:
        """
        Update a CPU float64 running sum using [B, E] embeddings.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                "Expected embeddings with shape [B, E], "
                f"received {tuple(embeddings.shape)}."
            )

        self.total += (
            embeddings
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .sum(dim=0)
        )

        self.count += int(embeddings.size(0))

    def finalize(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError(
                "Cannot finalize an empty embedding accumulator."
            )

        return (
            self.total / self.count
        ).to(dtype=torch.float32)


@dataclass
class ScalarStats:
    total: float = 0.0
    total_square: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        values = (
            values
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
        )

        self.total += float(values.sum().item())
        self.total_square += float(
            torch.square(values).sum().item()
        )
        self.count += int(values.numel())

    def finalize(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "sem": float("nan"),
                "count": 0,
            }

        mean = self.total / self.count

        if self.count > 1:
            variance = (
                self.total_square
                - self.count * mean * mean
            ) / (self.count - 1)

            variance = max(variance, 0.0)
            std = math.sqrt(variance)
            sem = std / math.sqrt(self.count)
        else:
            std = 0.0
            sem = 0.0

        return {
            "mean": mean,
            "std": std,
            "sem": sem,
            "count": self.count,
        }


def create_metric_accumulators(
    steps: Iterable[int],
) -> Dict[int, Dict[str, Dict[str, ScalarStats]]]:
    return {
        step: {
            intervention: {
                metric_name: ScalarStats()
                for metric_name in METRIC_NAMES
            }
            for intervention in ("start", "current")
        }
        for step in steps
    }


# ============================================================
# Pass 1: clean mean collection with the original environment
# ============================================================

def collect_step_conditioned_means(
    model,
    Env,
    records: Sequence[DatasetRecord],
    problem_size: int,
    batch_size: int,
    max_instances: int,
    device: torch.device,
    mode: str,
    load_coordinate_list,
    move_tensor_attrs_to_device,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], int]:
    """
    Collect clean Start/Current encoder embeddings at every LEHD decision step.
    """
    embedding_dim = int(model.model_params["embedding_dim"])

    start_accumulators = {
        step: VectorMean.create(embedding_dim)
        for step in range(1, problem_size)
    }

    current_accumulators = {
        step: VectorMean.create(embedding_dim)
        for step in range(1, problem_size)
    }

    total_instances = 0

    with torch.inference_mode():
        for batch_number, batch_coords in enumerate(
            iterate_coordinate_batches(
                records=records,
                problem_size=problem_size,
                batch_size=batch_size,
                max_instances=max_instances,
                load_coordinate_list=load_coordinate_list,
            ),
            start=1,
        ):
            problems = torch.as_tensor(
                batch_coords,
                dtype=torch.float32,
                device=device,
            )

            current_batch_size = int(problems.size(0))
            total_instances += current_batch_size

            env = build_lehd_env(
                Env=Env,
                mode=mode,
            )

            inject_problems_into_env(
                env=env,
                problems=problems,
                device=device,
                move_tensor_attrs_to_device=move_tensor_attrs_to_device,
            )

            state, _, _, done = reset_and_pre_step(
                env=env,
                mode=mode,
                device=device,
                move_tensor_attrs_to_device=move_tensor_attrs_to_device,
            )

            model.clear_cache()
            current_step = 0

            while not done:
                if current_step == 0:
                    # Pure-greedy LEHD always starts from node 0.
                    clean_selected = torch.zeros(
                        current_batch_size,
                        dtype=torch.long,
                        device=device,
                    )

                else:
                    # Match the original model behavior: encode once before
                    # the first model-based decision.
                    if current_step == 1:
                        model.pre_forward(
                            state,
                            detach=True,
                        )

                    step_embeddings = model.get_step_embeddings(
                        selected_node_list=env.selected_node_list,
                        detach=True,
                        clone=False,
                    )

                    start_accumulators[current_step].update(
                        step_embeddings["start_embedding"]
                    )

                    current_accumulators[current_step].update(
                        step_embeddings["current_embedding"]
                    )

                    clean_probs = model.get_action_probs(
                        selected_node_list=env.selected_node_list,
                    )

                    clean_selected = clean_probs.argmax(
                        dim=-1
                    )

                current_step += 1

                state, _, _, done = take_environment_step(
                    env=env,
                    selected_node=clean_selected,
                    device=device,
                    move_tensor_attrs_to_device=move_tensor_attrs_to_device,
                )

            print(
                f"      Mean pass batch {batch_number}: "
                f"{total_instances} instances"
            )

    if total_instances == 0:
        raise RuntimeError(
            "No instances were available for mean collection."
        )

    start_means = {
        step: accumulator.finalize()
        for step, accumulator in start_accumulators.items()
    }

    current_means = {
        step: accumulator.finalize()
        for step, accumulator in current_accumulators.items()
    }

    return start_means, current_means, total_instances


# ============================================================
# Pass 2: step-local patch evaluation with the original environment
# ============================================================

def normalize_candidate_probabilities(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """
    Renormalize LEHD candidate probabilities before TV computation.

    The canonical model adds a very small epsilon to tiny probabilities after
    softmax, so the candidate probabilities may sum to slightly more than one.
    """
    denominator = probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)

    return probabilities / denominator


def compute_patch_metrics(
    clean_details: Mapping[str, torch.Tensor],
    patched_details: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute matching, TV distance, and clean-next-node probability changes.
    """
    clean_candidate_probs = normalize_candidate_probabilities(
        clean_details["candidate_probs"]
    )

    patched_candidate_probs = normalize_candidate_probabilities(
        patched_details["candidate_probs"]
    )

    clean_next_node = clean_details["probs"].argmax(
        dim=-1
    )

    patched_next_node = patched_details["probs"].argmax(
        dim=-1
    )

    matching = (
        clean_next_node == patched_next_node
    ).to(dtype=torch.float32)

    change_rate = 1.0 - matching

    tv_distance = 0.5 * torch.abs(
        clean_candidate_probs - patched_candidate_probs
    ).sum(dim=-1)

    unselected_node_indices = clean_details[
        "unselected_node_indices"
    ]

    clean_candidate_position = (
        unselected_node_indices
        == clean_next_node[:, None]
    ).to(dtype=torch.long).argmax(dim=1)

    gather_index = clean_candidate_position[:, None]

    clean_next_probability = clean_candidate_probs.gather(
        dim=1,
        index=gather_index,
    ).squeeze(1)

    patched_probability_on_clean_next = (
        patched_candidate_probs.gather(
            dim=1,
            index=gather_index,
        ).squeeze(1)
    )

    next_probability_difference = (
        patched_probability_on_clean_next
        - clean_next_probability
    )

    next_probability_drop = (
        clean_next_probability
        - patched_probability_on_clean_next
    )

    absolute_next_probability_difference = torch.abs(
        next_probability_difference
    )

    return {
        "matching": matching,
        "change_rate": change_rate,
        "tv_distance": tv_distance,
        "clean_next_probability": clean_next_probability,
        "patched_probability_on_clean_next":
            patched_probability_on_clean_next,
        "next_probability_difference":
            next_probability_difference,
        "next_probability_drop":
            next_probability_drop,
        "absolute_next_probability_difference":
            absolute_next_probability_difference,
    }


def evaluate_step_local_patches(
    model,
    Env,
    records: Sequence[DatasetRecord],
    problem_size: int,
    batch_size: int,
    max_instances: int,
    device: torch.device,
    mode: str,
    start_means: Mapping[int, torch.Tensor],
    current_means: Mapping[int, torch.Tensor],
    include_forced_last_step: bool,
    load_coordinate_list,
    move_tensor_attrs_to_device,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, ScalarStats]]],
    int,
]:
    """
    Evaluate Start and Current patches on clean LEHD states.

    Every environment transition uses the clean greedy action.
    """
    metric_steps = list(range(1, problem_size))

    if not include_forced_last_step:
        metric_steps = metric_steps[:-1]

    accumulators = create_metric_accumulators(
        steps=metric_steps,
    )

    total_instances = 0

    with torch.inference_mode():
        for batch_number, batch_coords in enumerate(
            iterate_coordinate_batches(
                records=records,
                problem_size=problem_size,
                batch_size=batch_size,
                max_instances=max_instances,
                load_coordinate_list=load_coordinate_list,
            ),
            start=1,
        ):
            problems = torch.as_tensor(
                batch_coords,
                dtype=torch.float32,
                device=device,
            )

            current_batch_size = int(problems.size(0))
            total_instances += current_batch_size

            env = build_lehd_env(
                Env=Env,
                mode=mode,
            )

            inject_problems_into_env(
                env=env,
                problems=problems,
                device=device,
                move_tensor_attrs_to_device=move_tensor_attrs_to_device,
            )

            state, _, _, done = reset_and_pre_step(
                env=env,
                mode=mode,
                device=device,
                move_tensor_attrs_to_device=move_tensor_attrs_to_device,
            )

            model.clear_cache()
            current_step = 0

            while not done:
                if current_step == 0:
                    clean_selected = torch.zeros(
                        current_batch_size,
                        dtype=torch.long,
                        device=device,
                    )

                else:
                    if current_step == 1:
                        model.pre_forward(
                            state,
                            detach=True,
                        )

                    clean_details = model.get_action_probs(
                        selected_node_list=env.selected_node_list,
                        return_details=True,
                    )

                    if current_step in accumulators:
                        start_patch = start_means[
                            current_step
                        ].to(
                            device=device,
                            dtype=problems.dtype,
                        )

                        current_patch = current_means[
                            current_step
                        ].to(
                            device=device,
                            dtype=problems.dtype,
                        )

                        start_patched_details = model.get_action_probs(
                            selected_node_list=env.selected_node_list,
                            start_node_patch=start_patch,
                            return_details=True,
                        )

                        current_patched_details = model.get_action_probs(
                            selected_node_list=env.selected_node_list,
                            current_node_patch=current_patch,
                            return_details=True,
                        )

                        start_metrics = compute_patch_metrics(
                            clean_details=clean_details,
                            patched_details=start_patched_details,
                        )

                        current_metrics = compute_patch_metrics(
                            clean_details=clean_details,
                            patched_details=current_patched_details,
                        )

                        for metric_name, values in start_metrics.items():
                            accumulators[
                                current_step
                            ]["start"][metric_name].update(values)

                        for metric_name, values in current_metrics.items():
                            accumulators[
                                current_step
                            ]["current"][metric_name].update(values)

                    # The environment always advances with the clean action.
                    clean_selected = clean_details[
                        "probs"
                    ].argmax(dim=-1)

                current_step += 1

                state, _, _, done = take_environment_step(
                    env=env,
                    selected_node=clean_selected,
                    device=device,
                    move_tensor_attrs_to_device=move_tensor_attrs_to_device,
                )

            print(
                f"      Patch pass batch {batch_number}: "
                f"{total_instances} instances"
            )

    if total_instances == 0:
        raise RuntimeError(
            "No instances were available for patch evaluation."
        )

    return accumulators, total_instances


# ============================================================
# Result serialization
# ============================================================

def stats_to_columns(
    metric_name: str,
    stats: Mapping[str, float],
) -> Dict[str, float]:
    return {
        f"{metric_name}_mean": stats["mean"],
        f"{metric_name}_std": stats["std"],
        f"{metric_name}_sem": stats["sem"],
    }


def build_metric_rows(
    accumulators: Mapping[
        int,
        Mapping[str, Mapping[str, ScalarStats]],
    ],
    problem_size: int,
    distribution: str,
    intervention: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for decoding_step in sorted(accumulators):
        finalized = {
            metric_name: accumulator.finalize()
            for metric_name, accumulator
            in accumulators[decoding_step][intervention].items()
        }

        remaining_nodes = problem_size - decoding_step
        matching_count = finalized["matching"]["count"]
        matching_sum = accumulators[
            decoding_step
        ][intervention]["matching"].total

        row: Dict[str, Any] = {
            "problem_size": problem_size,
            "distribution": distribution,
            "intervention": intervention,
            "decoding_step": decoding_step,
            "prefix_length": decoding_step,
            "remaining_unvisited_before_decision": remaining_nodes,
            "random_matching_probability": 1.0 / remaining_nodes,
            "num_instances": matching_count,
            "num_matching_instances": int(round(matching_sum)),
            "matching_rate_over_all_instances": (
                matching_sum / matching_count
                if matching_count > 0
                else float("nan")
            ),
        }

        for metric_name, stats in finalized.items():
            row.update(
                stats_to_columns(
                    metric_name=metric_name,
                    stats=stats,
                )
            )

        rows.append(row)

    return rows


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    selected_columns: Optional[Sequence[str]] = None,
) -> None:
    if not rows:
        raise ValueError(
            f"Cannot write an empty CSV file: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        list(selected_columns)
        if selected_columns is not None
        else list(rows[0].keys())
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )


def save_intervention_results(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    means: Mapping[int, torch.Tensor],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "mean_embedding_by_step": {
                int(step): tensor.detach().cpu()
                for step, tensor in means.items()
            },
            "steps": sorted(int(step) for step in means),
        },
        output_dir / "mean_embedding_by_step.pt",
    )

    write_csv(
        rows=rows,
        path=output_dir / "metrics_by_step.csv",
    )

    common_columns = (
        "problem_size",
        "distribution",
        "intervention",
        "decoding_step",
        "prefix_length",
        "remaining_unvisited_before_decision",
        "random_matching_probability",
        "num_instances",
    )

    write_csv(
        rows=rows,
        path=output_dir / "matching_by_step.csv",
        selected_columns=common_columns + (
            "num_matching_instances",
            "matching_rate_over_all_instances",
            "matching_mean",
            "matching_std",
            "matching_sem",
            "change_rate_mean",
            "change_rate_std",
            "change_rate_sem",
        ),
    )

    write_csv(
        rows=rows,
        path=output_dir / "tv_distance_by_step.csv",
        selected_columns=common_columns + (
            "tv_distance_mean",
            "tv_distance_std",
            "tv_distance_sem",
        ),
    )

    write_csv(
        rows=rows,
        path=output_dir / "next_node_probability_by_step.csv",
        selected_columns=common_columns + (
            "clean_next_probability_mean",
            "clean_next_probability_std",
            "clean_next_probability_sem",
            "patched_probability_on_clean_next_mean",
            "patched_probability_on_clean_next_std",
            "patched_probability_on_clean_next_sem",
            "next_probability_difference_mean",
            "next_probability_difference_std",
            "next_probability_difference_sem",
            "next_probability_drop_mean",
            "next_probability_drop_std",
            "next_probability_drop_sem",
            "absolute_next_probability_difference_mean",
            "absolute_next_probability_difference_std",
            "absolute_next_probability_difference_sem",
        ),
    )


def combination_is_complete(output_dir: Path) -> bool:
    required = (
        output_dir / "metadata.json",
        output_dir / "start" / "metrics_by_step.csv",
        output_dir / "current" / "metrics_by_step.csv",
        output_dir / "start" / "mean_embedding_by_step.pt",
        output_dir / "current" / "mean_embedding_by_step.pt",
    )

    return all(path.exists() for path in required)


# ============================================================
# One problem-size/distribution combination
# ============================================================

def run_one_combination(
    model,
    Env,
    mean_records: Sequence[DatasetRecord],
    evaluation_records: Sequence[DatasetRecord],
    problem_size: int,
    distribution: str,
    batch_size: int,
    max_instances: int,
    device: torch.device,
    mode: str,
    output_dir: Path,
    include_forced_last_step: bool,
    load_coordinate_list,
    move_tensor_attrs_to_device,
    checkpoint_path: Path,
    patched_model_path: Path,
    model_params: Mapping[str, Any],
) -> Dict[str, Any]:
    start_time = time.time()

    print("\n============================================================")
    print(f"{problem_size} | distribution={distribution}")
    print(
        "  Mean instances available:",
        count_available_instances(
            mean_records,
            problem_size,
        ),
    )
    print(
        "  Evaluation instances available:",
        count_available_instances(
            evaluation_records,
            problem_size,
        ),
    )
    print("  Batch size:", batch_size)

    print("  Pass 1/2: collecting step-conditioned means")

    start_means, current_means, mean_instance_count = (
        collect_step_conditioned_means(
            model=model,
            Env=Env,
            records=mean_records,
            problem_size=problem_size,
            batch_size=batch_size,
            max_instances=max_instances,
            device=device,
            mode=mode,
            load_coordinate_list=load_coordinate_list,
            move_tensor_attrs_to_device=move_tensor_attrs_to_device,
        )
    )

    print("  Pass 2/2: evaluating step-local patches")

    accumulators, evaluation_instance_count = (
        evaluate_step_local_patches(
            model=model,
            Env=Env,
            records=evaluation_records,
            problem_size=problem_size,
            batch_size=batch_size,
            max_instances=max_instances,
            device=device,
            mode=mode,
            start_means=start_means,
            current_means=current_means,
            include_forced_last_step=include_forced_last_step,
            load_coordinate_list=load_coordinate_list,
            move_tensor_attrs_to_device=move_tensor_attrs_to_device,
        )
    )

    start_rows = build_metric_rows(
        accumulators=accumulators,
        problem_size=problem_size,
        distribution=distribution,
        intervention="start",
    )

    current_rows = build_metric_rows(
        accumulators=accumulators,
        problem_size=problem_size,
        distribution=distribution,
        intervention="current",
    )

    save_intervention_results(
        output_dir=output_dir / "start",
        rows=start_rows,
        means=start_means,
    )

    save_intervention_results(
        output_dir=output_dir / "current",
        rows=current_rows,
        means=current_means,
    )

    elapsed = time.time() - start_time

    metadata = {
        "status": "success",
        "problem_size": problem_size,
        "distribution": distribution,
        "batch_size": batch_size,
        "mean_instance_count": mean_instance_count,
        "evaluation_instance_count": evaluation_instance_count,
        "include_forced_last_step": include_forced_last_step,
        "checkpoint_path": str(checkpoint_path),
        "patched_model_path": str(patched_model_path),
        "model_params": dict(model_params),
        "use_rrc": False,
        "rrc_budget": 0,
        "use_repair": False,
        "start_policy": "node_0",
        "step_local_intervention": True,
        "rollout_advances_with_clean_action": True,
        "mean_dataset_files": [
            str(record.path)
            for record in mean_records
        ],
        "evaluation_dataset_files": [
            str(record.path)
            for record in evaluation_records
        ],
        "elapsed_seconds": elapsed,
    }

    save_json(
        payload=metadata,
        path=output_dir / "metadata.json",
    )

    return {
        "problem_size": problem_size,
        "distribution": distribution,
        "status": "success",
        "mean_instance_count": mean_instance_count,
        "evaluation_instance_count": evaluation_instance_count,
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "error_message": "",
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(
        args.checkpoint_path
    ).expanduser().resolve()

    patched_model_path = Path(
        args.patched_model_path
    ).expanduser().resolve()

    lehd_root = Path(
        args.lehd_root
    ).expanduser().resolve()

    utils_root = Path(
        args.utils_root
    ).expanduser().resolve()

    results_root = Path(
        args.results_root
    ).expanduser().resolve()

    data_roots = [
        Path(path).expanduser().resolve()
        for path in args.data_roots
    ]

    if args.mean_data_roots is None:
        mean_data_roots = data_roots
    else:
        mean_data_roots = [
            Path(path).expanduser().resolve()
            for path in args.mean_data_roots
        ]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    shared_utils = import_shared_utils(
        utils_root=utils_root,
    )

    setup_torch_device = shared_utils[
        "setup_torch_device"
    ]

    move_tensor_attrs_to_device = shared_utils[
        "move_tensor_attrs_to_device"
    ]

    load_coordinate_list = shared_utils[
        "load_coordinate_list"
    ]

    load_checkpoint = shared_utils[
        "load_checkpoint"
    ]

    get_state_dict_from_checkpoint = shared_utils[
        "get_state_dict_from_checkpoint"
    ]

    empty_cuda_cache_if_available = shared_utils[
        "empty_cuda_cache_if_available"
    ]

    device = setup_torch_device(
        cuda_device_num=args.cuda_device_num,
        set_default_device=True,
        legacy_cuda_default_tensor_type=False,
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Patched model:", patched_model_path)
    print("LEHD root:", lehd_root)
    print("Results root:", results_root)

    Env = import_lehd_env(
        lehd_root=lehd_root,
    )

    Model = import_patched_model(
        model_path=patched_model_path,
    )

    model, model_params = build_lehd_model(
        Model=Model,
        checkpoint_path=checkpoint_path,
        device=device,
        mode=args.mode,
        load_checkpoint=load_checkpoint,
        get_state_dict_from_checkpoint=get_state_dict_from_checkpoint,
    )

    print("LEHD patched model loaded.")
    print("Model parameters:", model_params)

    print("\nDiscovering mean datasets...")

    mean_manifest = discover_dataset_records(
        data_roots=mean_data_roots,
        allowed_problem_sizes=args.problem_sizes,
        allowed_distributions=args.distributions,
        load_coordinate_list=load_coordinate_list,
    )

    print(
        "Mean dataset files discovered:",
        len(mean_manifest),
    )

    if mean_data_roots == data_roots:
        evaluation_manifest = mean_manifest
    else:
        print("\nDiscovering evaluation datasets...")

        evaluation_manifest = discover_dataset_records(
            data_roots=data_roots,
            allowed_problem_sizes=args.problem_sizes,
            allowed_distributions=args.distributions,
            load_coordinate_list=load_coordinate_list,
        )

    print(
        "Evaluation dataset files discovered:",
        len(evaluation_manifest),
    )

    summary_rows: List[Dict[str, Any]] = []

    for problem_size in args.problem_sizes:
        batch_size = (
            args.batch_size
            if args.batch_size > 0
            else DEFAULT_BATCH_SIZE_BY_PROBLEM.get(
                problem_size,
                8,
            )
        )

        for distribution in args.distributions:
            output_dir = (
                results_root
                / str(problem_size)
                / distribution
            )

            if (
                combination_is_complete(output_dir)
                and not args.overwrite
            ):
                print(
                    "\nSkip completed combination: "
                    f"{problem_size} | {distribution}"
                )

                summary_rows.append(
                    {
                        "problem_size": problem_size,
                        "distribution": distribution,
                        "status": "skipped",
                        "mean_instance_count": "",
                        "evaluation_instance_count": "",
                        "elapsed_seconds": 0.0,
                        "output_dir": str(output_dir),
                        "error_message": "",
                    }
                )

                continue

            mean_records = select_records(
                records=mean_manifest,
                problem_size=problem_size,
                distribution=distribution,
            )

            evaluation_records = select_records(
                records=evaluation_manifest,
                problem_size=problem_size,
                distribution=distribution,
            )

            if not mean_records or not evaluation_records:
                message = (
                    "No matching dataset files were found for "
                    f"problem_size={problem_size}, "
                    f"distribution={distribution}."
                )

                print(f"WARNING: {message}")

                row = {
                    "problem_size": problem_size,
                    "distribution": distribution,
                    "status": "missing_data",
                    "mean_instance_count": 0,
                    "evaluation_instance_count": 0,
                    "elapsed_seconds": 0.0,
                    "output_dir": str(output_dir),
                    "error_message": message,
                }

                summary_rows.append(row)

                if args.strict:
                    raise RuntimeError(message)

                continue

            try:
                row = run_one_combination(
                    model=model,
                    Env=Env,
                    mean_records=mean_records,
                    evaluation_records=evaluation_records,
                    problem_size=problem_size,
                    distribution=distribution,
                    batch_size=batch_size,
                    max_instances=args.max_instances_per_combination,
                    device=device,
                    mode=args.mode,
                    output_dir=output_dir,
                    include_forced_last_step=(
                        not args.exclude_forced_last_step
                    ),
                    load_coordinate_list=load_coordinate_list,
                    move_tensor_attrs_to_device=move_tensor_attrs_to_device,
                    checkpoint_path=checkpoint_path,
                    patched_model_path=patched_model_path,
                    model_params=model_params,
                )

            except Exception as error:
                print(
                    f"ERROR for {problem_size} | {distribution}: "
                    f"{error}"
                )

                row = {
                    "problem_size": problem_size,
                    "distribution": distribution,
                    "status": "failed",
                    "mean_instance_count": "",
                    "evaluation_instance_count": "",
                    "elapsed_seconds": 0.0,
                    "output_dir": str(output_dir),
                    "error_message": str(error),
                }

                if args.strict:
                    raise

            summary_rows.append(row)

            write_csv(
                rows=summary_rows,
                path=results_root / "run_summary.csv",
            )

            empty_cuda_cache_if_available()

    write_csv(
        rows=summary_rows,
        path=results_root / "run_summary.csv",
    )

    print("\n============================================================")
    print("All combinations finished.")
    print(
        "Summary CSV:",
        results_root / "run_summary.csv",
    )


if __name__ == "__main__":
    main()

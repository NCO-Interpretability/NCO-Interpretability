from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Experiment defaults
# =============================================================================

DEFAULT_PROBLEM_SIZES = [20, 50, 100, 200, 500, 1000]

DEFAULT_MAX_INSTANCES = {
    20: 100,
    50: 100,
    100: 100,
    200: 64,
    500: 24,
    1000: 8,
}

DEFAULT_MIN_VISITED = {
    20: 8,
    50: 10,
    100: 20,
    200: 40,
    500: 100,
    1000: 200,
}

DEFAULT_STEPS_PER_INSTANCE = {
    20: 6,
    50: 8,
    100: 10,
    200: 12,
    500: 12,
    1000: 10,
}

CATEGORY_DISTANCE_MATCHED_MAX_ANGLE = "distance_matched_max_angle"
CATEGORY_MAX_ANGLE_UNCONSTRAINED = "max_angle_unconstrained"
CATEGORY_SAME_DIRECTION_DIFFERENT_DISTANCE = (
    "same_direction_different_distance"
)
CATEGORY_RANDOM_VISITED_CONTROL = "random_visited_control"

EPS = 1e-12


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class InstanceRecord:
    problem_size: int
    source_directory: str
    source_directory_name: str
    source_file: str
    source_file_name: str
    instance_index_in_file: int
    coords: np.ndarray


@dataclass
class DonorGeometry:
    donor_node: int
    distance_from_current: float
    distance_ratio_to_start: float
    absolute_distance_difference: float
    absolute_log_distance_ratio: float
    unsigned_angle_to_start_deg: float
    signed_angle_from_start_deg: float


@dataclass
class DonorChoice:
    category: str
    available: bool
    donor_node: Optional[int]
    reason: str
    category_candidate_count: int


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the post-encoder LEHD start representation using "
            "visited donor nodes from automatically discovered PKL datasets."
        )
    )

    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument(
        "--data_root",
        type=Path,
        help=(
            "Root directory recursively searched for directories containing "
            "PKL datasets."
        ),
    )
    data_source_group.add_argument(
        "--dataset_dirs",
        type=Path,
        nargs="+",
        help="Explicit dataset directories containing PKL files.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        required=True,
        help="Directory where experiment outputs are written.",
    )
    parser.add_argument(
        "--lehd_root",
        type=Path,
        required=True,
        help="Root directory containing the LEHD package.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to the pretrained LEHD checkpoint.",
    )
    parser.add_argument(
        "--utils_root",
        type=Path,
        required=True,
        help="Directory containing util.py or utils/util.py.",
    )
    parser.add_argument(
        "--problem_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_PROBLEM_SIZES,
    )
    parser.add_argument(
        "--max_instances_map",
        type=str,
        default="20:100,50:100,100:100,200:64,500:24,1000:8",
    )
    parser.add_argument(
        "--min_visited_map",
        type=str,
        default="20:8,50:10,100:20,200:40,500:100,1000:200",
    )
    parser.add_argument(
        "--steps_per_instance_map",
        type=str,
        default="20:6,50:8,100:10,200:12,500:12,1000:10",
    )
    parser.add_argument(
        "--instance_selection",
        choices=["first", "random"],
        default="first",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--cuda_device_num",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--distance_match_tolerance",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--same_direction_max_angle_deg",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--different_distance_ratio_low",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--different_distance_ratio_high",
        type=float,
        default=1.30,
    )
    parser.add_argument(
        "--exclude_previous_node",
        action="store_true",
    )
    parser.add_argument(
        "--top_k_overlap",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--geometry_epsilon",
        type=float,
        default=1e-12,
        help=(
            "Distances at or below this tolerance are treated as zero. "
            "A state with zero Current-to-Start distance is skipped and logged."
        ),
    )
    parser.add_argument(
        "--save_every_instances",
        type=int,
        default=5,
        help=(
            "Write per-size checkpoint CSV files after this many instances. "
            "Use 1 for maximum fault tolerance."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()
    if not args.problem_sizes or any(value <= 1 for value in args.problem_sizes):
        parser.error("--problem_sizes must contain integers greater than 1.")
    if args.cuda_device_num < 0:
        parser.error("--cuda_device_num cannot be negative.")
    if args.rollout_horizon <= 0:
        parser.error("--rollout_horizon must be positive.")
    if args.distance_match_tolerance < 0:
        parser.error("--distance_match_tolerance cannot be negative.")
    if not 0 <= args.same_direction_max_angle_deg <= 180:
        parser.error("--same_direction_max_angle_deg must be in [0, 180].")
    if args.different_distance_ratio_low <= 0:
        parser.error("--different_distance_ratio_low must be positive.")
    if args.different_distance_ratio_high <= args.different_distance_ratio_low:
        parser.error(
            "--different_distance_ratio_high must exceed "
            "--different_distance_ratio_low."
        )
    if args.top_k_overlap <= 0:
        parser.error("--top_k_overlap must be positive.")
    if args.geometry_epsilon <= 0:
        parser.error("--geometry_epsilon must be positive.")
    if args.save_every_instances <= 0:
        parser.error("--save_every_instances must be positive.")
    for field_name in (
        "max_instances_map",
        "min_visited_map",
        "steps_per_instance_map",
    ):
        try:
            parsed = parse_int_map(getattr(args, field_name))
        except (TypeError, ValueError) as exc:
            parser.error(f"--{field_name} is invalid: {exc}")
        if not parsed:
            parser.error(f"--{field_name} cannot be empty.")
        if field_name == "max_instances_map":
            invalid = [value for value in parsed.values() if value == 0 or value < -1]
        else:
            invalid = [value for value in parsed.values() if value <= 0]
        if invalid:
            parser.error(f"--{field_name} contains invalid values: {invalid}")
    return args


def parse_int_map(text: str) -> Dict[int, int]:
    result: Dict[int, int] = {}

    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        key_text, value_text = item.split(":", maxsplit=1)
        result[int(key_text)] = int(value_text)

    return result


# =============================================================================
# Local utilities and LEHD imports
# =============================================================================

def import_local_utilities(utils_root: Path):
    """Import shared utility functions from an explicit directory."""
    utils_root = utils_root.expanduser().resolve()
    direct_util_file = utils_root / "util.py"
    nested_util_file = utils_root / "utils" / "util.py"

    if direct_util_file.is_file():
        import_parent = utils_root.parent
    elif nested_util_file.is_file():
        import_parent = utils_root
    else:
        raise FileNotFoundError(
            "Could not find util.py. Expected one of:\n"
            f"  {direct_util_file}\n"
            f"  {nested_util_file}"
        )

    sys.path.insert(0, str(import_parent))

    from utils.util import (
        setup_torch_device,
        load_coordinate_list,
        group_indices_by_problem_size,
        load_checkpoint,
        get_state_dict_from_checkpoint,
    )

    return {
        "setup_torch_device": setup_torch_device,
        "load_coordinate_list": load_coordinate_list,
        "group_indices_by_problem_size": group_indices_by_problem_size,
        "load_checkpoint": load_checkpoint,
        "get_state_dict_from_checkpoint": get_state_dict_from_checkpoint,
    }


def import_lehd_model(lehd_root: Path):
    model_file = lehd_root / "LEHD" / "TSP" / "TSPModel.py"

    if not model_file.is_file():
        raise FileNotFoundError(
            f"LEHD TSPModel.py was not found: {model_file}"
        )

    sys.path.insert(0, str(lehd_root))

    from LEHD.TSP.TSPModel import TSPModel

    return TSPModel


def infer_decoder_layer_num(
    state_dict: Dict[str, torch.Tensor],
    fallback: int = 6,
) -> int:
    layer_indices: List[int] = []
    patterns = [
        re.compile(r"decoder\.layers\.(\d+)\."),
        re.compile(r"layers\.(\d+)\."),
    ]

    for key in state_dict.keys():
        for pattern in patterns:
            match = pattern.search(key)
            if match is not None:
                layer_indices.append(int(match.group(1)))

    if not layer_indices:
        print(
            f"WARNING: decoder layer count was not inferred; "
            f"using fallback={fallback}.",
            flush=True,
        )
        return fallback

    return max(layer_indices) + 1


def build_model(
    Model,
    checkpoint_path: Path,
    device: torch.device,
    load_checkpoint,
    get_state_dict_from_checkpoint,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, model_params


# =============================================================================
# Dataset discovery and loading
# =============================================================================

def discover_dataset_directories(data_root: Path) -> List[Path]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root was not found: {data_root}")

    candidates: List[Path] = []
    if any(data_root.glob("*.pkl")):
        candidates.append(data_root)

    for path in sorted(
        (item for item in data_root.rglob("*") if item.is_dir()),
        key=lambda item: str(item),
    ):
        if any(path.glob("*.pkl")):
            candidates.append(path)

    unique_candidates = list(dict.fromkeys(candidates))
    if not unique_candidates:
        raise FileNotFoundError(
            f"No directories containing PKL files were found under {data_root}."
        )
    return unique_candidates


def validate_explicit_dataset_directories(
    dataset_dirs: Sequence[Path],
) -> List[Path]:
    validated: List[Path] = []
    for directory in dataset_dirs:
        path = directory.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory was not found: {path}")
        if not any(path.glob("*.pkl")):
            raise FileNotFoundError(
                f"Dataset directory contains no PKL files: {path}"
            )
        validated.append(path)
    return list(dict.fromkeys(validated))

def normalize_coords(coords: Any) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float32)

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"Expected coordinate shape [N, 2], received {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError("Coordinates contain NaN or infinity.")

    return array


def scan_datasets(
    dataset_directories: Sequence[Path],
    problem_sizes: Sequence[int],
    load_coordinate_list,
    group_indices_by_problem_size,
) -> Tuple[
    Dict[int, Dict[str, List[InstanceRecord]]],
    List[Dict[str, Any]],
]:
    """
    Load PKL files and index instances by problem size and source directory.
    """
    requested_sizes = set(int(n) for n in problem_sizes)

    by_size_and_directory: Dict[
        int,
        Dict[str, List[InstanceRecord]],
    ] = defaultdict(lambda: defaultdict(list))

    inventory_rows: List[Dict[str, Any]] = []

    for directory in dataset_directories:
        pkl_files = sorted(directory.glob("*.pkl"))

        print(
            f"\nScanning {directory} | PKL files={len(pkl_files)}",
            flush=True,
        )

        for pkl_file in pkl_files:
            inventory_row: Dict[str, Any] = {
                "source_directory": str(directory.resolve()),
                "source_directory_name": directory.name,
                "source_file": str(pkl_file.resolve()),
                "source_file_name": pkl_file.name,
                "status": "success",
                "total_instances": 0,
                "counts_by_problem_size": "",
                "error_message": "",
            }

            try:
                coords_list = load_coordinate_list(str(pkl_file))
                size_to_indices = group_indices_by_problem_size(coords_list)

                inventory_row["total_instances"] = len(coords_list)
                inventory_row["counts_by_problem_size"] = json.dumps(
                    {
                        int(size): len(indices)
                        for size, indices in sorted(size_to_indices.items())
                    }
                )

                print(
                    f"  {pkl_file.name}: "
                    f"{inventory_row['counts_by_problem_size']}",
                    flush=True,
                )

                for problem_size, indices in size_to_indices.items():
                    problem_size = int(problem_size)

                    if problem_size not in requested_sizes:
                        continue

                    for instance_index in indices:
                        coords = normalize_coords(
                            coords_list[int(instance_index)]
                        )

                        by_size_and_directory[
                            problem_size
                        ][str(directory.resolve())].append(
                            InstanceRecord(
                                problem_size=problem_size,
                                source_directory=str(directory.resolve()),
                                source_directory_name=directory.name,
                                source_file=str(pkl_file.resolve()),
                                source_file_name=pkl_file.name,
                                instance_index_in_file=int(instance_index),
                                coords=coords,
                            )
                        )

            except Exception as exc:
                inventory_row["status"] = "failed"
                inventory_row["error_message"] = str(exc)

                print(
                    f"  ERROR {pkl_file.name}: {exc}",
                    flush=True,
                )

            inventory_rows.append(inventory_row)

    return by_size_and_directory, inventory_rows


def choose_source_and_instances(
    by_size_and_directory: Dict[
        int,
        Dict[str, List[InstanceRecord]],
    ],
    problem_sizes: Sequence[int],
    max_instances_map: Dict[int, int],
    instance_selection: str,
    seed: int,
) -> Tuple[
    Dict[int, List[InstanceRecord]],
    List[Dict[str, Any]],
]:
    """
    Select one preferred source directory per problem size.
    """
    selected_by_size: Dict[int, List[InstanceRecord]] = {}
    source_rows: List[Dict[str, Any]] = []

    for problem_size in problem_sizes:
        source_map = by_size_and_directory.get(
            int(problem_size),
            {},
        )

        available_sources = [
            (
                -len(records),
                source_dir,
                records,
            )
            for source_dir, records in source_map.items()
            if records
        ]

        available_sources.sort(
            key=lambda item: (item[0], item[1])
        )

        if not available_sources:
            selected_by_size[int(problem_size)] = []
            source_rows.append(
                {
                    "problem_size": int(problem_size),
                    "status": "not_found",
                    "selected_source_directory": "",
                    "selected_source_directory_name": "",
                    "source_priority": "",
                    "available_instances": 0,
                    "selected_instances": 0,
                }
            )

            print(
                f"WARNING: No PKL instances found for N={problem_size}.",
                flush=True,
            )
            continue

        _negative_available_count, source_dir, records = available_sources[0]
        source_priority = 0

        records = sorted(
            records,
            key=lambda record: (
                record.source_file_name,
                record.instance_index_in_file,
            ),
        )

        configured_limit = max_instances_map.get(
            int(problem_size), len(records)
        )
        limit = (
            len(records)
            if configured_limit < 0
            else min(configured_limit, len(records))
        )

        if instance_selection == "random":
            rng = random.Random(
                seed + int(problem_size) * 1_000_003
            )
            chosen = rng.sample(records, limit)
            chosen.sort(
                key=lambda record: (
                    record.source_file_name,
                    record.instance_index_in_file,
                )
            )
        else:
            chosen = records[:limit]

        selected_by_size[int(problem_size)] = chosen

        source_rows.append(
            {
                "problem_size": int(problem_size),
                "status": "selected",
                "selected_source_directory": source_dir,
                "selected_source_directory_name": Path(source_dir).name,
                "source_priority": source_priority,
                "available_instances": len(records),
                "selected_instances": len(chosen),
            }
        )

        print(
            f"N={problem_size}: source={source_dir} | "
            f"available={len(records)} | selected={len(chosen)}",
            flush=True,
        )

    return selected_by_size, source_rows


# =============================================================================
# LEHD greedy decoding
# =============================================================================

def call_decoder(
    model,
    encoded_nodes: torch.Tensor,
    selected_node_list: torch.Tensor,
) -> torch.Tensor:
    """
    Call the original LEHD decoder directly on encoded nodes.
    """
    try:
        probs = model.decoder(
            encoded_nodes,
            selected_node_list,
        )
    except TypeError:
        probs = model.decoder(
            encoded_nodes,
            selected_node_list,
            attention_intervention=None,
        )

    if probs.ndim != 2:
        raise RuntimeError(
            f"Unexpected decoder output shape: {tuple(probs.shape)}"
        )

    return probs


def build_clean_greedy_tour(
    model,
    coords_tensor: torch.Tensor,
) -> Tuple[List[int], torch.Tensor]:
    """
    Reproduce pure greedy LEHD decoding with node 0 as the forced start.
    """
    problem_size = int(coords_tensor.shape[0])

    with torch.no_grad():
        encoded_nodes = model.encoder(
            coords_tensor.unsqueeze(0)
        )

        selected = torch.zeros(
            (1, 1),
            dtype=torch.long,
            device=coords_tensor.device,
        )

        while selected.shape[1] < problem_size:
            probs = call_decoder(
                model=model,
                encoded_nodes=encoded_nodes,
                selected_node_list=selected,
            )

            next_node = probs.argmax(dim=1)

            selected = torch.cat(
                [selected, next_node[:, None].long()],
                dim=1,
            )

    return (
        selected[0].detach().cpu().tolist(),
        encoded_nodes,
    )


def rollout_from_prefix(
    model,
    encoded_nodes: torch.Tensor,
    selected_prefix: Sequence[int],
    horizon: int,
) -> Tuple[List[int], List[torch.Tensor]]:
    selected = torch.tensor(
        [list(selected_prefix)],
        dtype=torch.long,
        device=encoded_nodes.device,
    )

    future_nodes: List[int] = []
    distributions: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(horizon):
            probs = call_decoder(
                model=model,
                encoded_nodes=encoded_nodes,
                selected_node_list=selected,
            )

            distributions.append(probs[0].detach().clone())

            next_node = int(probs[0].argmax().item())
            future_nodes.append(next_node)

            selected = torch.cat(
                [
                    selected,
                    torch.tensor(
                        [[next_node]],
                        dtype=torch.long,
                        device=selected.device,
                    ),
                ],
                dim=1,
            )

    return future_nodes, distributions


# =============================================================================
# Step selection
# =============================================================================

def choose_selected_counts(
    problem_size: int,
    minimum_visited: int,
    steps_per_instance: int,
    rollout_horizon: int,
) -> List[int]:
    first_count = max(2, minimum_visited)
    last_count = problem_size - rollout_horizon

    if first_count > last_count or steps_per_instance <= 0:
        return []

    valid_counts = list(range(first_count, last_count + 1))

    if len(valid_counts) <= steps_per_instance:
        return valid_counts

    positions = np.linspace(
        0,
        len(valid_counts) - 1,
        num=steps_per_instance,
    )

    selected_positions = sorted(
        {int(round(position)) for position in positions}
    )

    return [
        valid_counts[position]
        for position in selected_positions
    ]


# =============================================================================
# Geometry helpers
# =============================================================================

def vector_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = vector_norm(vector)

    if norm <= EPS:
        return None

    return vector / norm


def cosine_alignment(
    first: Optional[np.ndarray],
    second: Optional[np.ndarray],
) -> float:
    if first is None or second is None:
        return float("nan")

    return float(
        np.clip(
            np.dot(first, second),
            -1.0,
            1.0,
        )
    )


def unsigned_angle_deg(
    first: Optional[np.ndarray],
    second: Optional[np.ndarray],
) -> float:
    cosine = cosine_alignment(first, second)

    if not np.isfinite(cosine):
        return float("nan")

    return float(math.degrees(math.acos(cosine)))


def signed_angle_deg(
    first: Optional[np.ndarray],
    second: Optional[np.ndarray],
) -> float:
    if first is None or second is None:
        return float("nan")

    dot = float(np.dot(first, second))
    cross = float(
        first[0] * second[1]
        - first[1] * second[0]
    )

    return float(math.degrees(math.atan2(cross, dot)))


def wrap_angle_deg(angle: float) -> float:
    if not np.isfinite(angle):
        return float("nan")

    return float((angle + 180.0) % 360.0 - 180.0)


def direction_from_current(
    coords: np.ndarray,
    current_node: int,
    target_node: int,
) -> Optional[np.ndarray]:
    return unit_vector(
        coords[target_node] - coords[current_node]
    )


def path_length_from_current(
    coords: np.ndarray,
    current_node: int,
    future_nodes: Sequence[int],
) -> float:
    total = 0.0
    previous = current_node

    for node in future_nodes:
        total += vector_norm(
            coords[int(node)] - coords[int(previous)]
        )
        previous = int(node)

    return float(total)


# =============================================================================
# Donor selection
# =============================================================================

def compute_donor_geometries(
    coords: np.ndarray,
    start_node: int,
    current_node: int,
    eligible_donors: Sequence[int],
    geometry_epsilon: float,
) -> Tuple[List[DonorGeometry], float, List[int]]:
    """
    Compute donor geometry without terminating on duplicate coordinates.

    A zero Current-to-Start distance makes the reference direction undefined.
    The caller skips and logs that state. Donors whose coordinates coincide
    with the current node are removed from the candidate geometry list.
    """
    start_vector = (
        coords[start_node] - coords[current_node]
    )
    start_distance = vector_norm(start_vector)

    if start_distance <= geometry_epsilon:
        return [], float(start_distance), []

    start_direction = unit_vector(start_vector)
    geometries: List[DonorGeometry] = []
    zero_distance_donor_nodes: List[int] = []

    for donor_node in eligible_donors:
        donor_vector = (
            coords[int(donor_node)] - coords[current_node]
        )
        donor_distance = vector_norm(donor_vector)

        if donor_distance <= geometry_epsilon:
            zero_distance_donor_nodes.append(
                int(donor_node)
            )
            continue

        donor_direction = unit_vector(donor_vector)
        ratio = donor_distance / start_distance

        geometries.append(
            DonorGeometry(
                donor_node=int(donor_node),
                distance_from_current=float(donor_distance),
                distance_ratio_to_start=float(ratio),
                absolute_distance_difference=float(
                    abs(donor_distance - start_distance)
                ),
                absolute_log_distance_ratio=float(
                    abs(math.log(max(ratio, EPS)))
                ),
                unsigned_angle_to_start_deg=unsigned_angle_deg(
                    start_direction,
                    donor_direction,
                ),
                signed_angle_from_start_deg=signed_angle_deg(
                    start_direction,
                    donor_direction,
                ),
            )
        )

    return (
        geometries,
        float(start_distance),
        zero_distance_donor_nodes,
    )


def select_donors(
    geometries: Sequence[DonorGeometry],
    distance_match_tolerance: float,
    same_direction_max_angle_deg: float,
    different_distance_ratio_low: float,
    different_distance_ratio_high: float,
    rng: random.Random,
) -> List[DonorChoice]:
    choices: List[DonorChoice] = []

    distance_matched = [
        geometry
        for geometry in geometries
        if abs(geometry.distance_ratio_to_start - 1.0)
        <= distance_match_tolerance
    ]

    if distance_matched:
        selected = max(
            distance_matched,
            key=lambda geometry: (
                geometry.unsigned_angle_to_start_deg,
                -abs(geometry.distance_ratio_to_start - 1.0),
                -geometry.donor_node,
            ),
        )
        choices.append(
            DonorChoice(
                category=CATEGORY_DISTANCE_MATCHED_MAX_ANGLE,
                available=True,
                donor_node=selected.donor_node,
                reason="selected",
                category_candidate_count=len(distance_matched),
            )
        )
    else:
        choices.append(
            DonorChoice(
                category=CATEGORY_DISTANCE_MATCHED_MAX_ANGLE,
                available=False,
                donor_node=None,
                reason=(
                    "no visited donor passed the distance-match threshold"
                ),
                category_candidate_count=0,
            )
        )

    if geometries:
        selected = max(
            geometries,
            key=lambda geometry: (
                geometry.unsigned_angle_to_start_deg,
                -geometry.donor_node,
            ),
        )
        choices.append(
            DonorChoice(
                category=CATEGORY_MAX_ANGLE_UNCONSTRAINED,
                available=True,
                donor_node=selected.donor_node,
                reason="selected",
                category_candidate_count=len(geometries),
            )
        )
    else:
        choices.append(
            DonorChoice(
                category=CATEGORY_MAX_ANGLE_UNCONSTRAINED,
                available=False,
                donor_node=None,
                reason="eligible visited-donor pool was empty",
                category_candidate_count=0,
            )
        )

    same_direction_different_distance = [
        geometry
        for geometry in geometries
        if (
            geometry.unsigned_angle_to_start_deg
            <= same_direction_max_angle_deg
        )
        and (
            geometry.distance_ratio_to_start
            < different_distance_ratio_low
            or geometry.distance_ratio_to_start
            > different_distance_ratio_high
        )
    ]

    if same_direction_different_distance:
        selected = max(
            same_direction_different_distance,
            key=lambda geometry: (
                geometry.absolute_log_distance_ratio,
                -geometry.unsigned_angle_to_start_deg,
                -geometry.donor_node,
            ),
        )
        choices.append(
            DonorChoice(
                category=(
                    CATEGORY_SAME_DIRECTION_DIFFERENT_DISTANCE
                ),
                available=True,
                donor_node=selected.donor_node,
                reason="selected",
                category_candidate_count=len(
                    same_direction_different_distance
                ),
            )
        )
    else:
        choices.append(
            DonorChoice(
                category=(
                    CATEGORY_SAME_DIRECTION_DIFFERENT_DISTANCE
                ),
                available=False,
                donor_node=None,
                reason=(
                    "no visited donor passed both same-direction "
                    "and different-distance thresholds"
                ),
                category_candidate_count=0,
            )
        )

    previously_selected = {
        choice.donor_node
        for choice in choices
        if choice.available and choice.donor_node is not None
    }

    random_pool = [
        geometry
        for geometry in geometries
        if geometry.donor_node not in previously_selected
    ]

    if not random_pool:
        random_pool = list(geometries)

    if random_pool:
        selected = rng.choice(random_pool)
        choices.append(
            DonorChoice(
                category=CATEGORY_RANDOM_VISITED_CONTROL,
                available=True,
                donor_node=selected.donor_node,
                reason="selected",
                category_candidate_count=len(random_pool),
            )
        )
    else:
        choices.append(
            DonorChoice(
                category=CATEGORY_RANDOM_VISITED_CONTROL,
                available=False,
                donor_node=None,
                reason="eligible visited-donor pool was empty",
                category_candidate_count=0,
            )
        )

    return choices


# =============================================================================
# Probability metrics
# =============================================================================

def build_candidate_mask(
    problem_size: int,
    selected_prefix: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(
        problem_size,
        dtype=torch.bool,
        device=device,
    )

    mask[
        torch.tensor(
            list(selected_prefix),
            dtype=torch.long,
            device=device,
        )
    ] = False

    return mask


def normalize_candidate_distribution(
    raw_probs: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    values = raw_probs.detach().float().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)

    candidate_values = np.clip(
        values[mask_np],
        0.0,
        None,
    )

    total = float(candidate_values.sum())

    if total <= EPS:
        candidate_values = np.full(
            candidate_values.shape,
            1.0 / max(candidate_values.size, 1),
            dtype=np.float64,
        )
    else:
        candidate_values = candidate_values / total

    distribution = np.zeros_like(
        values,
        dtype=np.float64,
    )
    distribution[mask_np] = candidate_values

    return distribution


def entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0]
    return float(
        -np.sum(positive * np.log(positive))
    )


def js_divergence(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    midpoint = 0.5 * (first + second)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        valid = p > 0
        return float(
            np.sum(
                p[valid]
                * np.log(
                    p[valid]
                    / np.clip(q[valid], EPS, None)
                )
            )
        )

    return float(
        0.5 * kl(first, midpoint)
        + 0.5 * kl(second, midpoint)
    )


def total_variation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        0.5 * np.abs(first - second).sum()
    )


def top_k_jaccard(
    first: np.ndarray,
    second: np.ndarray,
    candidate_nodes: Sequence[int],
    k: int,
) -> float:
    effective_k = min(k, len(candidate_nodes))

    first_nodes = set(
        sorted(
            candidate_nodes,
            key=lambda node: (
                first[int(node)],
                -int(node),
            ),
            reverse=True,
        )[:effective_k]
    )
    second_nodes = set(
        sorted(
            candidate_nodes,
            key=lambda node: (
                second[int(node)],
                -int(node),
            ),
            reverse=True,
        )[:effective_k]
    )

    union = first_nodes | second_nodes

    if not union:
        return float("nan")

    return float(
        len(first_nodes & second_nodes) / len(union)
    )


def expected_direction(
    distribution: np.ndarray,
    coords: np.ndarray,
    current_node: int,
    candidate_nodes: Sequence[int],
) -> Tuple[Optional[np.ndarray], float]:
    weighted = np.zeros(2, dtype=np.float64)

    for node in candidate_nodes:
        direction = direction_from_current(
            coords,
            current_node,
            int(node),
        )

        if direction is not None:
            weighted += distribution[int(node)] * direction

    concentration = vector_norm(weighted)
    return unit_vector(weighted), concentration


# =============================================================================
# Patch evaluation
# =============================================================================

def first_rollout_divergence(
    clean_nodes: Sequence[int],
    patched_nodes: Sequence[int],
) -> int:
    for horizon, (clean_node, patched_node) in enumerate(
        zip(clean_nodes, patched_nodes),
        start=1,
    ):
        if int(clean_node) != int(patched_node):
            return horizon

    return 0


def evaluate_patch(
    model,
    coords: np.ndarray,
    encoded_nodes: torch.Tensor,
    selected_prefix: Sequence[int],
    donor_node: int,
    clean_rollout: Sequence[int],
    clean_raw_distributions: Sequence[torch.Tensor],
    rollout_horizon: int,
    top_k_overlap: int,
) -> Dict[str, Any]:
    problem_size = int(coords.shape[0])
    start_node = int(selected_prefix[0])
    current_node = int(selected_prefix[-1])

    patched_encoded = encoded_nodes.clone()
    patched_encoded[
        0,
        start_node,
        :,
    ] = encoded_nodes[
        0,
        int(donor_node),
        :,
    ]

    (
        patched_rollout,
        patched_raw_distributions,
    ) = rollout_from_prefix(
        model=model,
        encoded_nodes=patched_encoded,
        selected_prefix=selected_prefix,
        horizon=rollout_horizon,
    )

    mask = build_candidate_mask(
        problem_size=problem_size,
        selected_prefix=selected_prefix,
        device=encoded_nodes.device,
    )

    candidate_nodes = (
        torch.where(mask)[0]
        .detach()
        .cpu()
        .tolist()
    )

    clean_distribution = normalize_candidate_distribution(
        clean_raw_distributions[0],
        mask,
    )
    patched_distribution = normalize_candidate_distribution(
        patched_raw_distributions[0],
        mask,
    )

    clean_next = int(clean_rollout[0])
    patched_next = int(patched_rollout[0])

    start_direction = direction_from_current(
        coords,
        current_node,
        start_node,
    )
    donor_direction = direction_from_current(
        coords,
        current_node,
        int(donor_node),
    )

    (
        clean_expected_direction,
        clean_expected_concentration,
    ) = expected_direction(
        clean_distribution,
        coords,
        current_node,
        candidate_nodes,
    )
    (
        patched_expected_direction,
        patched_expected_concentration,
    ) = expected_direction(
        patched_distribution,
        coords,
        current_node,
        candidate_nodes,
    )

    input_rotation = signed_angle_deg(
        start_direction,
        donor_direction,
    )
    output_rotation = signed_angle_deg(
        clean_expected_direction,
        patched_expected_direction,
    )

    if (
        np.isfinite(input_rotation)
        and abs(input_rotation) > 1e-8
        and np.isfinite(output_rotation)
    ):
        rotation_ratio = output_rotation / input_rotation
        rotation_sign_agreement = int(
            np.sign(output_rotation)
            == np.sign(input_rotation)
        )
        rotation_tracking_error = wrap_angle_deg(
            output_rotation - input_rotation
        )
    else:
        rotation_ratio = float("nan")
        rotation_sign_agreement = 0
        rotation_tracking_error = float("nan")

    clean_expected_angle_to_donor = unsigned_angle_deg(
        clean_expected_direction,
        donor_direction,
    )
    patched_expected_angle_to_donor = unsigned_angle_deg(
        patched_expected_direction,
        donor_direction,
    )

    result: Dict[str, Any] = {
        "clean_next_node": clean_next,
        "patched_next_node": patched_next,
        "next_node_changed": int(clean_next != patched_next),
        "clean_probability_of_clean_next": float(
            clean_distribution[clean_next]
        ),
        "patched_probability_of_clean_next": float(
            patched_distribution[clean_next]
        ),
        "clean_probability_of_patched_next": float(
            clean_distribution[patched_next]
        ),
        "patched_probability_of_patched_next": float(
            patched_distribution[patched_next]
        ),
        "clean_next_probability_drop": float(
            clean_distribution[clean_next]
            - patched_distribution[clean_next]
        ),
        "js_divergence": js_divergence(
            clean_distribution,
            patched_distribution,
        ),
        "total_variation_distance": total_variation(
            clean_distribution,
            patched_distribution,
        ),
        "clean_entropy": entropy(clean_distribution),
        "patched_entropy": entropy(patched_distribution),
        "entropy_change": (
            entropy(patched_distribution)
            - entropy(clean_distribution)
        ),
        "top_k_jaccard": top_k_jaccard(
            clean_distribution,
            patched_distribution,
            candidate_nodes,
            top_k_overlap,
        ),
        "clean_expected_concentration": (
            clean_expected_concentration
        ),
        "patched_expected_concentration": (
            patched_expected_concentration
        ),
        "clean_expected_angle_to_start_deg": (
            unsigned_angle_deg(
                clean_expected_direction,
                start_direction,
            )
        ),
        "patched_expected_angle_to_start_deg": (
            unsigned_angle_deg(
                patched_expected_direction,
                start_direction,
            )
        ),
        "clean_expected_angle_to_donor_deg": (
            clean_expected_angle_to_donor
        ),
        "patched_expected_angle_to_donor_deg": (
            patched_expected_angle_to_donor
        ),
        "steering_gain_expected_to_donor_deg": (
            clean_expected_angle_to_donor
            - patched_expected_angle_to_donor
        ),
        "steering_gain_expected_to_donor_cos": (
            cosine_alignment(
                patched_expected_direction,
                donor_direction,
            )
            - cosine_alignment(
                clean_expected_direction,
                donor_direction,
            )
        ),
        "input_rotation_signed_deg": input_rotation,
        "output_rotation_signed_deg": output_rotation,
        "rotation_ratio": rotation_ratio,
        "rotation_sign_agreement": rotation_sign_agreement,
        "rotation_tracking_error_deg": (
            rotation_tracking_error
        ),
        "clean_rollout_nodes": json.dumps(
            list(clean_rollout)
        ),
        "patched_rollout_nodes": json.dumps(
            list(patched_rollout)
        ),
        "first_rollout_divergence_horizon": (
            first_rollout_divergence(
                clean_rollout,
                patched_rollout,
            )
        ),
        "clean_rollout_path_length": (
            path_length_from_current(
                coords,
                current_node,
                clean_rollout,
            )
        ),
        "patched_rollout_path_length": (
            path_length_from_current(
                coords,
                current_node,
                patched_rollout,
            )
        ),
        "rollout_path_length_change": (
            path_length_from_current(
                coords,
                current_node,
                patched_rollout,
            )
            - path_length_from_current(
                coords,
                current_node,
                clean_rollout,
            )
        ),
        "start_to_donor_encoder_l2": float(
            torch.norm(
                encoded_nodes[0, start_node, :]
                - encoded_nodes[0, int(donor_node), :]
            ).item()
        ),
        "start_to_donor_encoder_cosine": float(
            F.cosine_similarity(
                encoded_nodes[
                    0,
                    start_node,
                    :,
                ].unsqueeze(0),
                encoded_nodes[
                    0,
                    int(donor_node),
                    :,
                ].unsqueeze(0),
                dim=1,
            )[0].item()
        ),
    }

    for horizon_index in range(rollout_horizon):
        horizon = horizon_index + 1
        clean_node = int(clean_rollout[horizon_index])
        patched_node = int(
            patched_rollout[horizon_index]
        )

        clean_direction = direction_from_current(
            coords,
            current_node,
            clean_node,
        )
        patched_direction = direction_from_current(
            coords,
            current_node,
            patched_node,
        )

        clean_angle_start = unsigned_angle_deg(
            clean_direction,
            start_direction,
        )
        patched_angle_start = unsigned_angle_deg(
            patched_direction,
            start_direction,
        )
        clean_angle_donor = unsigned_angle_deg(
            clean_direction,
            donor_direction,
        )
        patched_angle_donor = unsigned_angle_deg(
            patched_direction,
            donor_direction,
        )

        result[f"clean_h{horizon}_node"] = clean_node
        result[f"patched_h{horizon}_node"] = (
            patched_node
        )
        result[
            f"clean_h{horizon}_angle_to_start_deg"
        ] = clean_angle_start
        result[
            f"patched_h{horizon}_angle_to_start_deg"
        ] = patched_angle_start
        result[
            f"clean_h{horizon}_angle_to_donor_deg"
        ] = clean_angle_donor
        result[
            f"patched_h{horizon}_angle_to_donor_deg"
        ] = patched_angle_donor
        result[
            f"steering_gain_h{horizon}_to_donor_deg"
        ] = clean_angle_donor - patched_angle_donor
        result[
            f"alignment_change_h{horizon}_to_start_cos"
        ] = (
            cosine_alignment(
                patched_direction,
                start_direction,
            )
            - cosine_alignment(
                clean_direction,
                start_direction,
            )
        )
        result[
            f"steering_gain_h{horizon}_to_donor_cos"
        ] = (
            cosine_alignment(
                patched_direction,
                donor_direction,
            )
            - cosine_alignment(
                clean_direction,
                donor_direction,
            )
        )

    result["steering_gain_next_to_donor_deg"] = (
        result["steering_gain_h1_to_donor_deg"]
    )
    result["steering_gain_h4_to_donor_deg"] = (
        result[
            f"steering_gain_h{rollout_horizon}_to_donor_deg"
        ]
    )

    return result


# =============================================================================
# CSV output
# =============================================================================

def collect_fieldnames(
    rows: Sequence[Dict[str, Any]],
) -> List[str]:
    return sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = collect_fieldnames(rows)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        with gzip.open(
            path,
            "wt",
            encoding="utf-8",
        ):
            pass
        return

    fieldnames = collect_fieldnames(rows)

    with gzip.open(
        path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        Tuple[int, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                int(row["problem_size"]),
                str(row["category"]),
            )
        ].append(row)

    metrics = [
        "steering_gain_next_to_donor_deg",
        "steering_gain_h4_to_donor_deg",
        "steering_gain_expected_to_donor_deg",
        "steering_gain_expected_to_donor_cos",
        "next_node_changed",
        "js_divergence",
        "total_variation_distance",
        "rotation_sign_agreement",
        "rollout_path_length_change",
    ]

    output: List[Dict[str, Any]] = []

    for (problem_size, category), group_rows in sorted(
        grouped.items()
    ):
        summary_row: Dict[str, Any] = {
            "problem_size": problem_size,
            "category": category,
            "num_patches": len(group_rows),
        }

        for metric in metrics:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group_rows
                    if metric in row
                    and row[metric] != ""
                    and np.isfinite(float(row[metric]))
                ],
                dtype=np.float64,
            )

            if values.size == 0:
                summary_row[f"{metric}_mean"] = (
                    float("nan")
                )
                summary_row[f"{metric}_median"] = (
                    float("nan")
                )
                summary_row[
                    f"{metric}_positive_rate"
                ] = float("nan")
            else:
                summary_row[f"{metric}_mean"] = float(
                    values.mean()
                )
                summary_row[
                    f"{metric}_median"
                ] = float(np.median(values))
                summary_row[
                    f"{metric}_positive_rate"
                ] = float(np.mean(values > 0))

        output.append(summary_row)

    return output


def save_size_outputs(
    size_dir: Path,
    instance_rows: Sequence[Dict[str, Any]],
    state_rows: Sequence[Dict[str, Any]],
    skipped_state_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    selection_rows: Sequence[Dict[str, Any]],
    patch_rows: Sequence[Dict[str, Any]],
) -> None:
    """
    Persist one problem-size result set.

    This function is also called periodically during execution so a later
    failure does not erase all completed instances for the current size.
    """
    write_csv(
        size_dir / "selected_instances.csv",
        instance_rows,
    )
    write_csv(
        size_dir / "selected_states.csv",
        state_rows,
    )
    write_csv(
        size_dir / "skipped_states.csv",
        skipped_state_rows,
    )
    write_gzip_csv(
        size_dir / "donor_candidates.csv.gz",
        candidate_rows,
    )
    write_csv(
        size_dir / "donor_selections.csv",
        selection_rows,
    )
    write_csv(
        size_dir / "patch_results.csv",
        patch_rows,
    )
    write_csv(
        size_dir / "summary_by_category.csv",
        summarize(patch_rows),
    )


# =============================================================================
# Main
# =============================================================================

def prepare_results_directory(
    results_root: Path,
    problem_sizes: Sequence[int],
    overwrite: bool,
) -> Path:
    """
    Prepare direct output under results_root without creating a run subfolder.

    Only files and N-size directories created by this experiment are removed
    when overwrite is enabled. Existing historical run subdirectories are not
    deleted.
    """
    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = [
        "run_config.json",
        "dataset_inventory.csv",
        "selected_dataset_source_by_size.csv",
        "selected_instances_all_sizes.csv",
        "selected_states_all_sizes.csv",
        "skipped_states_all_sizes.csv",
        "donor_candidates_all_sizes.csv.gz",
        "donor_selections_all_sizes.csv",
        "patch_results_all_sizes.csv",
        "summary_all_sizes.csv",
        "COMPLETED.txt",
        "IN_PROGRESS.txt",
    ]

    generated_paths: List[Path] = [
        results_root / filename
        for filename in generated_files
    ]
    generated_paths.extend(
        results_root / f"N{int(problem_size)}"
        for problem_size in problem_sizes
    )

    existing_generated_paths = [
        path
        for path in generated_paths
        if path.exists()
    ]

    if existing_generated_paths and not overwrite:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing_generated_paths
        )
        raise FileExistsError(
            "Direct result files already exist under results_root.\n"
            "Use --overwrite to replace only this experiment's direct outputs:\n"
            f"{formatted}"
        )

    if overwrite:
        for path in existing_generated_paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    return results_root


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    utilities = import_local_utilities(args.utils_root)

    device = utilities["setup_torch_device"](
        cuda_device_num=args.cuda_device_num,
        set_default_device=True,
        legacy_cuda_default_tensor_type=False,
    )

    Model = import_lehd_model(args.lehd_root)

    model, model_params = build_model(
        Model=Model,
        checkpoint_path=args.checkpoint_path,
        device=device,
        load_checkpoint=utilities["load_checkpoint"],
        get_state_dict_from_checkpoint=(
            utilities["get_state_dict_from_checkpoint"]
        ),
    )

    max_instances_map = parse_int_map(
        args.max_instances_map
    )
    min_visited_map = parse_int_map(
        args.min_visited_map
    )
    steps_per_instance_map = parse_int_map(
        args.steps_per_instance_map
    )

    if args.dataset_dirs is not None:
        dataset_directories = validate_explicit_dataset_directories(
            args.dataset_dirs
        )
    else:
        dataset_directories = discover_dataset_directories(args.data_root)

    (
        by_size_and_directory,
        inventory_rows,
    ) = scan_datasets(
        dataset_directories=dataset_directories,
        problem_sizes=args.problem_sizes,
        load_coordinate_list=(
            utilities["load_coordinate_list"]
        ),
        group_indices_by_problem_size=(
            utilities[
                "group_indices_by_problem_size"
            ]
        ),
    )

    (
        selected_by_size,
        source_rows,
    ) = choose_source_and_instances(
        by_size_and_directory=(
            by_size_and_directory
        ),
        problem_sizes=args.problem_sizes,
        max_instances_map=max_instances_map,
        instance_selection=args.instance_selection,
        seed=args.seed,
    )

    run_dir = prepare_results_directory(
        results_root=args.results_root,
        problem_sizes=args.problem_sizes,
        overwrite=args.overwrite,
    )

    (
        run_dir / "IN_PROGRESS.txt"
    ).write_text(
        "Experiment is currently running.\n",
        encoding="utf-8",
    )

    config = {
        "data_root": (
            str(args.data_root.expanduser().resolve())
            if args.data_root is not None
            else None
        ),
        "dataset_dirs": (
            [str(path.expanduser().resolve()) for path in args.dataset_dirs]
            if args.dataset_dirs is not None
            else None
        ),
        "discovered_dataset_directories": [
            str(path.resolve())
            for path in dataset_directories
        ],
        "results_root": str(
            args.results_root.resolve()
        ),
        "run_dir": str(run_dir.resolve()),
        "output_layout": (
            "direct_results_root_without_run_subdirectory"
        ),
        "lehd_root": str(args.lehd_root.resolve()),
        "checkpoint_path": str(
            args.checkpoint_path.resolve()
        ),
        "utils_root": str(args.utils_root.resolve()),
        "problem_sizes": args.problem_sizes,
        "max_instances_map": max_instances_map,
        "min_visited_map": min_visited_map,
        "steps_per_instance_map": (
            steps_per_instance_map
        ),
        "distance_match_tolerance": (
            args.distance_match_tolerance
        ),
        "same_direction_max_angle_deg": (
            args.same_direction_max_angle_deg
        ),
        "different_distance_ratio_low": (
            args.different_distance_ratio_low
        ),
        "different_distance_ratio_high": (
            args.different_distance_ratio_high
        ),
        "exclude_previous_node": (
            args.exclude_previous_node
        ),
        "rollout_horizon": args.rollout_horizon,
        "top_k_overlap": args.top_k_overlap,
        "geometry_epsilon": args.geometry_epsilon,
        "save_every_instances": args.save_every_instances,
        "seed": args.seed,
        "model_params": model_params,
        "patch": (
            "patched_encoded[:, start_node, :] = "
            "encoded[:, donor_node, :]"
        ),
    }

    (
        run_dir / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        run_dir / "dataset_inventory.csv",
        inventory_rows,
    )
    write_csv(
        run_dir
        / "selected_dataset_source_by_size.csv",
        source_rows,
    )

    all_instance_rows: List[
        Dict[str, Any]
    ] = []
    all_state_rows: List[
        Dict[str, Any]
    ] = []
    all_skipped_state_rows: List[
        Dict[str, Any]
    ] = []
    all_candidate_rows: List[
        Dict[str, Any]
    ] = []
    all_selection_rows: List[
        Dict[str, Any]
    ] = []
    all_patch_rows: List[
        Dict[str, Any]
    ] = []

    for problem_size in args.problem_sizes:
        records = selected_by_size.get(
            int(problem_size),
            [],
        )

        if not records:
            continue

        size_dir = run_dir / f"N{problem_size}"
        size_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        minimum_visited = min_visited_map.get(
            int(problem_size),
            max(8, int(math.ceil(0.2 * problem_size))),
        )
        steps_per_instance = (
            steps_per_instance_map.get(
                int(problem_size),
                10,
            )
        )

        size_instance_rows: List[
            Dict[str, Any]
        ] = []
        size_state_rows: List[
            Dict[str, Any]
        ] = []
        size_skipped_state_rows: List[
            Dict[str, Any]
        ] = []
        size_candidate_rows: List[
            Dict[str, Any]
        ] = []
        size_selection_rows: List[
            Dict[str, Any]
        ] = []
        size_patch_rows: List[
            Dict[str, Any]
        ] = []

        print(
            f"\n[N={problem_size}] instances={len(records)} | "
            f"minimum_visited={minimum_visited} | "
            f"steps_per_instance={steps_per_instance}",
            flush=True,
        )

        for selected_instance_order, record in enumerate(
            records
        ):
            coords_tensor = torch.tensor(
                record.coords,
                dtype=torch.float32,
                device=device,
            )

            clean_tour, encoded_nodes = (
                build_clean_greedy_tour(
                    model,
                    coords_tensor,
                )
            )

            selected_counts = choose_selected_counts(
                problem_size=int(problem_size),
                minimum_visited=minimum_visited,
                steps_per_instance=(
                    steps_per_instance
                ),
                rollout_horizon=(
                    args.rollout_horizon
                ),
            )

            instance_row = {
                "problem_size": int(problem_size),
                "selected_instance_order": (
                    selected_instance_order
                ),
                "source_directory": (
                    record.source_directory
                ),
                "source_directory_name": (
                    record.source_directory_name
                ),
                "source_file": record.source_file,
                "source_file_name": (
                    record.source_file_name
                ),
                "instance_index_in_file": (
                    record.instance_index_in_file
                ),
                "clean_tour": json.dumps(
                    clean_tour
                ),
                "selected_counts": json.dumps(
                    selected_counts
                ),
            }

            size_instance_rows.append(instance_row)
            all_instance_rows.append(instance_row)

            for selected_count in selected_counts:
                prefix = clean_tour[:selected_count]

                start_node = int(prefix[0])
                current_node = int(prefix[-1])
                previous_node = int(prefix[-2])

                common_state = {
                    "problem_size": int(problem_size),
                    "selected_instance_order": (
                        selected_instance_order
                    ),
                    "source_directory": (
                        record.source_directory
                    ),
                    "source_directory_name": (
                        record.source_directory_name
                    ),
                    "source_file": record.source_file,
                    "source_file_name": (
                        record.source_file_name
                    ),
                    "instance_index_in_file": (
                        record.instance_index_in_file
                    ),
                    "selected_count": selected_count,
                    "current_step_zero_based": (
                        selected_count - 1
                    ),
                    "tour_progress_fraction": (
                        selected_count / problem_size
                    ),
                    "start_node": start_node,
                    "previous_node": previous_node,
                    "current_node": current_node,
                }

                eligible_donors = [
                    int(node)
                    for node in prefix
                    if int(node) not in {
                        start_node,
                        current_node,
                    }
                ]

                if args.exclude_previous_node:
                    eligible_donors = [
                        node
                        for node in eligible_donors
                        if node != previous_node
                    ]

                (
                    donor_geometries,
                    start_distance,
                    zero_distance_donor_nodes,
                ) = compute_donor_geometries(
                    coords=record.coords,
                    start_node=start_node,
                    current_node=current_node,
                    eligible_donors=eligible_donors,
                    geometry_epsilon=(
                        args.geometry_epsilon
                    ),
                )

                if start_distance <= args.geometry_epsilon:
                    start_coord = record.coords[
                        start_node
                    ]
                    current_coord = record.coords[
                        current_node
                    ]

                    distances_to_start_coord = (
                        np.linalg.norm(
                            record.coords
                            - start_coord[None, :],
                            axis=1,
                        )
                    )
                    duplicate_coordinate_nodes = (
                        np.where(
                            distances_to_start_coord
                            <= args.geometry_epsilon
                        )[0]
                        .astype(int)
                        .tolist()
                    )

                    skipped_row = {
                        **common_state,
                        "state_status": "skipped",
                        "skip_reason": (
                            "undefined_start_direction_zero_current_to_start_distance"
                        ),
                        "selected_prefix": json.dumps(
                            prefix
                        ),
                        "distance_from_current_to_start": (
                            start_distance
                        ),
                        "start_x": float(
                            start_coord[0]
                        ),
                        "start_y": float(
                            start_coord[1]
                        ),
                        "current_x": float(
                            current_coord[0]
                        ),
                        "current_y": float(
                            current_coord[1]
                        ),
                        "duplicate_coordinate_nodes": (
                            json.dumps(
                                duplicate_coordinate_nodes
                            )
                        ),
                        "geometry_epsilon": (
                            args.geometry_epsilon
                        ),
                    }

                    size_skipped_state_rows.append(
                        skipped_row
                    )
                    all_skipped_state_rows.append(
                        skipped_row
                    )
                    size_state_rows.append(
                        skipped_row
                    )
                    all_state_rows.append(
                        skipped_row
                    )

                    print(
                        f"[N={problem_size}] skipped degenerate geometry | "
                        f"file={record.source_file_name} | "
                        f"instance={record.instance_index_in_file} | "
                        f"selected_count={selected_count} | "
                        f"start_node={start_node} | "
                        f"current_node={current_node} | "
                        f"distance={start_distance:.3e}",
                        flush=True,
                    )
                    continue

                state_rng = random.Random(
                    args.seed
                    + int(problem_size) * 1_000_003
                    + record.instance_index_in_file
                    * 10_007
                    + selected_count * 101
                )

                donor_choices = select_donors(
                    geometries=donor_geometries,
                    distance_match_tolerance=(
                        args.distance_match_tolerance
                    ),
                    same_direction_max_angle_deg=(
                        args.same_direction_max_angle_deg
                    ),
                    different_distance_ratio_low=(
                        args.different_distance_ratio_low
                    ),
                    different_distance_ratio_high=(
                        args.different_distance_ratio_high
                    ),
                    rng=state_rng,
                )

                geometry_by_node = {
                    geometry.donor_node: geometry
                    for geometry in donor_geometries
                }

                (
                    clean_rollout,
                    clean_raw_distributions,
                ) = rollout_from_prefix(
                    model=model,
                    encoded_nodes=encoded_nodes,
                    selected_prefix=prefix,
                    horizon=args.rollout_horizon,
                )

                state_row = {
                    **common_state,
                    "state_status": "processed",
                    "skip_reason": "",
                    "selected_prefix": json.dumps(prefix),
                    "clean_rollout_nodes": json.dumps(
                        clean_rollout
                    ),
                    "eligible_donor_count": len(
                        donor_geometries
                    ),
                    "zero_distance_donor_count": len(
                        zero_distance_donor_nodes
                    ),
                    "zero_distance_donor_nodes": json.dumps(
                        zero_distance_donor_nodes
                    ),
                    "available_categories": json.dumps(
                        [
                            choice.category
                            for choice in donor_choices
                            if choice.available
                        ]
                    ),
                    "missing_categories": json.dumps(
                        [
                            choice.category
                            for choice in donor_choices
                            if not choice.available
                        ]
                    ),
                }

                size_state_rows.append(state_row)
                all_state_rows.append(state_row)

                for geometry in donor_geometries:
                    candidate_row = {
                        **common_state,
                        "donor_node": (
                            geometry.donor_node
                        ),
                        "distance_from_current_to_start": (
                            start_distance
                        ),
                        "distance_from_current_to_donor": (
                            geometry.distance_from_current
                        ),
                        "distance_ratio_to_start": (
                            geometry.distance_ratio_to_start
                        ),
                        "absolute_distance_difference": (
                            geometry.absolute_distance_difference
                        ),
                        "absolute_log_distance_ratio": (
                            geometry.absolute_log_distance_ratio
                        ),
                        "unsigned_angle_to_start_deg": (
                            geometry.unsigned_angle_to_start_deg
                        ),
                        "signed_angle_from_start_deg": (
                            geometry.signed_angle_from_start_deg
                        ),
                        "passes_distance_match": int(
                            abs(
                                geometry.distance_ratio_to_start
                                - 1.0
                            )
                            <= args.distance_match_tolerance
                        ),
                        "passes_same_direction": int(
                            geometry.unsigned_angle_to_start_deg
                            <= args.same_direction_max_angle_deg
                        ),
                        "passes_different_distance": int(
                            geometry.distance_ratio_to_start
                            < args.different_distance_ratio_low
                            or geometry.distance_ratio_to_start
                            > args.different_distance_ratio_high
                        ),
                    }

                    size_candidate_rows.append(candidate_row)
                    all_candidate_rows.append(candidate_row)

                for choice in donor_choices:
                    selection_row = {
                        **common_state,
                        "category": choice.category,
                        "available": int(
                            choice.available
                        ),
                        "donor_node": (
                            choice.donor_node
                            if choice.donor_node
                            is not None
                            else ""
                        ),
                        "reason": choice.reason,
                        "category_candidate_count": (
                            choice.category_candidate_count
                        ),
                        "eligible_donor_count": len(
                            donor_geometries
                        ),
                    }

                    if (
                        choice.available
                        and choice.donor_node
                        is not None
                    ):
                        geometry = geometry_by_node[
                            choice.donor_node
                        ]

                        selection_row.update(
                            {
                                "distance_from_current_to_start": (
                                    start_distance
                                ),
                                "distance_from_current_to_donor": (
                                    geometry.distance_from_current
                                ),
                                "distance_ratio_to_start": (
                                    geometry.distance_ratio_to_start
                                ),
                                "unsigned_angle_to_start_deg": (
                                    geometry.unsigned_angle_to_start_deg
                                ),
                                "signed_angle_from_start_deg": (
                                    geometry.signed_angle_from_start_deg
                                ),
                            }
                        )

                    size_selection_rows.append(
                        selection_row
                    )
                    all_selection_rows.append(
                        selection_row
                    )

                    if (
                        not choice.available
                        or choice.donor_node is None
                    ):
                        continue

                    geometry = geometry_by_node[
                        choice.donor_node
                    ]

                    patch_metrics = evaluate_patch(
                        model=model,
                        coords=record.coords,
                        encoded_nodes=encoded_nodes,
                        selected_prefix=prefix,
                        donor_node=choice.donor_node,
                        clean_rollout=clean_rollout,
                        clean_raw_distributions=(
                            clean_raw_distributions
                        ),
                        rollout_horizon=(
                            args.rollout_horizon
                        ),
                        top_k_overlap=(
                            args.top_k_overlap
                        ),
                    )

                    patch_row = {
                        **common_state,
                        "selected_prefix": json.dumps(
                            prefix
                        ),
                        "category": choice.category,
                        "donor_node": choice.donor_node,
                        "eligible_donor_count": len(
                            donor_geometries
                        ),
                        "category_candidate_count": (
                            choice.category_candidate_count
                        ),
                        "distance_from_current_to_start": (
                            start_distance
                        ),
                        "distance_from_current_to_donor": (
                            geometry.distance_from_current
                        ),
                        "distance_ratio_to_start": (
                            geometry.distance_ratio_to_start
                        ),
                        "absolute_distance_difference": (
                            geometry.absolute_distance_difference
                        ),
                        "absolute_log_distance_ratio": (
                            geometry.absolute_log_distance_ratio
                        ),
                        "unsigned_angle_to_start_deg": (
                            geometry.unsigned_angle_to_start_deg
                        ),
                        "signed_angle_from_start_deg": (
                            geometry.signed_angle_from_start_deg
                        ),
                    }
                    patch_row.update(patch_metrics)

                    size_patch_rows.append(patch_row)
                    all_patch_rows.append(patch_row)

            print(
                f"[N={problem_size}] completed "
                f"{selected_instance_order + 1}/{len(records)} | "
                f"{record.source_file_name} | "
                f"instance={record.instance_index_in_file}",
                flush=True,
            )

            completed_count = (
                selected_instance_order + 1
            )
            should_checkpoint = (
                completed_count % max(
                    args.save_every_instances,
                    1,
                )
                == 0
                or completed_count == len(records)
            )

            if should_checkpoint:
                save_size_outputs(
                    size_dir=size_dir,
                    instance_rows=size_instance_rows,
                    state_rows=size_state_rows,
                    skipped_state_rows=(
                        size_skipped_state_rows
                    ),
                    candidate_rows=size_candidate_rows,
                    selection_rows=size_selection_rows,
                    patch_rows=size_patch_rows,
                )

                (
                    size_dir / "progress.json"
                ).write_text(
                    json.dumps(
                        {
                            "problem_size": int(
                                problem_size
                            ),
                            "completed_instances": (
                                completed_count
                            ),
                            "total_instances": len(
                                records
                            ),
                            "skipped_states": len(
                                size_skipped_state_rows
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        save_size_outputs(
            size_dir=size_dir,
            instance_rows=size_instance_rows,
            state_rows=size_state_rows,
            skipped_state_rows=(
                size_skipped_state_rows
            ),
            candidate_rows=size_candidate_rows,
            selection_rows=size_selection_rows,
            patch_rows=size_patch_rows,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(
        run_dir / "selected_instances_all_sizes.csv",
        all_instance_rows,
    )
    write_csv(
        run_dir / "selected_states_all_sizes.csv",
        all_state_rows,
    )
    write_csv(
        run_dir / "skipped_states_all_sizes.csv",
        all_skipped_state_rows,
    )
    write_gzip_csv(
        run_dir / "donor_candidates_all_sizes.csv.gz",
        all_candidate_rows,
    )
    write_csv(
        run_dir / "donor_selections_all_sizes.csv",
        all_selection_rows,
    )
    write_csv(
        run_dir / "patch_results_all_sizes.csv",
        all_patch_rows,
    )
    write_csv(
        run_dir / "summary_all_sizes.csv",
        summarize(all_patch_rows),
    )

    in_progress_file = run_dir / "IN_PROGRESS.txt"

    if in_progress_file.exists():
        in_progress_file.unlink()

    (
        run_dir / "COMPLETED.txt"
    ).write_text(
        "Experiment completed successfully.\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("Experiment completed successfully.")
    print("Results:", run_dir)
    print("=" * 90)


if __name__ == "__main__":
    main()

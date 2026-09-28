import os
import csv
import pickle
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 123) -> None:
    """
    Set random seeds for reproducibility.

    This affects:
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device helpers
# ============================================================

def setup_torch_device(
    cuda_device_num: int = 0,
    set_default_device: bool = True,
    legacy_cuda_default_tensor_type: bool = False,
) -> torch.device:
    """
    Configure and return the torch device.

    Some original NCO codebases create tensors internally without explicitly
    passing a device argument. This can cause CPU/CUDA mismatch errors.

    Parameters
    ----------
    cuda_device_num:
        CUDA device index.

    set_default_device:
        If True and PyTorch supports it, set torch default device.
        This is the recommended PyTorch 2.x style.

    legacy_cuda_default_tensor_type:
        If True, use torch.set_default_tensor_type(torch.cuda.FloatTensor)
        when CUDA is available. Some older codebases such as LEHD rely on this.
        This API is deprecated in newer PyTorch versions, so use it only when needed.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device_num)
        device = torch.device("cuda", cuda_device_num)

        torch.set_default_dtype(torch.float32)

        if set_default_device and hasattr(torch, "set_default_device"):
            torch.set_default_device(device)
        elif legacy_cuda_default_tensor_type:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    else:
        device = torch.device("cpu")

        torch.set_default_dtype(torch.float32)

        if set_default_device and hasattr(torch, "set_default_device"):
            torch.set_default_device(device)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

    return device


def move_tensor_attrs_to_device(obj: Any, device: torch.device) -> Any:
    """
    Move all direct tensor attributes of an object to the target device.

    Many original Env and State objects store tensors as direct attributes.
    This helper avoids CPU/CUDA mismatch errors after env.reset(), env.pre_step(),
    and env.step().

    Notes
    -----
    This function only moves direct tensor attributes, not nested dictionaries
    or nested custom objects.
    """
    if obj is None:
        return obj

    if not hasattr(obj, "__dict__"):
        return obj

    for name, value in obj.__dict__.items():
        if torch.is_tensor(value):
            setattr(obj, name, value.to(device))

    return obj


def empty_cuda_cache_if_available() -> None:
    """
    Empty CUDA cache safely if CUDA is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Dataset loading and coordinate normalization
# ============================================================

def normalize_coordinates(coords: Any) -> np.ndarray:
    """
    Convert one TSP instance to a NumPy array with shape (N, 2).

    Supported input shapes:
    - (N, 2)
    - (2, N)
    - (1, N, 2)

    Returns
    -------
    np.ndarray
        Coordinate array with shape (N, 2), dtype float32.
    """
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()

    arr = np.asarray(coords, dtype=np.float32)

    # Handle shape: (1, N, 2)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Standard shape: (N, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.astype(np.float32)

    # Transposed shape: (2, N)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T.astype(np.float32)

    raise ValueError(f"Invalid coordinate shape: {arr.shape}")


def extract_coordinates_from_sample(sample: Any) -> np.ndarray:
    """
    Extract coordinates from one dataset sample.

    Supported formats:
    - coords
    - (coords, tour)
    - {"coordinates": coords}
    - {"coords": coords}
    - {"nodes": coords}
    - {"problem": coords}
    - {"data": coords}
    """
    if isinstance(sample, dict):
        for key in ["coordinates", "coords", "nodes", "problem", "problems", "data"]:
            if key in sample:
                return normalize_coordinates(sample[key])

        raise ValueError(
            f"Cannot find coordinates in sample dict. Keys: {list(sample.keys())}"
        )

    # Common format: (coords, tour)
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        try:
            return normalize_coordinates(sample[0])
        except Exception:
            pass

    return normalize_coordinates(sample)


def load_coordinate_list(pkl_path: str) -> List[np.ndarray]:
    """
    Load one pickle dataset and return a list of coordinate arrays.

    Supported dataset formats:
    - list of coordinates
    - list of (coordinates, tour)
    - tuple: (all_coordinates, all_tours)
    - dict containing coordinates
    - numpy array with shape (B, N, 2)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        for key in ["coordinates", "coords", "problems", "nodes", "data"]:
            if key in data:
                source = data[key]
                break
        else:
            raise ValueError(
                f"Cannot find coordinate key in dataset dict. Keys: {list(data.keys())}"
            )

    elif isinstance(data, tuple) and len(data) == 2:
        # Usually stored as (coordinates, tours)
        source = data[0]

    else:
        source = data

    if torch.is_tensor(source):
        source = source.detach().cpu().numpy()

    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            samples = [source]
        elif source.ndim == 3:
            samples = [source[i] for i in range(source.shape[0])]
        else:
            raise ValueError(f"Invalid numpy dataset shape: {source.shape}")
    else:
        samples = list(source)

    coords_list = [extract_coordinates_from_sample(sample) for sample in samples]

    return coords_list


def list_pkl_files(data_dir: str, recursive: bool = False) -> List[str]:
    """
    Return sorted pickle files from a directory.

    Parameters
    ----------
    data_dir:
        Directory containing .pkl files.

    recursive:
        If True, search recursively.
    """
    if recursive:
        pattern = os.path.join(data_dir, "**", "*.pkl")
        return sorted(glob_recursive(pattern))

    pattern = os.path.join(data_dir, "*.pkl")
    return sorted([p for p in glob_simple(pattern)])


def glob_simple(pattern: str) -> List[str]:
    """
    Small wrapper around glob.glob to keep imports centralized.
    """
    import glob
    return glob.glob(pattern)


def glob_recursive(pattern: str) -> List[str]:
    """
    Small wrapper around glob.glob with recursive=True.
    """
    import glob
    return glob.glob(pattern, recursive=True)


def group_indices_by_problem_size(coords_list: Sequence[np.ndarray]) -> Dict[int, List[int]]:
    """
    Group instance indices by problem size.

    This allows one pickle file to contain instances with different node counts.
    """
    size_to_indices = defaultdict(list)

    for idx, coords in enumerate(coords_list):
        size_to_indices[int(coords.shape[0])].append(idx)

    return dict(size_to_indices)


# ============================================================
# Tour utilities
# ============================================================

def normalize_tour(tour: Any, problem_size: Optional[int] = None) -> np.ndarray:
    """
    Normalize a tour to a 1D NumPy int64 array.

    If problem_size is provided, this function also handles:
    - closed tours like [0, ..., 0]
    - one-based tours like [1, 2, ..., N]
    """
    if torch.is_tensor(tour):
        tour = tour.detach().cpu().numpy()

    arr = np.asarray(tour, dtype=np.int64).reshape(-1)

    if problem_size is not None:
        # Remove repeated closing node if present.
        if len(arr) == problem_size + 1 and arr[0] == arr[-1]:
            arr = arr[:-1]

        # Convert one-based indexing to zero-based indexing.
        if len(arr) > 0 and arr.min() >= 1 and arr.max() == problem_size:
            arr = arr - 1

    return arr.astype(np.int64)


def compute_tour_length(coords: Any, tour: Any) -> float:
    """
    Compute the closed-loop TSP tour length.

    The edge from the last node back to the first node is included.
    """
    coords = normalize_coordinates(coords)
    tour = normalize_tour(tour, problem_size=len(coords))

    total_length = 0.0

    for i in range(len(tour)):
        a = int(tour[i])
        b = int(tour[(i + 1) % len(tour)])
        total_length += np.linalg.norm(coords[a] - coords[b])

    return float(total_length)


def validate_tour(tour: Any, problem_size: int) -> Dict[str, Any]:
    """
    Validate whether a predicted tour is a valid TSP permutation.

    A valid tour must:
    - have length equal to problem_size
    - contain every node exactly once
    - contain only indices in [0, problem_size - 1]
    """
    tour = normalize_tour(tour, problem_size=problem_size)

    result = {
        "is_valid": True,
        "reason": "",
        "missing_nodes": [],
        "duplicate_nodes": [],
        "out_of_range_nodes": [],
    }

    if len(tour) != problem_size:
        result["is_valid"] = False
        result["reason"] = f"Length mismatch: got {len(tour)}, expected {problem_size}"
        return result

    out_of_range = tour[(tour < 0) | (tour >= problem_size)]

    if len(out_of_range) > 0:
        result["is_valid"] = False
        result["out_of_range_nodes"] = sorted(set(out_of_range.tolist()))

    unique_nodes, counts = np.unique(tour, return_counts=True)

    duplicates = unique_nodes[counts > 1]

    if len(duplicates) > 0:
        result["is_valid"] = False
        result["duplicate_nodes"] = duplicates.tolist()

    expected_nodes = set(range(problem_size))
    actual_nodes = set(tour.tolist())

    missing = sorted(expected_nodes - actual_nodes)

    if len(missing) > 0:
        result["is_valid"] = False
        result["missing_nodes"] = missing

    if not result["is_valid"] and result["reason"] == "":
        result["reason"] = "Tour is not a valid permutation of all nodes."

    return result


def build_plotfriendly_instances(
    coords_list: Sequence[np.ndarray],
    tours: Sequence[Any],
    method: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build plot-friendly output instances.

    Output format:
    [
        {
            "coords": coords,
            "tour": tour,
            "tour_length": length,
            "is_valid": True/False,
            ...
        },
        ...
    ]

    Returns
    -------
    output_instances:
        List of dictionaries compatible with plotting and comparison scripts.

    summary:
        Summary dictionary containing valid count, invalid count, and mean length.
    """
    if extra_fields is None:
        extra_fields = {}

    output_instances = []

    valid_count = 0
    invalid_count = 0
    all_lengths = []

    for coords, tour in zip(coords_list, tours):
        coords = normalize_coordinates(coords)
        tour = normalize_tour(tour, problem_size=len(coords))

        validation = validate_tour(tour, problem_size=len(coords))

        if len(validation["out_of_range_nodes"]) == 0 and len(tour) == len(coords):
            tour_length = compute_tour_length(coords, tour)
            all_lengths.append(tour_length)
        else:
            tour_length = float("nan")

        if validation["is_valid"]:
            valid_count += 1
        else:
            invalid_count += 1

        item = {
            "coords": coords,
            "tour": tour,
            "tour_length": float(tour_length),
            "is_valid": bool(validation["is_valid"]),
            "validation_reason": validation["reason"],
            "missing_nodes": validation["missing_nodes"],
            "duplicate_nodes": validation["duplicate_nodes"],
            "out_of_range_nodes": validation["out_of_range_nodes"],
            "problem_size": int(len(coords)),
            "start_node": int(tour[0]) if len(tour) > 0 else -1,
            "method": method,
        }

        item.update(extra_fields)

        output_instances.append(item)

    mean_length = (
        float(np.mean(all_lengths))
        if len(all_lengths) > 0
        else float("nan")
    )

    summary = {
        "num_valid_tours": valid_count,
        "num_invalid_tours": invalid_count,
        "mean_tour_length": mean_length,
    }

    return output_instances, summary


# ============================================================
# Checkpoint helpers
# ============================================================

def load_checkpoint(path: str, device: torch.device) -> Any:
    """
    Load a PyTorch checkpoint.

    weights_only=False is used because original research checkpoints often
    contain metadata or Python objects.
    """
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    return checkpoint


def get_state_dict_from_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Extract model state_dict from checkpoint.

    Supported:
    - {"model_state_dict": state_dict}
    - raw state_dict
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value

    return cleaned_state_dict


# ============================================================
# Saving helpers
# ============================================================

def save_pickle(obj: Any, path: str) -> None:
    """
    Save an object to a pickle file.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """
    Load an object from a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_summary_csv(
    rows: Sequence[Dict[str, Any]],
    csv_path: str,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """
    Save evaluation summary rows into a CSV file.

    If fieldnames is None, a default common evaluation schema is used.
    """
    if fieldnames is None:
        fieldnames = [
            "filename",
            "status",
            "num_instances",
            "problem_sizes",
            "num_valid_tours",
            "num_invalid_tours",
            "mean_tour_length",
            "elapsed_seconds",
            "elapsed_minutes",
            "batch_size",
            "method",
            "output_path",
            "error_message",
        ]

    parent = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(parent, exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def make_summary_row(
    filename: str,
    status: str,
    num_instances: Any = "",
    problem_sizes: Any = "",
    num_valid_tours: Any = "",
    num_invalid_tours: Any = "",
    mean_tour_length: Any = "",
    elapsed_seconds: float = 0.0,
    batch_size: Any = "",
    method: str = "",
    output_path: str = "",
    error_message: str = "",
) -> Dict[str, Any]:
    """
    Create a standard summary CSV row.
    """
    if isinstance(problem_sizes, (list, tuple)):
        problem_sizes = "|".join(map(str, problem_sizes))

    if isinstance(mean_tour_length, float):
        if np.isnan(mean_tour_length):
            mean_tour_length_out = ""
        else:
            mean_tour_length_out = round(mean_tour_length, 6)
    else:
        mean_tour_length_out = mean_tour_length

    return {
        "filename": filename,
        "status": status,
        "num_instances": num_instances,
        "problem_sizes": problem_sizes,
        "num_valid_tours": num_valid_tours,
        "num_invalid_tours": num_invalid_tours,
        "mean_tour_length": mean_tour_length_out,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "elapsed_minutes": round(float(elapsed_seconds) / 60.0, 6),
        "batch_size": batch_size,
        "method": method,
        "output_path": output_path,
        "error_message": error_message,
    }

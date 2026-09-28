from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class WorkerConfig:
    scale_factor: int


_WORKER_CONFIG: WorkerConfig | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve pickle-based Euclidean TSP datasets with Concorde."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where solved pickle files will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern used to select input files inside --input-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max((mp.cpu_count() or 1) - 1, 1),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of instances solved in each worker task.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=40,
        help="Restart each worker after this many tasks; use 0 to disable.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=1_000_000,
        help="Coordinate scaling factor used before integer Concorde input.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum instances solved from each file; omit to solve all.",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=30.0,
        help="Minimum interval between progress messages; use 0 to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing solved files instead of skipping them.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_optimal",
        help="Suffix appended to each input stem for the output filename.",
    )
    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive.")
    if args.max_tasks_per_child < 0:
        parser.error("--max-tasks-per-child cannot be negative.")
    if args.scale_factor <= 0:
        parser.error("--scale-factor must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided.")
    if args.progress_every_seconds < 0:
        parser.error("--progress-every-seconds cannot be negative.")
    if not args.output_suffix:
        parser.error("--output-suffix cannot be empty.")


def initialize_worker(scale_factor: int) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = WorkerConfig(scale_factor=scale_factor)


def extract_coordinates(sample: Any) -> np.ndarray:
    """Convert supported sample layouts to a validated ``(n, 2)`` array."""
    if isinstance(sample, dict):
        coords = sample.get("coords")
        if coords is None:
            coords = sample.get("nodes")
        if coords is None:
            raise ValueError("Dictionary sample must contain 'coords' or 'nodes'.")
    elif isinstance(sample, (list, tuple)):
        if not sample:
            raise ValueError("Empty sample.")
        first = np.asarray(sample[0])
        coords = first if first.ndim == 2 and first.shape[-1] == 2 else sample
    else:
        coords = sample

    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape (n, 2), got {array.shape}.")
    if array.shape[0] < 2:
        raise ValueError("A TSP instance must contain at least two nodes.")
    if not np.isfinite(array).all():
        raise ValueError("Coordinates contain NaN or infinite values.")
    return array


def unpack_dataset(payload: Any) -> list[Any]:
    """Extract the instance sequence from common dataset container formats."""
    if isinstance(payload, dict):
        for key in ("coords", "instances", "data"):
            if key in payload:
                return list(payload[key])
        raise ValueError("Dataset dictionary must contain coords, instances, or data.")

    # Some datasets are stored as (instances, auxiliary_data). Distinguish that
    # layout from a tuple that simply contains exactly two TSP instances.
    if isinstance(payload, tuple) and len(payload) == 2:
        candidate = payload[0]
        if isinstance(candidate, (list, tuple, np.ndarray)) and len(candidate) > 0:
            try:
                extract_coordinates(candidate[0])
            except (TypeError, ValueError, IndexError):
                pass
            else:
                return list(candidate)

    if isinstance(payload, (list, tuple, np.ndarray)):
        return list(payload)

    raise ValueError(f"Unsupported dataset container type: {type(payload).__name__}.")


def solve_tsp(coords: np.ndarray, scale_factor: int) -> np.ndarray:
    try:
        from concorde.tsp import TSPSolver
    except ImportError as exc:
        raise RuntimeError(
            "Concorde Python bindings are not installed or are not importable."
        ) from exc

    shifted = coords.copy()
    shifted -= shifted.min(axis=0, keepdims=True)
    scaled = np.rint(shifted * scale_factor).astype(np.int64)

    solver = TSPSolver.from_data(scaled[:, 0], scaled[:, 1], norm="EUC_2D")
    solution = solver.solve()
    if not solution.found_tour:
        raise RuntimeError("Concorde did not return a tour.")

    tour = np.asarray(solution.tour, dtype=np.int64)
    if tour.shape != (coords.shape[0],):
        raise RuntimeError(
            f"Concorde returned tour shape {tour.shape}; expected {(coords.shape[0],)}."
        )
    zero_positions = np.flatnonzero(tour == 0)
    if len(zero_positions) != 1:
        raise RuntimeError("Returned tour does not contain node 0 exactly once.")
    return np.roll(tour, -int(zero_positions[0]))


def compute_tour_length(coords: np.ndarray, tour: np.ndarray) -> float:
    ordered = coords[tour]
    closed = np.vstack((ordered, ordered[0]))
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def process_chunk(chunk: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration was not initialized.")

    outputs: list[dict[str, Any]] = []
    for instance_index, sample in chunk:
        try:
            coords = extract_coordinates(sample)
            tour = solve_tsp(coords, _WORKER_CONFIG.scale_factor)
            outputs.append(
                {
                    "ok": True,
                    "instance_idx": instance_index,
                    "coords": coords,
                    "tour": tour.tolist(),
                    "length": compute_tour_length(coords, tour),
                }
            )
        except Exception as exc:  # Keep other instances running after one failure.
            outputs.append(
                {
                    "ok": False,
                    "instance_idx": instance_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return outputs


def make_chunks(
    instances: list[Any], chunk_size: int
) -> Iterable[list[tuple[int, Any]]]:
    indexed = enumerate(instances)
    chunk: list[tuple[int, Any]] = []
    for item in indexed:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def infer_problem_size(instances: list[Any]) -> int:
    if not instances:
        raise ValueError("Dataset contains no instances.")
    return int(extract_coordinates(instances[0]).shape[0])


def process_file(input_path: Path, args: argparse.Namespace) -> None:
    output_path = args.output_dir / f"{input_path.stem}{args.output_suffix}.pkl"
    error_path = args.output_dir / (
        f"{input_path.stem}{args.output_suffix}_errors.pkl"
    )
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing output: {output_path.name}")
        return
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

    with input_path.open("rb") as handle:
        instances = unpack_dataset(pickle.load(handle))
    if args.limit is not None:
        instances = instances[: args.limit]

    problem_size = infer_problem_size(instances)
    chunks = list(make_chunks(instances, args.chunk_size))
    print(
        f"Solving {input_path.name}: TSP-{problem_size}, "
        f"{len(instances):,} instances, {len(chunks):,} chunks"
    )

    results: list[dict[str, Any]] = []
    completed_chunks = 0
    last_report = time.monotonic()
    pool_options: dict[str, Any] = {
        "processes": args.workers,
        "initializer": initialize_worker,
        "initargs": (args.scale_factor,),
    }
    if args.max_tasks_per_child:
        pool_options["maxtasksperchild"] = args.max_tasks_per_child

    with mp.Pool(**pool_options) as pool:
        for chunk_result in pool.imap_unordered(process_chunk, chunks):
            results.extend(chunk_result)
            completed_chunks += 1
            now = time.monotonic()
            if (
                args.progress_every_seconds
                and now - last_report >= args.progress_every_seconds
            ):
                print(
                    f"  {completed_chunks:,}/{len(chunks):,} chunks completed "
                    f"for {input_path.name}"
                )
                last_report = now

    results.sort(key=lambda item: item["instance_idx"])
    solved_results = []
    failed_results = []
    for item in results:
        if item.pop("ok"):
            solved_results.append(item)
        else:
            failed_results.append(item)

    # Preserve the original list-of-result-dictionaries output layout.
    with output_path.open("wb") as handle:
        pickle.dump(solved_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if failed_results:
        with error_path.open("wb") as handle:
            pickle.dump(failed_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Failure details saved: {error_path}")
    else:
        error_path.unlink(missing_ok=True)

    print(
        f"Saved {output_path} "
        f"({len(solved_results):,} solved, {len(failed_results):,} failed)"
    )


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in args.input_dir.glob(args.pattern) if path.is_file())
    if not files:
        raise SystemExit(
            f"No files matching {args.pattern!r} were found in {args.input_dir}."
        )

    for input_path in files:
        process_file(input_path, args)


if __name__ == "__main__":
    mp.freeze_support()
    main()

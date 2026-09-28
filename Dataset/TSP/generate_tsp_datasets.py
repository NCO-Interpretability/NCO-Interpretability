from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Callable

import numpy as np


DEFAULT_PROBLEM_SIZES = (200, 500, 1000)
DEFAULT_SAMPLE_COUNTS = (5000, 5000, 100)
DEFAULT_DISTRIBUTIONS = (
    "cluster",
    "grid",
    "expansion",
    "uniform",
    "explosion",
    "implosion",
    "mixed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Euclidean TSP datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated pickle files will be written.",
    )
    parser.add_argument(
        "--problem-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROBLEM_SIZES),
        help="TSP node counts to generate.",
    )
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_COUNTS),
        help="Number of instances for each problem size, in matching order.",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=DEFAULT_DISTRIBUTIONS,
        default=list(DEFAULT_DISTRIBUTIONS),
        help="Distributions to generate.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of skipping them.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many generated instances; use 0 to disable.",
    )

    # Distribution parameters
    parser.add_argument("--cluster-count", type=int, default=5)
    parser.add_argument("--mixed-cluster-count", type=int, default=5)
    parser.add_argument("--cluster-center-low", type=float, default=0.2)
    parser.add_argument("--cluster-center-high", type=float, default=0.8)
    parser.add_argument("--cluster-std", type=float, default=0.07)
    parser.add_argument("--grid-jitter-ratio", type=float, default=0.30)
    parser.add_argument("--expansion-factor", type=float, default=1.45)
    parser.add_argument("--implosion-factor", type=float, default=0.55)
    parser.add_argument("--implosion-noise-std", type=float, default=0.015)
    parser.add_argument("--explosion-radius-min", type=float, default=0.05)
    parser.add_argument("--explosion-radius-max", type=float, default=0.72)
    parser.add_argument("--explosion-beta-a", type=float, default=2.0)
    parser.add_argument("--explosion-beta-b", type=float, default=0.65)
    parser.add_argument("--explosion-noise-std", type=float, default=0.012)

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if len(args.problem_sizes) != len(args.sample_counts):
        parser.error("--problem-sizes and --sample-counts must have equal lengths.")
    if any(size <= 1 for size in args.problem_sizes):
        parser.error("Every problem size must be greater than 1.")
    if any(count <= 0 for count in args.sample_counts):
        parser.error("Every sample count must be positive.")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative.")
    if args.cluster_count <= 0 or args.mixed_cluster_count <= 0:
        parser.error("Cluster counts must be positive.")
    if not 0.0 <= args.cluster_center_low < args.cluster_center_high <= 1.0:
        parser.error("Cluster center bounds must satisfy 0 <= low < high <= 1.")
    if args.cluster_std < 0.0:
        parser.error("--cluster-std cannot be negative.")
    if args.grid_jitter_ratio < 0.0:
        parser.error("--grid-jitter-ratio cannot be negative.")
    if args.expansion_factor <= 0.0 or args.implosion_factor <= 0.0:
        parser.error("Expansion and implosion factors must be positive.")
    if args.implosion_noise_std < 0.0 or args.explosion_noise_std < 0.0:
        parser.error("Noise standard deviations cannot be negative.")
    if not 0.0 <= args.explosion_radius_min < args.explosion_radius_max:
        parser.error("Explosion radii must satisfy 0 <= min < max.")
    if args.explosion_beta_a <= 0.0 or args.explosion_beta_b <= 0.0:
        parser.error("Explosion beta parameters must be positive.")


def clip_unit_square(coords: np.ndarray) -> np.ndarray:
    return np.clip(coords, 0.0, 1.0).astype(np.float32, copy=False)


def shuffle_nodes(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return coords[rng.permutation(coords.shape[0])]


def split_counts(total: int, groups: int) -> np.ndarray:
    counts = np.full(groups, total // groups, dtype=int)
    counts[: total % groups] += 1
    return counts


def generate_uniform(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    del args
    return rng.uniform(0.0, 1.0, size=(problem_size, 2)).astype(np.float32)


def generate_cluster(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    centers = rng.uniform(
        args.cluster_center_low,
        args.cluster_center_high,
        size=(args.cluster_count, 2),
    )
    parts = [
        rng.normal(loc=center, scale=args.cluster_std, size=(count, 2))
        for center, count in zip(
            centers, split_counts(problem_size, args.cluster_count), strict=True
        )
        if count > 0
    ]
    return shuffle_nodes(clip_unit_square(np.concatenate(parts, axis=0)), rng)


def generate_mixed(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    clustered_count = problem_size // 2
    uniform_count = problem_size - clustered_count
    uniform_part = rng.uniform(0.0, 1.0, size=(uniform_count, 2))
    centers = rng.uniform(
        args.cluster_center_low,
        args.cluster_center_high,
        size=(args.mixed_cluster_count, 2),
    )
    clustered_parts = [
        rng.normal(loc=center, scale=args.cluster_std, size=(count, 2))
        for center, count in zip(
            centers,
            split_counts(clustered_count, args.mixed_cluster_count),
            strict=True,
        )
        if count > 0
    ]
    clustered_part = np.concatenate(clustered_parts, axis=0)
    coords = np.concatenate((uniform_part, clustered_part), axis=0)
    return shuffle_nodes(clip_unit_square(coords), rng)


def generate_grid(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    side = math.ceil(math.sqrt(problem_size))
    axis = np.linspace(0.05, 0.95, side, dtype=np.float32)
    grid_points = np.array([(x, y) for x in axis for y in axis], dtype=np.float32)
    coords = grid_points[rng.choice(len(grid_points), size=problem_size, replace=False)]
    spacing = 0.90 / max(side - 1, 1)
    coords = coords + rng.normal(
        0.0, args.grid_jitter_ratio * spacing, size=coords.shape
    )
    return shuffle_nodes(clip_unit_square(coords), rng)


def generate_expansion(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    center = np.array([0.5, 0.5], dtype=np.float32)
    base = rng.uniform(0.0, 1.0, size=(problem_size, 2))
    coords = center + args.expansion_factor * (base - center)
    return shuffle_nodes(clip_unit_square(coords), rng)


def generate_implosion(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    center = np.array([0.5, 0.5], dtype=np.float32)
    base = rng.uniform(0.0, 1.0, size=(problem_size, 2))
    coords = center + args.implosion_factor * (base - center)
    coords += rng.normal(0.0, args.implosion_noise_std, size=coords.shape)
    return shuffle_nodes(clip_unit_square(coords), rng)


def generate_explosion(
    problem_size: int, rng: np.random.Generator, args: argparse.Namespace
) -> np.ndarray:
    center = np.array([0.5, 0.5], dtype=np.float32)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=problem_size)
    radius_unit = rng.beta(
        args.explosion_beta_a, args.explosion_beta_b, size=problem_size
    )
    radii = args.explosion_radius_min + (
        args.explosion_radius_max - args.explosion_radius_min
    ) * radius_unit
    coords = np.column_stack(
        (center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles))
    )
    coords += rng.normal(0.0, args.explosion_noise_std, size=coords.shape)
    return shuffle_nodes(clip_unit_square(coords), rng)


GENERATORS: dict[
    str, Callable[[int, np.random.Generator, argparse.Namespace], np.ndarray]
] = {
    "uniform": generate_uniform,
    "cluster": generate_cluster,
    "mixed": generate_mixed,
    "grid": generate_grid,
    "expansion": generate_expansion,
    "implosion": generate_implosion,
    "explosion": generate_explosion,
}


def serialize_instance(coords: np.ndarray) -> tuple[list[float], ...]:
    return tuple(coords.astype(np.float32, copy=False).tolist())


def generate_dataset(
    distribution: str,
    problem_size: int,
    sample_count: int,
    seed: int,
    args: argparse.Namespace,
) -> list[tuple[list[float], ...]]:
    rng = np.random.default_rng(seed)
    generator = GENERATORS[distribution]
    dataset: list[tuple[list[float], ...]] = []

    for index in range(sample_count):
        dataset.append(serialize_instance(generator(problem_size, rng, args)))
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"  generated {index + 1:,}/{sample_count:,}")

    return dataset


def output_filename(distribution: str, problem_size: int, sample_count: int) -> str:
    return f"tsp_{distribution}_n{problem_size}_{sample_count}_instances.pkl"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_count_by_size = dict(zip(args.problem_sizes, args.sample_counts, strict=True))

    seed_sequence = np.random.SeedSequence(args.seed)
    jobs = [
        (problem_size, distribution)
        for problem_size in args.problem_sizes
        for distribution in args.distributions
    ]
    child_sequences = seed_sequence.spawn(len(jobs))

    for (problem_size, distribution), child_sequence in zip(
        jobs, child_sequences, strict=True
    ):
        sample_count = sample_count_by_size[problem_size]
        destination = args.output_dir / output_filename(
            distribution, problem_size, sample_count
        )

        if destination.exists() and not args.overwrite:
            print(f"Skipping existing file: {destination.name}")
            continue

        file_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
        print(
            f"Generating {distribution} TSP-{problem_size} "
            f"({sample_count:,} instances)"
        )
        dataset = generate_dataset(
            distribution, problem_size, sample_count, file_seed, args
        )
        with destination.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {destination}")

    print("Dataset generation completed.")


if __name__ == "__main__":
    main()

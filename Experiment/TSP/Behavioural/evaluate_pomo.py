import os
import sys
import glob
import time
import argparse

import numpy as np
import torch


# ============================================================
# Make local utils importable
# ============================================================

# Import the bundled shared utilities from the neighboring utils package.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_UTILS_ROOT = os.path.join(CURRENT_DIR, "utils")
sys.path.insert(0, BUNDLED_UTILS_ROOT)

from utils.util import (
    setup_torch_device,
    move_tensor_attrs_to_device,
    load_coordinate_list,
    group_indices_by_problem_size,
    load_checkpoint,
    build_plotfriendly_instances,
    save_pickle,
    save_summary_csv,
    make_summary_row,
    empty_cuda_cache_if_available,
)


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    """
    Parse command-line arguments.

    The most important arguments are:
    - --data_dir: input folder containing .pkl files
    - --output_dir: output folder for result .pkl files and summary CSV
    """
    parser = argparse.ArgumentParser(
        description="Evaluate POMO on TSP PKL datasets."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input directory containing TSP .pkl dataset files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for plot-friendly result files.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained POMO checkpoint.",
    )

    parser.add_argument(
        "--pomo_root",
        type=str,
        required=True,
        help="Path to the POMO source directory containing TSPModel.py and TSPEnv.py.",
    )

    parser.add_argument(
        "--pomo_tsp_root",
        type=str,
        required=True,
        help="Path to the parent TSP source directory.",
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
        default=256,
        help="Batch size for inference.",
    )

    parser.add_argument(
        "--use_full_pomo",
        action="store_true",
        help=(
            "If set, use full POMO with pomo_size=problem_size. "
            "This improves quality but uses much more GPU memory."
        ),
    )

    parser.add_argument(
        "--summary_filename",
        type=str,
        required=True,
        help="Name of the summary CSV file saved inside output_dir.",
    )

    args = parser.parse_args()
    if args.cuda_device_num < 0:
        parser.error("--cuda_device_num cannot be negative.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if not args.summary_filename.strip():
        parser.error("--summary_filename cannot be empty.")
    return args


# ============================================================
# Dynamic POMO import
# ============================================================

def import_pomo_modules(pomo_root, pomo_tsp_root):
    """
    Add POMO paths to sys.path and import TSPModel/TSPEnv.

    This is done after parsing args so source paths can be passed
    from the command line.
    """
    sys.path.insert(0, pomo_root)
    sys.path.insert(0, pomo_tsp_root)

    from TSPModel import TSPModel
    from TSPEnv import TSPEnv

    return TSPModel, TSPEnv


# ============================================================
# Model loading
# ============================================================

def build_pomo_model(TSPModel, checkpoint_path, device):
    """
    Build the POMO TSP model and load checkpoint weights.
    """
    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }

    model = TSPModel(**model_params)

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        device=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# ============================================================
# POMO inference
# ============================================================

def solve_batch(
    model,
    TSPEnv,
    batch_coords,
    problem_size,
    device,
    use_full_pomo=False,
):
    """
    Solve one batch of TSP instances using POMO.

    Parameters
    ----------
    model:
        Loaded POMO model.

    TSPEnv:
        POMO environment class.

    batch_coords:
        List of coordinate arrays, each with shape (N, 2).

    problem_size:
        Number of nodes.

    device:
        torch device.

    use_full_pomo:
        If False, use pomo_size=1.
        If True, use pomo_size=problem_size.

    Returns
    -------
    np.ndarray
        Best tour for each instance, shape (batch, problem_size).
    """
    current_batch = len(batch_coords)

    problems = torch.tensor(
        np.asarray(batch_coords, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    if use_full_pomo:
        pomo_size = problem_size
    else:
        pomo_size = 1

    # Create a new environment for this problem size.
    env = TSPEnv(
        problem_size=problem_size,
        pomo_size=pomo_size,
    )

    # Inject external problems directly into the environment.
    env.problems = problems
    env.batch_size = current_batch

    # Create index tensors on the correct device.
    env.BATCH_IDX = torch.arange(
        current_batch,
        dtype=torch.long,
        device=device,
    )[:, None].expand(current_batch, pomo_size)

    env.POMO_IDX = torch.arange(
        pomo_size,
        dtype=torch.long,
        device=device,
    )[None, :].expand(current_batch, pomo_size)

    # Reset environment.
    reset_state, _, _ = env.reset()

    # Move internally created tensors to the selected device.
    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(reset_state, device)

    # Encode nodes once before rollout.
    model.pre_forward(reset_state)

    # Initial decoding state.
    state, reward, done = env.pre_step()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(state, device)

    # Greedy rollout.
    while not done:
        selected, _ = model(state)

        # Ensure selected nodes are on the correct device.
        selected = selected.to(device).long()

        move_tensor_attrs_to_device(env, device)

        state, reward, done = env.step(selected)

        move_tensor_attrs_to_device(env, device)
        move_tensor_attrs_to_device(state, device)

    # reward shape: (batch, pomo)
    # reward is negative tour length.
    reward = reward.detach()

    # selected_node_list shape: (batch, pomo, problem_size)
    selected_node_list = env.selected_node_list.detach()

    batch_index = torch.arange(
        current_batch,
        dtype=torch.long,
        device=device,
    )

    # Select best POMO rollout for each instance.
    best_pomo_index = reward.argmax(dim=1)

    best_tours = selected_node_list[
        batch_index,
        best_pomo_index,
    ].cpu().numpy()

    return best_tours


def solve_coordinate_group(
    model,
    TSPEnv,
    coords_group,
    problem_size,
    device,
    batch_size,
    use_full_pomo=False,
):
    """
    Solve all instances with the same problem size.
    """
    all_tours = []

    with torch.no_grad():
        for start in range(0, len(coords_group), batch_size):
            batch_coords = coords_group[start:start + batch_size]

            tours = solve_batch(
                model=model,
                TSPEnv=TSPEnv,
                batch_coords=batch_coords,
                problem_size=problem_size,
                device=device,
                use_full_pomo=use_full_pomo,
            )

            all_tours.extend(tours)

            print(
                f"    Solved {min(start + batch_size, len(coords_group))}/"
                f"{len(coords_group)}"
            )

    return all_tours


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate_one_dataset(
    dataset_file,
    model,
    TSPEnv,
    output_dir,
    device,
    batch_size,
    use_full_pomo=False,
):
    """
    Evaluate one PKL dataset file.

    Returns
    -------
    dict
        Summary row for CSV.
    """
    filename = os.path.basename(dataset_file)

    print("\n============================================================")
    print("Processing:", filename)

    file_start_time = time.time()
    output_path = os.path.join(output_dir, "results_" + filename)

    try:
        # Load all coordinates from the dataset.
        coords_list = load_coordinate_list(dataset_file)

        # Group instances by problem size.
        size_to_indices = group_indices_by_problem_size(coords_list)
        detected_problem_sizes = sorted(size_to_indices.keys())

        print("Detected problem sizes:", detected_problem_sizes)

        final_tours = [None] * len(coords_list)

        # Solve each problem-size group separately.
        for problem_size, indices in sorted(size_to_indices.items()):
            print(f"\n  Problem size: {problem_size}")
            print(f"  Number of instances: {len(indices)}")

            coords_group = [coords_list[idx] for idx in indices]

            tours = solve_coordinate_group(
                model=model,
                TSPEnv=TSPEnv,
                coords_group=coords_group,
                problem_size=problem_size,
                device=device,
                batch_size=batch_size,
                use_full_pomo=use_full_pomo,
            )

            for local_idx, original_idx in enumerate(indices):
                final_tours[original_idx] = tours[local_idx]

        # Build plot-friendly result structure.
        output_instances, eval_summary = build_plotfriendly_instances(
            coords_list=coords_list,
            tours=final_tours,
            method="POMO_GREEDY" if not use_full_pomo else "POMO_FULL",
            extra_fields={
                "use_full_pomo": bool(use_full_pomo),
                "pomo_size_mode": "problem_size" if use_full_pomo else "1",
            },
        )

        # Save result PKL.
        save_pickle(output_instances, output_path)

        elapsed = time.time() - file_start_time

        print(f"Elapsed time: {elapsed:.4f} seconds")
        print(f"Valid tours:   {eval_summary['num_valid_tours']}")
        print(f"Invalid tours: {eval_summary['num_invalid_tours']}")
        print(f"Mean length:   {eval_summary['mean_tour_length']}")
        print(f"Saved: {output_path}")

        return make_summary_row(
            filename=filename,
            status="success",
            num_instances=len(coords_list),
            problem_sizes=detected_problem_sizes,
            num_valid_tours=eval_summary["num_valid_tours"],
            num_invalid_tours=eval_summary["num_invalid_tours"],
            mean_tour_length=eval_summary["mean_tour_length"],
            elapsed_seconds=elapsed,
            batch_size=batch_size,
            method="POMO_GREEDY" if not use_full_pomo else "POMO_FULL",
            output_path=output_path,
            error_message="",
        )

    except Exception as e:
        elapsed = time.time() - file_start_time

        print(f"ERROR while processing {filename}")
        print("Error message:", str(e))

        return make_summary_row(
            filename=filename,
            status="failed",
            elapsed_seconds=elapsed,
            batch_size=batch_size,
            method="POMO_GREEDY" if not use_full_pomo else "POMO_FULL",
            output_path="",
            error_message=str(e),
        )


# ============================================================
# Main
# ============================================================

def main():
    """
    Main POMO evaluation pipeline.
    """
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_csv_path = os.path.join(
        args.output_dir,
        args.summary_filename,
    )

    # Import POMO modules after reading source-path arguments.
    TSPModel, TSPEnv = import_pomo_modules(
        pomo_root=args.pomo_root,
        pomo_tsp_root=args.pomo_tsp_root,
    )

    # Configure torch device.
    device = setup_torch_device(
        cuda_device_num=args.cuda_device_num,
        set_default_device=True,
        legacy_cuda_default_tensor_type=False,
    )

    print("Device:", device)

    # Build model.
    model = build_pomo_model(
        TSPModel=TSPModel,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    print("POMO model loaded.")

    # Collect dataset files.
    dataset_files = sorted(
        glob.glob(os.path.join(args.data_dir, "*.pkl"))
    )

    print("Number of dataset files:", len(dataset_files))

    summary_rows = []

    for dataset_file in dataset_files:
        row = evaluate_one_dataset(
            dataset_file=dataset_file,
            model=model,
            TSPEnv=TSPEnv,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            use_full_pomo=args.use_full_pomo,
        )

        summary_rows.append(row)

        # Update summary CSV after each dataset.
        save_summary_csv(
            rows=summary_rows,
            csv_path=summary_csv_path,
        )

        print("Summary CSV updated:", summary_csv_path)

        empty_cuda_cache_if_available()

    print("\n============================================================")
    print("All done.")
    print("Summary CSV:", summary_csv_path)


if __name__ == "__main__":
    main()

import os
import sys
import glob
import time
import argparse
import re

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
    get_state_dict_from_checkpoint,
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
        description="Evaluate LEHD-Greedy on TSP PKL datasets."
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
        help="Path to the trained LEHD checkpoint.",
    )

    parser.add_argument(
        "--lehd_root",
        type=str,
        required=True,
        help="Path to the LEHD root directory containing the LEHD package.",
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
        "--summary_filename",
        type=str,
        required=True,
        help="Name of the summary CSV file saved inside output_dir.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="LEHD model/env mode. Usually should be 'test'.",
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
# Dynamic LEHD import
# ============================================================

def import_lehd_modules(lehd_root):
    """
    Add LEHD path to sys.path and import LEHD TSPModel/TSPEnv.

    This is done after parsing args so the source path can be passed
    from the command line.
    """
    sys.path.insert(0, lehd_root)

    from LEHD.TSP.TSPModel import TSPModel as Model
    from LEHD.TSP.TSPEnv import TSPEnv as Env

    return Model, Env


# ============================================================
# Model helpers
# ============================================================

def infer_decoder_layer_num(state_dict, fallback=6):
    """
    Infer LEHD decoder_layer_num from checkpoint keys.

    This avoids errors like:
        Unexpected key(s): decoder.layers.4..., decoder.layers.5...

    If checkpoint has decoder.layers.0 ... decoder.layers.5,
    this function returns 6.
    """
    layer_indices = []

    patterns = [
        re.compile(r"decoder\.layers\.(\d+)\."),
        re.compile(r"layers\.(\d+)\."),
    ]

    for key in state_dict.keys():
        for pattern in patterns:
            match = pattern.search(key)
            if match is not None:
                layer_indices.append(int(match.group(1)))

    if len(layer_indices) == 0:
        print(f"WARNING: Could not infer decoder_layer_num. Using fallback={fallback}.")
        return fallback

    decoder_layer_num = max(layer_indices) + 1
    print(f"Inferred decoder_layer_num from checkpoint: {decoder_layer_num}")

    return decoder_layer_num


def build_lehd_model(Model, checkpoint_path, device, mode):
    """
    Build the LEHD TSP model and load checkpoint weights.
    """
    checkpoint = load_checkpoint(
        path=checkpoint_path,
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

    return model


def build_lehd_env(Env, mode):
    """
    Build LEHD TSP environment.

    This script is pure greedy:
    - RRC_budget = 0
    - no repair
    - no destroy_solution

    data_path is not used because external PKL problems are injected directly.
    """
    env_params = {
        "mode": mode,
        "data_path": "",
        "sub_path": False,
        "RRC_budget": 0,
    }

    env = Env(**env_params)

    return env


def inject_problems_into_env(env, problems, device):
    """
    Inject external PKL problems into the LEHD environment.

    The original LEHD code expects env.solution to exist.
    Since external PKL datasets only provide coordinates, we create a dummy
    identity solution only for interface compatibility.

    The predicted tour is taken from env.selected_node_list after rollout.
    """
    batch_size, problem_size, _ = problems.shape

    env.problems = problems.to(device)
    env.batch_size = batch_size
    env.problem_size = problem_size

    # Dummy identity solution required by the model/env interface.
    # It is not used as the final predicted solution.
    env.solution = torch.arange(
        problem_size,
        dtype=torch.long,
        device=device,
    )[None, :].expand(batch_size, problem_size).clone()

    move_tensor_attrs_to_device(env, device)


# ============================================================
# LEHD pure-greedy inference
# ============================================================

def lehd_greedy_rollout(
    model,
    env,
    problems,
    device,
    mode,
):
    """
    Run pure greedy LEHD decoding.

    This follows only the greedy construction part of the original LEHD tester:
    - current_step == 0: start from node 0
    - current_step > 0: use model prediction
    - no RRC
    - no repair
    - no destroy_solution
    """
    batch_size, problem_size, _ = problems.shape

    model.eval()
    model.mode = mode

    inject_problems_into_env(
        env=env,
        problems=problems,
        device=device,
    )

    # Reset environment.
    # Some LEHD versions use reset(mode), others use reset().
    try:
        reset_state, _, _ = env.reset(mode)
    except TypeError:
        reset_state, _, _ = env.reset()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(reset_state, device)

    # Initial environment state.
    state, reward, reward_student, done = env.pre_step()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(state, device)

    current_step = 0

    while not done:
        if current_step == 0:
            # Original LEHD greedy starts from node 0.
            selected_teacher = torch.zeros(
                batch_size,
                dtype=torch.int64,
                device=device,
            )

            selected_student = selected_teacher

        else:
            selected_teacher, _, _, selected_student = model(
                state,
                env.selected_node_list,
                env.solution,
                current_step,
            )

            selected_teacher = selected_teacher.to(device).long()
            selected_student = selected_student.to(device).long()

        current_step += 1

        move_tensor_attrs_to_device(env, device)

        state, reward, reward_student, done = env.step(
            selected_teacher,
            selected_student,
        )

        move_tensor_attrs_to_device(env, device)
        move_tensor_attrs_to_device(state, device)

    # Final predicted tour.
    final_tour = env.selected_node_list.detach().clone()

    return final_tour


def solve_batch(
    model,
    Env,
    batch_coords,
    problem_size,
    device,
    mode,
):
    """
    Solve one batch using pure greedy LEHD decoding.

    A fresh environment is created for each batch to avoid state leakage.
    """
    env = build_lehd_env(
        Env=Env,
        mode=mode,
    )

    problems = torch.tensor(
        np.asarray(batch_coords, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        tours = lehd_greedy_rollout(
            model=model,
            env=env,
            problems=problems,
            device=device,
            mode=mode,
        )

    return tours.detach().cpu().numpy()


def solve_coordinate_group(
    model,
    Env,
    coords_group,
    problem_size,
    device,
    batch_size,
    mode,
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
                Env=Env,
                batch_coords=batch_coords,
                problem_size=problem_size,
                device=device,
                mode=mode,
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
    Env,
    output_dir,
    device,
    batch_size,
    mode,
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
                Env=Env,
                coords_group=coords_group,
                problem_size=problem_size,
                device=device,
                batch_size=batch_size,
                mode=mode,
            )

            for local_idx, original_idx in enumerate(indices):
                final_tours[original_idx] = tours[local_idx]

        # Build plot-friendly result structure.
        output_instances, eval_summary = build_plotfriendly_instances(
            coords_list=coords_list,
            tours=final_tours,
            method="LEHD_GREEDY",
            extra_fields={
                "use_rrc": False,
                "rrc_budget": 0,
                "use_repair": False,
                "start_policy": "node_0",
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
            method="LEHD_GREEDY",
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
            method="LEHD_GREEDY",
            output_path="",
            error_message=str(e),
        )


# ============================================================
# Main
# ============================================================

def main():
    """
    Main LEHD pure-greedy evaluation pipeline.
    """
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_csv_path = os.path.join(
        args.output_dir,
        args.summary_filename,
    )

    # Import LEHD modules after reading source-path arguments.
    Model, Env = import_lehd_modules(
        lehd_root=args.lehd_root,
    )

    # Configure torch device.
    #
    # LEHD original code sometimes relies on default CUDA tensor behavior.
    # setup_torch_device with set_default_device=True is usually enough on
    # PyTorch 2.x. The legacy option is left False to avoid deprecation noise.
    device = setup_torch_device(
        cuda_device_num=args.cuda_device_num,
        set_default_device=True,
        legacy_cuda_default_tensor_type=False,
    )

    print("Device:", device)

    # Build model.
    model = build_lehd_model(
        Model=Model,
        checkpoint_path=args.checkpoint_path,
        device=device,
        mode=args.mode,
    )

    print("LEHD model loaded.")

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
            Env=Env,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            mode=args.mode,
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

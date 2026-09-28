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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_UTILS_ROOT = os.path.join(CURRENT_DIR, "utils")
sys.path.insert(0, BUNDLED_UTILS_ROOT)

from utils.util import (
    set_seed,
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
    """
    parser = argparse.ArgumentParser(
        description="Evaluate L2C-Insert Greedy and save efficient insertion trace."
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
        help="Path to the trained L2C-Insert checkpoint.",
    )

    parser.add_argument(
        "--l2c_root",
        type=str,
        required=True,
        help="Path to the L2C_Insert root directory.",
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
        help="L2C-Insert model/env mode. Usually should be 'test'.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.",
    )

    parser.add_argument(
        "--k_nearest_edges",
        type=int,
        default=100,
        help="k nearest edges option for L2C-Insert decoder.",
    )

    parser.add_argument(
        "--k_nearest_scatter",
        type=int,
        default=100,
        help="k nearest scatter option for L2C-Insert decoder.",
    )

    parser.add_argument(
        "--coor_norm",
        action="store_true",
        help="Enable coordinate normalization option if supported by model.",
    )

    parser.add_argument(
        "--distribution_keywords",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional case-insensitive filename keywords used to select datasets. "
            "When omitted, all PKL files in data_dir are evaluated."
        ),
    )

    args = parser.parse_args()
    if args.cuda_device_num < 0:
        parser.error("--cuda_device_num cannot be negative.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.k_nearest_edges <= 0:
        parser.error("--k_nearest_edges must be positive.")
    if args.k_nearest_scatter <= 0:
        parser.error("--k_nearest_scatter must be positive.")
    if not args.summary_filename.strip():
        parser.error("--summary_filename cannot be empty.")
    if args.distribution_keywords is not None:
        args.distribution_keywords = [
            item.strip().lower()
            for item in args.distribution_keywords
            if item.strip()
        ]
        if not args.distribution_keywords:
            parser.error(
                "--distribution_keywords requires at least one non-empty keyword."
            )
    return args


# ============================================================
# Dynamic L2C-Insert import
# ============================================================

def import_l2c_modules(l2c_root):
    """
    Add L2C-Insert source path to sys.path and import model/env classes.
    """
    sys.path.insert(0, l2c_root)

    from L2C_Insert.TSP.Test.TSPModel import TSPModel as Model
    from L2C_Insert.TSP.Test.TSPEnv import TSPEnv as Env

    return Model, Env


# ============================================================
# Model and environment helpers
# ============================================================

def infer_decoder_layer_num(state_dict, fallback=9):
    """
    Infer decoder_layer_num from checkpoint keys.

    The original pretrained L2C-Insert model usually uses 9 decoder layers.
    """
    layer_indices = []

    patterns = [
        re.compile(r"decoder\.layers\.(\d+)\."),
        re.compile(r"decoder_layer\.(\d+)\."),
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


def build_l2c_model(
    Model,
    checkpoint_path,
    device,
    mode,
    k_nearest_edges,
    k_nearest_scatter,
    coor_norm,
):
    """
    Build L2C-Insert model and load checkpoint.
    """
    checkpoint = load_checkpoint(
        path=checkpoint_path,
        device=device,
    )

    state_dict = get_state_dict_from_checkpoint(checkpoint)

    decoder_layer_num = infer_decoder_layer_num(
        state_dict=state_dict,
        fallback=9,
    )

    model_params = {
        "mode": mode,
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "decoder_layer_num": decoder_layer_num,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
        "knearest": False,
        "k_nearest_edges": k_nearest_edges,
        "k_nearest_scatter": k_nearest_scatter,
        "coor_norm": coor_norm,
    }

    model = Model(**model_params)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    model.mode = mode

    return model


def build_l2c_env(Env, mode):
    """
    Build L2C-Insert environment.

    This script is pure greedy:
    - no RRC
    - no repair
    - no random insertion
    """
    env_params = {
        "mode": mode,
        "test_in_tsplib": False,
        "tsplib_path": "",
        "data_path": "",
        "sub_path": False,
        "RRC_budget": 0,
        "max_RRC_range": 0,
        "mix_sample_strategy": False,
        "turn_to_cluster_strategy": False,
        "random_insertion": False,
    }

    return Env(**env_params)


def set_model_dynamic_options(model, problem_size):
    """
    Set model options dynamically based on problem size.

    Original L2C-Insert scripts usually use:
    - knearest = False for TSP100
    - knearest = True for larger problem sizes
    """
    use_knearest = bool(problem_size > 100)

    if hasattr(model, "knearest"):
        model.knearest = use_knearest

    if hasattr(model, "decoder") and hasattr(model.decoder, "knearest"):
        model.decoder.knearest = use_knearest

    if hasattr(model, "coor_norm"):
        model.coor_norm = bool(problem_size > 1000)

    if hasattr(model, "decoder") and hasattr(model.decoder, "coor_norm"):
        model.decoder.coor_norm = bool(problem_size > 1000)


def inject_problems_into_env(env, problems, device):
    """
    Inject external PKL problems into L2C-Insert environment.

    The original environment expects env.solution to exist.
    We create a dummy identity solution only for interface compatibility.
    """
    batch_size, problem_size, _ = problems.shape

    env.problems = problems.to(device)
    env.batch_size = batch_size
    env.problem_size = problem_size
    env.test_in_tsplib = False

    env.solution = torch.arange(
        problem_size,
        dtype=torch.long,
        device=device,
    )[None, :].expand(batch_size, problem_size).clone()

    move_tensor_attrs_to_device(env, device)


# ============================================================
# Distance helpers
# ============================================================

def euclidean_distance(x, y):
    """
    Compute Euclidean distance.

    x shape:
        (B, seq, D)

    y shape:
        (B, 1, D)

    output shape:
        (B, seq)
    """
    return ((x - y) ** 2).sum(dim=2).sqrt()


def gather_node_coords(problems, node_indices):
    """
    Gather node coordinates from a batch of TSP instances.

    problems shape:
        (B, N, 2)

    node_indices shape:
        (B,)

    output shape:
        (B, 2)
    """
    batch_size = problems.size(0)

    batch_idx = torch.arange(
        batch_size,
        dtype=torch.long,
        device=problems.device,
    )

    return problems[batch_idx, node_indices.long()]


# ============================================================
# Efficient vectorized trace helpers
# ============================================================

def make_empty_trace_tensor_lists():
    """
    Create containers for vectorized trace tensors.

    Each list contains one tensor per insertion step.
    Each tensor has shape (B,).
    """
    return {
        "steps": [],
        "inserted_nodes": [],
        "insert_positions": [],

        "insert_prev_path": [],
        "insert_next_path": [],

        "insert_prev_cyclic": [],
        "insert_next_cyclic": [],

        "insertion_delta_cyclic": [],
        "nearest_distance": [],

        "partial_len_before": [],
        "partial_len_after": [],
    }


def append_vectorized_trace_event(trace_tensors, event):
    """
    Append one vectorized trace event to trace containers.
    """
    for key, value in event.items():
        trace_tensors[key].append(value.detach())


def compute_cyclic_insertion_delta_vectorized(
    problems,
    prev_nodes,
    inserted_nodes,
    next_nodes,
    valid_mask,
):
    """
    Compute cyclic insertion delta for the whole batch on GPU.

    delta = dist(prev, inserted) + dist(inserted, next) - dist(prev, next)
    """
    problem_size = problems.size(1)

    safe_prev = prev_nodes.clamp(0, problem_size - 1).long()
    safe_inserted = inserted_nodes.clamp(0, problem_size - 1).long()
    safe_next = next_nodes.clamp(0, problem_size - 1).long()

    prev_xy = gather_node_coords(problems, safe_prev)
    inserted_xy = gather_node_coords(problems, safe_inserted)
    next_xy = gather_node_coords(problems, safe_next)

    delta = (
        torch.norm(prev_xy - inserted_xy, dim=1)
        + torch.norm(inserted_xy - next_xy, dim=1)
        - torch.norm(prev_xy - next_xy, dim=1)
    )

    nan_values = torch.full_like(delta, float("nan"))
    delta = torch.where(valid_mask, delta, nan_values)

    return delta.float()


def build_vectorized_insertion_event(
    problems,
    old_partial,
    new_partial,
    inserted_nodes,
    nearest_distances,
    current_step,
):
    """
    Build insertion trace event for the whole batch without CPU loops.

    old_partial shape:
        (B, L)

    new_partial shape:
        (B, L + 1)

    inserted_nodes shape:
        (B,)

    nearest_distances shape:
        (B,)
    """
    device = problems.device
    batch_size = problems.size(0)
    problem_size = problems.size(1)

    old_partial = old_partial.long()
    new_partial = new_partial.long()
    inserted_nodes = inserted_nodes.reshape(batch_size).long()
    nearest_distances = nearest_distances.reshape(batch_size).float()

    old_len = old_partial.size(1)
    new_len = new_partial.size(1)

    # Find insertion position in the new partial solution.
    match = new_partial.eq(inserted_nodes[:, None])

    valid_inserted = (
        (inserted_nodes >= 0)
        & (inserted_nodes < problem_size)
    )

    found = match.any(dim=1) & valid_inserted

    raw_insert_pos = match.to(torch.int64).argmax(dim=1)
    minus_one = torch.full_like(raw_insert_pos, -1)

    insert_pos = torch.where(
        found,
        raw_insert_pos,
        minus_one,
    )

    # --------------------------------------------------------
    # Path interpretation neighbors
    # --------------------------------------------------------
    prev_path_gather_idx = (insert_pos - 1).clamp(0, old_len - 1)
    next_path_gather_idx = insert_pos.clamp(0, old_len - 1)

    prev_path_candidate = old_partial.gather(
        dim=1,
        index=prev_path_gather_idx[:, None],
    ).squeeze(1)

    next_path_candidate = old_partial.gather(
        dim=1,
        index=next_path_gather_idx[:, None],
    ).squeeze(1)

    prev_path_valid = found & (insert_pos > 0)
    next_path_valid = found & (insert_pos >= 0) & (insert_pos < old_len)

    prev_path = torch.where(
        prev_path_valid,
        prev_path_candidate,
        minus_one,
    )

    next_path = torch.where(
        next_path_valid,
        next_path_candidate,
        minus_one,
    )

    # --------------------------------------------------------
    # Cyclic interpretation neighbors
    # --------------------------------------------------------
    if old_len == 1:
        prev_cyclic_candidate = old_partial[:, 0]
        next_cyclic_candidate = old_partial[:, 0]

    else:
        prev_cyclic_from_before = prev_path_candidate
        prev_cyclic_from_end = old_partial[:, -1]

        prev_cyclic_candidate = torch.where(
            insert_pos > 0,
            prev_cyclic_from_before,
            prev_cyclic_from_end,
        )

        next_cyclic_from_after = next_path_candidate
        next_cyclic_from_start = old_partial[:, 0]

        next_cyclic_candidate = torch.where(
            insert_pos < old_len,
            next_cyclic_from_after,
            next_cyclic_from_start,
        )

    prev_cyclic = torch.where(
        found,
        prev_cyclic_candidate,
        minus_one,
    )

    next_cyclic = torch.where(
        found,
        next_cyclic_candidate,
        minus_one,
    )

    delta_cyclic = compute_cyclic_insertion_delta_vectorized(
        problems=problems,
        prev_nodes=prev_cyclic,
        inserted_nodes=inserted_nodes,
        next_nodes=next_cyclic,
        valid_mask=found,
    )

    step_tensor = torch.full(
        (batch_size,),
        int(current_step),
        dtype=torch.int32,
        device=device,
    )

    partial_len_before = torch.full(
        (batch_size,),
        int(old_len),
        dtype=torch.int32,
        device=device,
    )

    partial_len_after = torch.full(
        (batch_size,),
        int(new_len),
        dtype=torch.int32,
        device=device,
    )

    event = {
        "steps": step_tensor,

        "inserted_nodes": inserted_nodes.to(torch.int32),
        "insert_positions": insert_pos.to(torch.int32),

        "insert_prev_path": prev_path.to(torch.int32),
        "insert_next_path": next_path.to(torch.int32),

        "insert_prev_cyclic": prev_cyclic.to(torch.int32),
        "insert_next_cyclic": next_cyclic.to(torch.int32),

        "insertion_delta_cyclic": delta_cyclic.to(torch.float32),
        "nearest_distance": nearest_distances.to(torch.float32),

        "partial_len_before": partial_len_before,
        "partial_len_after": partial_len_after,
    }

    return event


def convert_trace_tensor_lists_to_arrays(trace_tensors, batch_size):
    """
    Convert vectorized trace tensors to per-instance compact NumPy arrays.

    Input:
        trace_tensors[key] = list of T tensors, each shape (B,)

    Output:
        traces[b][key] = NumPy array of shape (T,)
    """
    if len(trace_tensors["steps"]) == 0:
        empty_traces = []

        for _ in range(batch_size):
            empty_traces.append(
                {
                    "steps": np.asarray([], dtype=np.int32),
                    "inserted_nodes": np.asarray([], dtype=np.int32),
                    "insert_positions": np.asarray([], dtype=np.int32),

                    "insert_prev_path": np.asarray([], dtype=np.int32),
                    "insert_next_path": np.asarray([], dtype=np.int32),

                    "insert_prev_cyclic": np.asarray([], dtype=np.int32),
                    "insert_next_cyclic": np.asarray([], dtype=np.int32),

                    "insertion_delta_cyclic": np.asarray([], dtype=np.float32),
                    "nearest_distance": np.asarray([], dtype=np.float32),

                    "partial_len_before": np.asarray([], dtype=np.int32),
                    "partial_len_after": np.asarray([], dtype=np.int32),
                }
            )

        return empty_traces

    stacked = {}

    for key, values in trace_tensors.items():
        stacked[key] = torch.stack(values, dim=0).detach().cpu().numpy()
        # Shape: (T, B)

    _, batch_size = stacked["steps"].shape

    traces = []

    for b in range(batch_size):
        trace = {
            "steps": stacked["steps"][:, b].astype(np.int32),

            "inserted_nodes": stacked["inserted_nodes"][:, b].astype(np.int32),
            "insert_positions": stacked["insert_positions"][:, b].astype(np.int32),

            "insert_prev_path": stacked["insert_prev_path"][:, b].astype(np.int32),
            "insert_next_path": stacked["insert_next_path"][:, b].astype(np.int32),

            "insert_prev_cyclic": stacked["insert_prev_cyclic"][:, b].astype(np.int32),
            "insert_next_cyclic": stacked["insert_next_cyclic"][:, b].astype(np.int32),

            "insertion_delta_cyclic": stacked["insertion_delta_cyclic"][:, b].astype(np.float32),
            "nearest_distance": stacked["nearest_distance"][:, b].astype(np.float32),

            "partial_len_before": stacked["partial_len_before"][:, b].astype(np.int32),
            "partial_len_after": stacked["partial_len_after"][:, b].astype(np.int32),
        }

        traces.append(trace)

    return traces


# ============================================================
# L2C-Insert greedy rollout with efficient insertion trace
# ============================================================

def l2c_insert_greedy_rollout_with_trace(
    model,
    env,
    problems,
    device,
    mode,
):
    """
    Run pure greedy L2C-Insert decoding and record insertion trace efficiently.

    This version avoids:
    - copying full partial tours from GPU to CPU at every step
    - Python loop over batch instances at every step
    - searching insertion position on CPU

    Instead, it computes trace tensors on GPU and moves them to CPU
    only once at the end of the batch.
    """
    batch_size, problem_size, _ = problems.shape

    model.eval()
    model.mode = mode

    set_model_dynamic_options(model, problem_size)

    inject_problems_into_env(
        env=env,
        problems=problems,
        device=device,
    )

    try:
        reset_state, _, _ = env.reset(mode)
    except TypeError:
        reset_state, _, _ = env.reset()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(reset_state, device)

    state, reward, reward_student, done = env.pre_step()

    move_tensor_attrs_to_device(env, device)
    move_tensor_attrs_to_device(state, device)

    trace_tensors = make_empty_trace_tensor_lists()

    current_step = 0

    while not done:
        if current_step == 0:
            # Initial remaining nodes: 1, 2, ..., N-1.
            abs_scatter_solu_1 = torch.arange(
                start=1,
                end=problem_size,
                dtype=torch.int64,
                device=device,
            )[None, :].repeat(batch_size, 1)

            # Initial partial solution starts from node 0.
            abs_partial_solu_2 = torch.zeros(
                batch_size,
                1,
                dtype=torch.int64,
                device=device,
            )

            last_node_index = abs_partial_solu_2[:, [-1]]

        else:
            # Clone on GPU only. This avoids CPU transfer and protects
            # old tensors from any possible in-place modification.
            old_partial = env.abs_partial_solu_2.detach().clone()
            old_scatter = env.abs_scatter_solu_1.detach().clone()

            # Encode current endpoint.
            partial_end_node_coor = model.decoder._get_encoding(
                state.data,
                last_node_index.reshape(batch_size, 1),
            )

            # Encode remaining scatter nodes.
            scatter_node_coors = model.decoder._get_encoding(
                state.data,
                old_scatter,
            )

            # Greedy candidate selection:
            # choose nearest remaining node to the current endpoint.
            distances = euclidean_distance(
                scatter_node_coors,
                partial_end_node_coor,
            )

            greedy_index = torch.argmin(
                distances,
                dim=1,
            ).reshape(batch_size, 1)

            greedy_nearest_distance = distances.gather(
                dim=1,
                index=greedy_index,
            ).reshape(batch_size)

            greedy_selected_node = old_scatter.gather(
                dim=1,
                index=greedy_index,
            ).reshape(batch_size)

            # Model updates partial solution and remaining scatter nodes.
            (
                abs_partial_solu_2,
                abs_scatter_solu_1,
                abs_scatter_solu_1_selected,
            ) = model(
                state.data,
                env.solution,
                old_scatter,
                old_partial,
                greedy_index,
                current_step,
                last_node_index,
            )

            abs_partial_solu_2 = abs_partial_solu_2.to(device).long()
            abs_scatter_solu_1 = abs_scatter_solu_1.to(device).long()
            abs_scatter_solu_1_selected = abs_scatter_solu_1_selected.to(device).long()

            selected_nodes = abs_scatter_solu_1_selected.reshape(batch_size).long()

            # Fallback if the model-returned selected node is invalid.
            valid_selected = (
                (selected_nodes >= 0)
                & (selected_nodes < problem_size)
            )

            selected_nodes = torch.where(
                valid_selected,
                selected_nodes,
                greedy_selected_node.long(),
            )

            last_node_index = selected_nodes.reshape(batch_size, 1)

            # Build and append vectorized trace event on GPU.
            event = build_vectorized_insertion_event(
                problems=problems,
                old_partial=old_partial,
                new_partial=abs_partial_solu_2,
                inserted_nodes=selected_nodes,
                nearest_distances=greedy_nearest_distance,
                current_step=current_step,
            )

            append_vectorized_trace_event(
                trace_tensors=trace_tensors,
                event=event,
            )

        current_step += 1

        move_tensor_attrs_to_device(env, device)

        state, reward, reward_student, done = env.step(
            abs_scatter_solu_1,
            abs_partial_solu_2,
            mode=mode,
        )

        move_tensor_attrs_to_device(env, device)
        move_tensor_attrs_to_device(state, device)

    final_tour = env.abs_partial_solu_2.detach().clone()

    insertion_traces = convert_trace_tensor_lists_to_arrays(
        trace_tensors=trace_tensors,
        batch_size=batch_size,
    )

    return final_tour, insertion_traces


# ============================================================
# Batch solving
# ============================================================

def solve_batch(
    model,
    Env,
    batch_coords,
    problem_size,
    device,
    mode,
):
    """
    Solve one batch and return final tours plus insertion traces.
    """
    env = build_l2c_env(
        Env=Env,
        mode=mode,
    )

    problems = torch.tensor(
        np.asarray(batch_coords, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        tours, traces = l2c_insert_greedy_rollout_with_trace(
            model=model,
            env=env,
            problems=problems,
            device=device,
            mode=mode,
        )

    return tours.detach().cpu().numpy(), traces


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
    all_traces = []

    with torch.no_grad():
        for start in range(0, len(coords_group), batch_size):
            batch_coords = coords_group[start:start + batch_size]

            tours, traces = solve_batch(
                model=model,
                Env=Env,
                batch_coords=batch_coords,
                problem_size=problem_size,
                device=device,
                mode=mode,
            )

            all_tours.extend(tours)
            all_traces.extend(traces)

            print(
                f"    Solved {min(start + batch_size, len(coords_group))}/"
                f"{len(coords_group)}"
            )

    return all_tours, all_traces


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
    Evaluate one dataset file and save plot-friendly output with trace.
    """
    filename = os.path.basename(dataset_file)

    print("\n============================================================")
    print("Processing:", filename)

    file_start_time = time.time()
    output_path = os.path.join(output_dir, "results_" + filename)

    try:
        coords_list = load_coordinate_list(dataset_file)

        size_to_indices = group_indices_by_problem_size(coords_list)
        detected_problem_sizes = sorted(size_to_indices.keys())

        print("Detected problem sizes:", detected_problem_sizes)

        final_tours = [None] * len(coords_list)
        final_traces = [None] * len(coords_list)

        for problem_size, indices in sorted(size_to_indices.items()):
            print(f"\n  Problem size: {problem_size}")
            print(f"  Number of instances: {len(indices)}")

            coords_group = [coords_list[idx] for idx in indices]

            tours, traces = solve_coordinate_group(
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
                final_traces[original_idx] = traces[local_idx]

        # This keeps all previous fields:
        # coords, tour, tour_length, is_valid, validation_reason,
        # missing_nodes, duplicate_nodes, out_of_range_nodes,
        # problem_size, start_node, method.
        output_instances, eval_summary = build_plotfriendly_instances(
            coords_list=coords_list,
            tours=final_tours,
            method="L2C_INSERT_GREEDY",
            extra_fields={
                "use_rrc": False,
                "rrc_budget": 0,
                "use_repair": False,
                "use_random_insertion": False,
                "candidate_policy": "nearest_to_current_endpoint",
                "has_insertion_trace": True,
                "trace_format": "compact_numpy_arrays_vectorized_gpu",
            },
        )

        # Add only the new trace field.
        for item, trace in zip(output_instances, final_traces):
            item["insertion_trace"] = trace

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
            method="L2C_INSERT_GREEDY",
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
            method="L2C_INSERT_GREEDY",
            output_path="",
            error_message=str(e),
        )


# ============================================================
# Dataset selection
# ============================================================

def select_dataset_files(data_dir, distribution_keywords=None):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    dataset_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if distribution_keywords is not None:
        normalized_keywords = [keyword.lower() for keyword in distribution_keywords]
        dataset_files = [
            path
            for path in dataset_files
            if any(
                keyword in os.path.basename(path).lower()
                for keyword in normalized_keywords
            )
        ]

    if not dataset_files:
        raise FileNotFoundError(
            "No PKL dataset files matched the requested input and filters."
        )
    return dataset_files


# ============================================================
# Main
# ============================================================

def main():
    """
    Main L2C-Insert greedy evaluation with efficient insertion trace.
    """
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_csv_path = os.path.join(
        args.output_dir,
        args.summary_filename,
    )

    set_seed(args.seed)

    Model, Env = import_l2c_modules(
        l2c_root=args.l2c_root,
    )

    device = setup_torch_device(
        cuda_device_num=args.cuda_device_num,
        set_default_device=True,
        legacy_cuda_default_tensor_type=False,
    )

    print("Device:", device)

    model = build_l2c_model(
        Model=Model,
        checkpoint_path=args.checkpoint_path,
        device=device,
        mode=args.mode,
        k_nearest_edges=args.k_nearest_edges,
        k_nearest_scatter=args.k_nearest_scatter,
        coor_norm=args.coor_norm,
    )

    print("L2C-Insert model loaded.")

    dataset_files = select_dataset_files(
        args.data_dir, args.distribution_keywords
    )
    if args.distribution_keywords is not None:
        print("Dataset filename filters:", args.distribution_keywords)
    else:
        print("Dataset filename filters: none")

    print("Number of selected dataset files:", len(dataset_files))

    for path in dataset_files:
        print("Selected:", os.path.basename(path))

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

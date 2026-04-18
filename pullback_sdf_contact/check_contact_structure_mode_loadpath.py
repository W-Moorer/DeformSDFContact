import argparse
import os
import time

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import build_indenter_state, build_load_schedule
from coupled_solver.staggered import solve_staggered_contact_loadpath


def summarize_result(result, contact_structure_mode, max_outer_iter, relaxation_u, elapsed_s):
    accepted_history = result["accepted_history"]
    attempt_history = result["attempt_history"]
    final_accepted = accepted_history[-1] if accepted_history else None
    return {
        "contact_structure_mode": contact_structure_mode,
        "max_outer_iter": max_outer_iter,
        "relaxation_u": relaxation_u,
        "requested_final_target_load": result["requested_final_target_load"],
        "final_accepted_load": result["final_accepted_load"],
        "reached_final_target": result["reached_final_target"],
        "terminated_early": result["terminated_early"],
        "termination_reason": result["termination_reason"],
        "accepted_step_count": result["accepted_step_count"],
        "attempt_count": result["attempt_count"],
        "total_outer_iterations_accepted": int(
            sum(item["outer_iterations"] for item in accepted_history)
        ),
        "total_outer_iterations_attempts": int(
            sum(item["outer_iterations"] for item in attempt_history)
        ),
        "cutback_count": int(sum(1 for item in attempt_history if item["cutback_triggered"])),
        "final_reaction_norm": 0.0 if final_accepted is None else float(final_accepted["reaction_norm"]),
        "final_max_penetration": 0.0 if final_accepted is None else float(final_accepted["max_penetration"]),
        "elapsed_s": float(elapsed_s),
    }


def truncate_to_common_load(result, common_load):
    accepted_rows = [
        item for item in result["accepted_history"] if item["load_value"] <= common_load + 1e-12
    ]
    if not accepted_rows:
        return {
            "common_load": common_load,
            "accepted_step_count": 0,
            "total_outer_iterations": 0,
            "total_attempts": 0,
            "cutback_count": 0,
            "reaction_norm": 0.0,
            "max_penetration": 0.0,
        }
    last_accepted = accepted_rows[-1]
    attempt_limit = last_accepted["attempt_state_index"]
    attempt_rows = [
        item for item in result["attempt_history"] if item["attempt_state_index"] <= attempt_limit
    ]
    return {
        "common_load": common_load,
        "accepted_step_count": len(accepted_rows),
        "total_outer_iterations": int(sum(item["outer_iterations"] for item in accepted_rows)),
        "total_attempts": len(attempt_rows),
        "cutback_count": int(sum(1 for item in attempt_rows if item["cutback_triggered"])),
        "reaction_norm": float(last_accepted["reaction_norm"]),
        "max_penetration": float(last_accepted["max_penetration"]),
    }


def run_case(mode, contact_structure_mode, max_outer_iter, relaxation_u):
    state, solver_cfg, mode_cfg = build_indenter_state(mode)
    tic = time.perf_counter()
    _, result = solve_staggered_contact_loadpath(
        state,
        solver_cfg,
        build_load_schedule(mode),
        contact_structure_mode=contact_structure_mode,
        max_outer_iter=max_outer_iter,
        max_cutbacks=mode_cfg["max_cutbacks"],
        min_cutback_increment=mode_cfg["min_cutback_increment"],
        relaxation_u=relaxation_u,
        write_outputs=False,
        history_path=None,
        verbose=False,
        step_verbose=False,
    )
    elapsed_s = time.perf_counter() - tic
    summary = summarize_result(result, contact_structure_mode, max_outer_iter, relaxation_u, elapsed_s)
    return result, summary


def tuning_cases_for_mode(mode):
    if mode == "baseline":
        return [
            ("rhs_only", 4, 1.0),
            ("consistent_linearized", 4, 1.0),
            ("consistent_linearized", 8, 1.0),
            ("consistent_linearized", 8, 0.7),
        ]
    if mode == "aggressive":
        return [
            ("rhs_only", 4, 1.0),
            ("consistent_linearized", 4, 1.0),
            ("consistent_linearized", 6, 1.0),
            ("consistent_linearized", 8, 1.0),
            ("consistent_linearized", 10, 1.0),
            ("consistent_linearized", 12, 1.0),
            ("consistent_linearized", 8, 0.7),
            ("consistent_linearized", 8, 0.5),
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def print_complete_table(summaries):
    print("Complete result table:")
    print(
        "  mode | max_outer_iter | relaxation_u | requested_final_target_load | "
        "final_accepted_load | reached_final_target | terminated_early | "
        "termination_reason | accepted_step_count | attempt_count | "
        "total_outer_iterations_accepted | total_outer_iterations_attempts | "
        "cutback_count | final_reaction_norm | final_max_penetration"
    )
    for summary in summaries:
        print(
            f"  {summary['contact_structure_mode']} | {summary['max_outer_iter']} | "
            f"{summary['relaxation_u']:.1f} | {summary['requested_final_target_load']:.6f} | "
            f"{summary['final_accepted_load']:.6f} | {summary['reached_final_target']} | "
            f"{summary['terminated_early']} | {summary['termination_reason']} | "
            f"{summary['accepted_step_count']} | {summary['attempt_count']} | "
            f"{summary['total_outer_iterations_accepted']} | "
            f"{summary['total_outer_iterations_attempts']} | {summary['cutback_count']} | "
            f"{summary['final_reaction_norm']:.6e} | {summary['final_max_penetration']:.6e}"
        )


def print_common_load_table(results, summaries):
    common_load = min(summary["final_accepted_load"] for summary in summaries)
    print(f"Common accepted-load fair comparison (load <= {common_load:.6f}):")
    print(
        "  mode | max_outer_iter | relaxation_u | accepted_step_count | total_outer_iterations | "
        "total_attempts | cutback_count | reaction_norm | max_penetration"
    )
    for result, summary in zip(results, summaries):
        truncated = truncate_to_common_load(result, common_load)
        print(
            f"  {summary['contact_structure_mode']} | {summary['max_outer_iter']} | "
            f"{summary['relaxation_u']:.1f} | {truncated['accepted_step_count']} | "
            f"{truncated['total_outer_iterations']} | {truncated['total_attempts']} | "
            f"{truncated['cutback_count']} | {truncated['reaction_norm']:.6e} | "
            f"{truncated['max_penetration']:.6e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive"], default="baseline")
    args = parser.parse_args()

    results = []
    summaries = []
    for contact_structure_mode, max_outer_iter, relaxation_u in tuning_cases_for_mode(args.mode):
        result, summary = run_case(args.mode, contact_structure_mode, max_outer_iter, relaxation_u)
        results.append(result)
        summaries.append(summary)

    print(f"Contact structure mode loadpath tuning ({args.mode})")
    print("")
    print_complete_table(summaries)
    print("")
    if len({summary['final_accepted_load'] for summary in summaries}) > 1:
        print_common_load_table(results, summaries)


if __name__ == "__main__":
    main()

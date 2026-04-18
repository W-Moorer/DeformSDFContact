import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np

from coupled_solver.monolithic import (
    recommended_monolithic_contact_options,
    solve_monolithic_contact_loadpath,
)
from post.xdmf import write_scalar_field

from check_indenter_block_contact import build_indenter_state, build_load_schedule


def summarize_monolithic_result(result, mode, max_newton_iter, line_search, damping):
    accepted_history = result["accepted_history"]
    final_accepted = accepted_history[-1] if accepted_history else None
    return {
        "mode": mode,
        "line_search": bool(line_search),
        "damping": float(damping),
        "requested_final_target_load": result["requested_final_target_load"],
        "final_accepted_load": result["final_accepted_load"],
        "reached_final_target": result["reached_final_target"],
        "terminated_early": result["terminated_early"],
        "termination_reason": result["termination_reason"],
        "accepted_step_count": result["accepted_step_count"],
        "attempt_count": result["attempt_count"],
        "total_newton_iterations_accepted": int(
            sum(item["newton_iterations"] for item in accepted_history)
        ),
        "total_newton_iterations_attempts": int(
            sum(item["newton_iterations"] for item in result["attempt_history"])
        ),
        "cutback_count": int(sum(1 for item in result["attempt_history"] if item["cutback_triggered"])),
        "final_residual_norm": 0.0 if final_accepted is None else float(final_accepted["residual_norm"]),
        "final_reaction_norm": 0.0 if final_accepted is None else float(final_accepted["reaction_norm"]),
        "final_max_penetration": 0.0 if final_accepted is None else float(final_accepted["max_penetration"]),
        "max_newton_iter": max_newton_iter,
    }


def run_monolithic_case(
    mode,
    *,
    max_newton_iter=None,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=None,
    damping=None,
    max_backtracks=None,
    backtrack_factor=None,
    write_outputs=True,
    verbose=False,
):
    state, solver_cfg, mode_cfg = build_indenter_state(mode)
    load_schedule = build_load_schedule(mode)
    history_path = f"monolithic_history_{mode}.csv"
    recommended = recommended_monolithic_contact_options()
    max_newton_iter = recommended["max_newton_iter"] if max_newton_iter is None else max_newton_iter
    line_search = recommended["line_search"] if line_search is None else bool(line_search)
    damping = recommended["initial_damping"] if damping is None else float(damping)
    max_backtracks = recommended["max_backtracks"] if max_backtracks is None else int(max_backtracks)
    backtrack_factor = (
        recommended["backtrack_factor"] if backtrack_factor is None else float(backtrack_factor)
    )

    final_state, result = solve_monolithic_contact_loadpath(
        state,
        solver_cfg,
        load_schedule,
        max_newton_iter=max_newton_iter,
        max_cutbacks=mode_cfg["max_cutbacks"],
        tol_res=tol_res,
        tol_inc=tol_inc,
        line_search=line_search,
        initial_damping=damping,
        max_backtracks=max_backtracks,
        backtrack_factor=backtrack_factor,
        write_outputs=True,
        history_path=history_path,
        verbose=verbose,
        min_cutback_increment=mode_cfg["min_cutback_increment"],
    )
    summary = summarize_monolithic_result(result, mode, max_newton_iter, line_search, damping)

    phi_delta = final_state["phi"].vector.array_r - final_state["phi0"].vector.array_r
    summary["u_norm"] = float(np.linalg.norm(final_state["u"].vector.array_r))
    summary["phi_delta_norm"] = float(np.linalg.norm(phi_delta))
    summary["history_path"] = history_path

    if write_outputs:
        write_scalar_field(final_state["domain"], final_state["u"], f"output_u_monolithic_{mode}.xdmf")
        write_scalar_field(final_state["domain"], final_state["phi"], f"output_phi_monolithic_{mode}.xdmf")

    return final_state, result, summary


def print_case(result, summary):
    print(f"mode = {summary['mode']}")
    print(f"  line_search = {summary['line_search']}")
    print(f"  damping = {summary['damping']}")
    print(f"  requested_final_target_load = {summary['requested_final_target_load']}")
    print(f"  final_accepted_load = {summary['final_accepted_load']}")
    print(f"  reached_final_target = {summary['reached_final_target']}")
    print(f"  terminated_early = {summary['terminated_early']}")
    print(f"  termination_reason = {summary['termination_reason']}")
    print(f"  accepted_step_count = {summary['accepted_step_count']}")
    print(f"  attempt_count = {summary['attempt_count']}")
    print(f"  total_newton_iterations_accepted = {summary['total_newton_iterations_accepted']}")
    print(f"  total_newton_iterations_attempts = {summary['total_newton_iterations_attempts']}")
    print(f"  cutback_count = {summary['cutback_count']}")
    print(f"  final_residual_norm = {summary['final_residual_norm']}")
    print(f"  final_reaction_norm = {summary['final_reaction_norm']}")
    print(f"  final_max_penetration = {summary['final_max_penetration']}")
    print(f"  ||u||_2 = {summary['u_norm']}")
    print(f"  ||phi - phi0||_2 = {summary['phi_delta_norm']}")
    print("  accepted steps:")
    for item in result["accepted_history"]:
        print(
            "    state={accepted_state_index:02d} step={step:02d} load={load_value:.4f} "
            "newton_iterations={newton_iterations} residual_norm={residual_norm:.6e} "
            "reaction_norm={reaction_norm:.6e} active={active_contact_points} "
            "step_length={step_length:.3f}".format(**item)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    parser.add_argument("--max-newton-iter", type=int, default=None)
    parser.add_argument("--line-search", choices=["on", "off"], default=None)
    parser.add_argument("--damping", type=float, default=None)
    args = parser.parse_args()

    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]
    recommended = recommended_monolithic_contact_options()
    resolved_line_search = (
        recommended["line_search"] if args.line_search is None else args.line_search == "on"
    )
    resolved_max_newton_iter = (
        recommended["max_newton_iter"] if args.max_newton_iter is None else args.max_newton_iter
    )
    resolved_damping = (
        recommended["initial_damping"] if args.damping is None else args.damping
    )
    print("Monolithic contact regression")
    print("")
    print(f"max_newton_iter = {resolved_max_newton_iter}")
    print(f"line_search = {resolved_line_search}")
    print(f"damping = {resolved_damping}")
    print("")
    for mode in modes:
        _, result, summary = run_monolithic_case(
            mode,
            max_newton_iter=resolved_max_newton_iter,
            line_search=resolved_line_search,
            damping=resolved_damping,
            write_outputs=True,
            verbose=False,
        )
        print_case(result, summary)
        print("")


if __name__ == "__main__":
    main()

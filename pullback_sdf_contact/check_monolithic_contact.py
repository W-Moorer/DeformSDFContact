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


def summarize_monolithic_result(
    result,
    mode,
    backend,
    build_path,
    max_newton_iter,
    line_search,
    damping,
    linear_solver_mode,
    ksp_type,
    pc_type,
    block_pc_name,
):
    accepted_history = result["accepted_history"]
    final_accepted = accepted_history[-1] if accepted_history else None
    resolved_linear_solver_mode = (
        linear_solver_mode if final_accepted is None else final_accepted.get("linear_solver_mode", linear_solver_mode)
    )
    resolved_ksp_type = ksp_type if final_accepted is None else final_accepted.get("ksp_type", ksp_type)
    resolved_pc_type = pc_type if final_accepted is None else final_accepted.get("pc_type", pc_type)
    resolved_block_pc_name = (
        block_pc_name if final_accepted is None else final_accepted.get("block_pc_name", block_pc_name)
    )
    total_newton_iterations_accepted = int(
        sum(item["newton_iterations"] for item in accepted_history)
    )
    total_newton_iterations_attempts = int(
        sum(item["newton_iterations"] for item in result["attempt_history"])
    )
    total_linear_iterations_accepted = int(
        sum(item.get("linear_iterations", 0) for item in accepted_history)
    )
    total_linear_iterations_attempts = int(
        sum(item.get("linear_iterations", 0) for item in result["attempt_history"])
    )
    total_assembly_time_accepted = float(
        sum(sum(item.get("assembly_time_list", [])) for item in accepted_history)
    )
    total_block_build_time_accepted = float(
        sum(sum(item.get("block_build_time_list", [])) for item in accepted_history)
    )
    total_linear_solve_time_accepted = float(
        sum(sum(item.get("linear_solve_time_list", [])) for item in accepted_history)
    )
    total_state_update_time_accepted = float(
        sum(sum(item.get("state_update_time_list", [])) for item in accepted_history)
    )
    total_newton_step_walltime_accepted = float(
        sum(sum(item.get("newton_step_walltime_list", [])) for item in accepted_history)
    )
    per_newton_linear_iterations = []
    per_newton_ksp_reasons = []
    per_newton_outer_residual_before = []
    per_newton_outer_residual_after = []
    per_newton_relative_reduction = []
    for item in accepted_history:
        per_newton_linear_iterations.extend(item.get("linear_iterations_list", []))
        per_newton_ksp_reasons.extend(item.get("ksp_reason_list", []))
        per_newton_outer_residual_before.extend(
            item.get("outer_residual_norm_before_linear_list", [])
        )
        per_newton_outer_residual_after.extend(
            item.get("outer_residual_norm_after_linear_list", [])
        )
        per_newton_relative_reduction.extend(
            item.get("relative_linear_reduction_list", [])
        )
    ksp_reason_list = [int(item.get("ksp_reason", 0)) for item in accepted_history]
    ksp_reason_histogram = {}
    for reason in ksp_reason_list:
        key = str(reason)
        ksp_reason_histogram[key] = ksp_reason_histogram.get(key, 0) + 1
    return {
        "mode": mode,
        "backend": backend,
        "build_path": build_path,
        "mesh_resolution": result.get("mesh_resolution", ""),
        "ndof_u": int(result.get("ndof_u", 0)),
        "ndof_phi": int(result.get("ndof_phi", 0)),
        "total_dofs": int(result.get("ndof_u", 0) + result.get("ndof_phi", 0)),
        "line_search": bool(line_search),
        "damping": float(damping),
        "requested_final_target_load": result["requested_final_target_load"],
        "final_accepted_load": result["final_accepted_load"],
        "reached_final_target": result["reached_final_target"],
        "terminated_early": result["terminated_early"],
        "termination_reason": result["termination_reason"],
        "accepted_step_count": result["accepted_step_count"],
        "accepted_nonzero_step_count": int(result.get("accepted_nonzero_step_count", 0)),
        "attempt_count": result["attempt_count"],
        "total_newton_iterations_accepted": total_newton_iterations_accepted,
        "total_newton_iterations_attempts": total_newton_iterations_attempts,
        "cutback_count": int(sum(1 for item in result["attempt_history"] if item["cutback_triggered"])),
        "linear_solver_mode": resolved_linear_solver_mode,
        "ksp_type": resolved_ksp_type,
        "pc_type": resolved_pc_type,
        "block_pc_name": resolved_block_pc_name,
        "total_linear_iterations_accepted": total_linear_iterations_accepted,
        "total_linear_iterations_attempts": total_linear_iterations_attempts,
        "avg_linear_iterations_per_newton": (
            0.0
            if total_newton_iterations_accepted == 0
            else total_linear_iterations_accepted / total_newton_iterations_accepted
        ),
        "total_assembly_time_accepted": total_assembly_time_accepted,
        "total_block_build_time_accepted": total_block_build_time_accepted,
        "total_linear_solve_time_accepted": total_linear_solve_time_accepted,
        "total_state_update_time_accepted": total_state_update_time_accepted,
        "total_newton_step_walltime_accepted": total_newton_step_walltime_accepted,
        "final_residual_norm": 0.0 if final_accepted is None else float(final_accepted["residual_norm"]),
        "final_reaction_norm": 0.0 if final_accepted is None else float(final_accepted["reaction_norm"]),
        "final_max_penetration": 0.0 if final_accepted is None else float(final_accepted["max_penetration"]),
        "max_newton_iter": max_newton_iter,
        "ksp_reason_list": ksp_reason_list,
        "ksp_reason_histogram": ksp_reason_histogram,
        "per_newton_linear_iterations": per_newton_linear_iterations,
        "per_newton_ksp_reasons": per_newton_ksp_reasons,
        "per_newton_outer_residual_before": per_newton_outer_residual_before,
        "per_newton_outer_residual_after": per_newton_outer_residual_after,
        "per_newton_relative_reduction": per_newton_relative_reduction,
        "completed_full_run": bool(result.get("completed_full_run", False)),
        "terminated_by_walltime": bool(result.get("terminated_by_walltime", False)),
        "terminated_by_step_limit": bool(result.get("terminated_by_step_limit", False)),
        "terminated_by_nonconvergence": bool(result.get("terminated_by_nonconvergence", False)),
        "termination_category": result.get("termination_category", ""),
        "total_walltime": float(result.get("total_walltime", 0.0)),
        "total_newton_steps_used": int(result.get("total_newton_steps_used", 0)),
    }


def run_monolithic_case(
    mode,
    *,
    mesh_scale="small",
    nx=None,
    ny=None,
    nz=None,
    backend=None,
    build_path="current",
    linear_solver_mode=None,
    ksp_type=None,
    pc_type=None,
    block_pc_name=None,
    reuse_ksp=False,
    reuse_matrix_pattern=False,
    reuse_fieldsplit_is=False,
    ksp_rtol=None,
    ksp_atol=None,
    ksp_max_it=None,
    max_newton_iter=None,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=None,
    damping=None,
    max_backtracks=None,
    backtrack_factor=None,
    max_load_steps=None,
    max_newton_steps=None,
    max_walltime_seconds=None,
    stop_after_first_nonzero_accepted_step=False,
    profile_assembly_detail=False,
    write_outputs=True,
    verbose=False,
):
    state, solver_cfg, mode_cfg = build_indenter_state(
        mode,
        mesh_scale=mesh_scale,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    load_schedule = build_load_schedule(mode, mesh_scale=mesh_scale, nx=nx, ny=ny, nz=nz)
    history_path = None
    recommended = recommended_monolithic_contact_options()
    backend = recommended["backend"] if backend is None else backend
    linear_solver_mode = (
        recommended["linear_solver_mode"] if linear_solver_mode is None else linear_solver_mode
    )
    ksp_type = recommended["ksp_type"] if ksp_type is None else ksp_type
    pc_type = recommended["pc_type"] if pc_type is None else pc_type
    block_pc_name = recommended["block_pc_name"] if block_pc_name is None else block_pc_name
    if linear_solver_mode == "lu":
        block_pc_name = "global_lu"
    elif pc_type != "fieldsplit" and block_pc_name.startswith("fieldsplit_"):
        block_pc_name = f"global_{pc_type}"
    ksp_rtol = recommended["ksp_rtol"] if ksp_rtol is None else float(ksp_rtol)
    ksp_atol = recommended["ksp_atol"] if ksp_atol is None else float(ksp_atol)
    ksp_max_it = recommended["ksp_max_it"] if ksp_max_it is None else int(ksp_max_it)
    max_newton_iter = recommended["max_newton_iter"] if max_newton_iter is None else max_newton_iter
    line_search = recommended["line_search"] if line_search is None else bool(line_search)
    damping = recommended["initial_damping"] if damping is None else float(damping)
    max_backtracks = recommended["max_backtracks"] if max_backtracks is None else int(max_backtracks)
    backtrack_factor = (
        recommended["backtrack_factor"] if backtrack_factor is None else float(backtrack_factor)
    )

    mesh_resolution = state.get("mesh_resolution", "")
    history_path = (
        f"monolithic_history_{mode}_{mesh_resolution}_{backend}_{linear_solver_mode}.csv"
    )
    final_state, result = solve_monolithic_contact_loadpath(
        state,
        solver_cfg,
        load_schedule,
        backend=backend,
        build_path=build_path,
        linear_solver_mode=linear_solver_mode,
        ksp_type=ksp_type,
        pc_type=pc_type,
        block_pc_name=block_pc_name,
        reuse_ksp=reuse_ksp,
        reuse_matrix_pattern=reuse_matrix_pattern,
        reuse_fieldsplit_is=reuse_fieldsplit_is,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
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
        max_load_steps=max_load_steps,
        max_newton_steps=max_newton_steps,
        max_walltime_seconds=max_walltime_seconds,
        stop_after_first_nonzero_accepted_step=stop_after_first_nonzero_accepted_step,
        profile_assembly_detail=profile_assembly_detail,
    )
    summary = summarize_monolithic_result(
        result,
        mode,
        backend,
        build_path,
        max_newton_iter,
        line_search,
        damping,
        linear_solver_mode,
        ksp_type,
        pc_type,
        block_pc_name,
    )

    phi_delta = final_state["phi"].vector.array_r - final_state["phi0"].vector.array_r
    summary["u_norm"] = float(np.linalg.norm(final_state["u"].vector.array_r))
    summary["phi_delta_norm"] = float(np.linalg.norm(phi_delta))
    summary["history_path"] = history_path
    summary["mesh_resolution"] = mesh_resolution

    if write_outputs:
        write_scalar_field(
            final_state["domain"],
            final_state["u"],
            f"output_u_monolithic_{mode}_{mesh_resolution}_{backend}_{linear_solver_mode}.xdmf",
        )
        write_scalar_field(
            final_state["domain"],
            final_state["phi"],
            f"output_phi_monolithic_{mode}_{mesh_resolution}_{backend}_{linear_solver_mode}.xdmf",
        )

    return final_state, result, summary


def print_case(result, summary):
    print(f"mode = {summary['mode']}")
    print(f"  mesh_resolution = {summary['mesh_resolution']}")
    print(f"  ndof_u = {summary['ndof_u']}")
    print(f"  ndof_phi = {summary['ndof_phi']}")
    print(f"  total_dofs = {summary['total_dofs']}")
    print(f"  backend = {summary['backend']}")
    print(f"  build_path = {summary['build_path']}")
    print(f"  linear_solver_mode = {summary['linear_solver_mode']}")
    print(f"  ksp_type = {summary['ksp_type']}")
    print(f"  pc_type = {summary['pc_type']}")
    print(f"  block_pc_name = {summary['block_pc_name']}")
    print(f"  line_search = {summary['line_search']}")
    print(f"  damping = {summary['damping']}")
    print(f"  requested_final_target_load = {summary['requested_final_target_load']}")
    print(f"  final_accepted_load = {summary['final_accepted_load']}")
    print(f"  reached_final_target = {summary['reached_final_target']}")
    print(f"  terminated_early = {summary['terminated_early']}")
    print(f"  termination_reason = {summary['termination_reason']}")
    print(f"  accepted_step_count = {summary['accepted_step_count']}")
    print(f"  accepted_nonzero_step_count = {summary['accepted_nonzero_step_count']}")
    print(f"  attempt_count = {summary['attempt_count']}")
    print(f"  total_newton_iterations_accepted = {summary['total_newton_iterations_accepted']}")
    print(f"  total_newton_iterations_attempts = {summary['total_newton_iterations_attempts']}")
    print(f"  total_linear_iterations_accepted = {summary['total_linear_iterations_accepted']}")
    print(f"  total_linear_iterations_attempts = {summary['total_linear_iterations_attempts']}")
    print(f"  avg_linear_iterations_per_newton = {summary['avg_linear_iterations_per_newton']}")
    print(f"  total_assembly_time_accepted = {summary['total_assembly_time_accepted']}")
    print(f"  total_block_build_time_accepted = {summary['total_block_build_time_accepted']}")
    print(f"  total_linear_solve_time_accepted = {summary['total_linear_solve_time_accepted']}")
    print(f"  total_state_update_time_accepted = {summary['total_state_update_time_accepted']}")
    print(f"  total_newton_step_walltime_accepted = {summary['total_newton_step_walltime_accepted']}")
    print(f"  completed_full_run = {summary['completed_full_run']}")
    print(f"  terminated_by_walltime = {summary['terminated_by_walltime']}")
    print(f"  terminated_by_step_limit = {summary['terminated_by_step_limit']}")
    print(f"  terminated_by_nonconvergence = {summary['terminated_by_nonconvergence']}")
    print(f"  termination_category = {summary['termination_category']}")
    print(f"  total_walltime = {summary['total_walltime']}")
    print(f"  cutback_count = {summary['cutback_count']}")
    print(f"  final_residual_norm = {summary['final_residual_norm']}")
    print(f"  final_reaction_norm = {summary['final_reaction_norm']}")
    print(f"  final_max_penetration = {summary['final_max_penetration']}")
    print(f"  ||u||_2 = {summary['u_norm']}")
    print(f"  ||phi - phi0||_2 = {summary['phi_delta_norm']}")
    print(f"  ksp_reason_list = {summary['ksp_reason_list']}")
    print(f"  ksp_reason_histogram = {summary['ksp_reason_histogram']}")
    print(f"  per_newton_linear_iterations = {summary['per_newton_linear_iterations']}")
    print(f"  per_newton_ksp_reasons = {summary['per_newton_ksp_reasons']}")
    print(f"  per_newton_outer_residual_before = {summary['per_newton_outer_residual_before']}")
    print(f"  per_newton_outer_residual_after = {summary['per_newton_outer_residual_after']}")
    print(f"  per_newton_relative_reduction = {summary['per_newton_relative_reduction']}")
    print(f"  history_path = {summary['history_path']}")
    print("  accepted steps:")
    for item in result["accepted_history"]:
        print(
            "    state={accepted_state_index:02d} step={step:02d} load={load_value:.4f} "
            "newton_iterations={newton_iterations} residual_norm={residual_norm:.6e} "
            "reaction_norm={reaction_norm:.6e} active={active_contact_points} "
            "linear_iterations={linear_iterations} linear_iterations_list={linear_iterations_list} "
            "ksp_reason_list={ksp_reason_list} reduction_list={relative_linear_reduction_list} "
            "step_length={step_length:.3f}".format(**item)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    parser.add_argument("--mesh-scale", choices=["small", "larger", "xlarger"], default="small")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--backend", choices=["dense", "petsc_block"], default=None)
    parser.add_argument("--linear-solver", choices=["lu", "krylov"], default=None)
    parser.add_argument("--max-newton-iter", type=int, default=None)
    parser.add_argument("--line-search", choices=["on", "off"], default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--pc-type", default=None)
    parser.add_argument("--block-pc-name", default=None)
    parser.add_argument("--ksp-rtol", type=float, default=None)
    parser.add_argument("--ksp-atol", type=float, default=None)
    parser.add_argument("--ksp-max-it", type=int, default=None)
    parser.add_argument("--max-load-steps", type=int, default=None)
    parser.add_argument("--max-newton-steps", type=int, default=None)
    parser.add_argument("--max-walltime-seconds", type=float, default=None)
    parser.add_argument("--stop-after-first-nonzero-accepted-step", action="store_true")
    args = parser.parse_args()

    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]
    recommended = recommended_monolithic_contact_options()
    resolved_line_search = (
        recommended["line_search"] if args.line_search is None else args.line_search == "on"
    )
    resolved_max_newton_iter = (
        recommended["max_newton_iter"] if args.max_newton_iter is None else args.max_newton_iter
    )
    resolved_backend = recommended["backend"] if args.backend is None else args.backend
    resolved_linear_solver = (
        recommended["linear_solver_mode"] if args.linear_solver is None else args.linear_solver
    )
    resolved_damping = (
        recommended["initial_damping"] if args.damping is None else args.damping
    )
    resolved_ksp_type = recommended["ksp_type"] if args.ksp_type is None else args.ksp_type
    resolved_pc_type = recommended["pc_type"] if args.pc_type is None else args.pc_type
    resolved_block_pc_name = (
        recommended["block_pc_name"] if args.block_pc_name is None else args.block_pc_name
    )
    resolved_ksp_rtol = recommended["ksp_rtol"] if args.ksp_rtol is None else args.ksp_rtol
    resolved_ksp_atol = recommended["ksp_atol"] if args.ksp_atol is None else args.ksp_atol
    resolved_ksp_max_it = (
        recommended["ksp_max_it"] if args.ksp_max_it is None else args.ksp_max_it
    )
    print("Monolithic contact regression")
    print("")
    print(f"mesh_scale = {args.mesh_scale}")
    if args.nx is not None or args.ny is not None or args.nz is not None:
        print(f"mesh_override = ({args.nx}, {args.ny}, {args.nz})")
    print(f"backend = {resolved_backend}")
    print(f"linear_solver_mode = {resolved_linear_solver}")
    print(f"ksp_type = {resolved_ksp_type}")
    print(f"pc_type = {resolved_pc_type}")
    print(f"block_pc_name = {resolved_block_pc_name}")
    print(f"max_newton_iter = {resolved_max_newton_iter}")
    print(f"max_load_steps = {args.max_load_steps}")
    print(f"max_newton_steps = {args.max_newton_steps}")
    print(f"max_walltime_seconds = {args.max_walltime_seconds}")
    print(
        f"stop_after_first_nonzero_accepted_step = "
        f"{args.stop_after_first_nonzero_accepted_step}"
    )
    print(f"line_search = {resolved_line_search}")
    print(f"damping = {resolved_damping}")
    print("")
    for mode in modes:
        _, result, summary = run_monolithic_case(
            mode,
            mesh_scale=args.mesh_scale,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            backend=resolved_backend,
            linear_solver_mode=resolved_linear_solver,
            ksp_type=resolved_ksp_type,
            pc_type=resolved_pc_type,
            block_pc_name=resolved_block_pc_name,
            ksp_rtol=resolved_ksp_rtol,
            ksp_atol=resolved_ksp_atol,
            ksp_max_it=resolved_ksp_max_it,
            max_newton_iter=resolved_max_newton_iter,
            line_search=resolved_line_search,
            damping=resolved_damping,
            max_load_steps=args.max_load_steps,
            max_newton_steps=args.max_newton_steps,
            max_walltime_seconds=args.max_walltime_seconds,
            stop_after_first_nonzero_accepted_step=args.stop_after_first_nonzero_accepted_step,
            write_outputs=True,
            verbose=False,
        )
        print_case(result, summary)
        print("")


if __name__ == "__main__":
    main()

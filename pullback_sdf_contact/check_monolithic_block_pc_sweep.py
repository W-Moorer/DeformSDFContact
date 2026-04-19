import argparse
import csv
import json
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import (
    get_petsc_runtime_info,
    monolithic_block_pc_names,
    recommended_monolithic_contact_options,
)


def build_cases(include_aggressive=False, include_6x6x6=False):
    cases = [
        ("baseline_3x3x3", "baseline", {"nx": 3, "ny": 3, "nz": 3}),
        ("baseline_4x4x4", "baseline", {"nx": 4, "ny": 4, "nz": 4}),
        ("baseline_5x5x5", "baseline", {"nx": 5, "ny": 5, "nz": 5}),
    ]
    if include_6x6x6:
        cases.append(("baseline_6x6x6", "baseline", {"nx": 6, "ny": 6, "nz": 6}))
    if include_aggressive:
        cases.append(("aggressive_5x5x5", "aggressive", {"nx": 5, "ny": 5, "nz": 5}))
    return cases


def build_solver_configs(include_additive=True, include_symmetric=True, include_schur=True):
    configs = [
        {
            "label": "reference_lu",
            "linear_solver_mode": "lu",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "block_pc_name": "global_lu",
        },
        {
            "label": "gmres_ilu",
            "linear_solver_mode": "krylov",
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "block_pc_name": "global_ilu",
        },
    ]
    if include_additive:
        configs.append(
            {
                "label": "gmres_fieldsplit_additive_ilu",
                "linear_solver_mode": "krylov",
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "block_pc_name": "fieldsplit_additive_ilu",
            }
        )
    configs.append(
        {
            "label": "gmres_fieldsplit_multiplicative_ilu",
            "linear_solver_mode": "krylov",
            "ksp_type": "gmres",
            "pc_type": "fieldsplit",
            "block_pc_name": "fieldsplit_multiplicative_ilu",
        }
    )
    if include_symmetric:
        configs.append(
            {
                "label": "gmres_fieldsplit_symmetric_multiplicative_ilu",
                "linear_solver_mode": "krylov",
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "block_pc_name": "fieldsplit_symmetric_multiplicative_ilu",
            }
        )
    if include_schur:
        configs.append(
            {
                "label": "gmres_fieldsplit_schur_lower_selfp_ilu",
                "linear_solver_mode": "krylov",
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "block_pc_name": "fieldsplit_schur_lower_selfp_ilu",
            }
        )
    return configs


def summarize_row(case_label, mode, mesh_kwargs, cfg, summary):
    return {
        "case_label": case_label,
        "mode": mode,
        "mesh_resolution": summary["mesh_resolution"],
        "nx": mesh_kwargs["nx"],
        "ny": mesh_kwargs["ny"],
        "nz": mesh_kwargs["nz"],
        "ndof_u": summary["ndof_u"],
        "ndof_phi": summary["ndof_phi"],
        "total_dofs": summary["total_dofs"],
        "solver_label": cfg["label"],
        "backend": summary["backend"],
        "linear_solver_mode": summary["linear_solver_mode"],
        "ksp_type": summary["ksp_type"],
        "pc_type": summary["pc_type"],
        "block_pc_name": summary["block_pc_name"],
        "converged": summary["reached_final_target"],
        "terminated_early": summary["terminated_early"],
        "termination_reason": summary["termination_reason"],
        "requested_final_target_load": summary["requested_final_target_load"],
        "final_accepted_load": summary["final_accepted_load"],
        "accepted_step_count": summary["accepted_step_count"],
        "attempt_count": summary["attempt_count"],
        "total_newton_iterations": summary["total_newton_iterations_accepted"],
        "total_linear_iterations": summary["total_linear_iterations_accepted"],
        "avg_linear_iterations_per_newton": summary["avg_linear_iterations_per_newton"],
        "cutback_count": summary["cutback_count"],
        "final_residual_norm": summary["final_residual_norm"],
        "final_reaction_norm": summary["final_reaction_norm"],
        "final_max_penetration": summary["final_max_penetration"],
        "u_norm": summary["u_norm"],
        "phi_delta_norm": summary["phi_delta_norm"],
        "ksp_reason_histogram": json.dumps(summary["ksp_reason_histogram"], sort_keys=True),
        "ksp_reason_list": json.dumps(summary["ksp_reason_list"]),
        "history_path": summary["history_path"],
    }


def print_case_header(case_label, mode, mesh_kwargs):
    print(f"case = {case_label}")
    print(f"  mode = {mode}")
    print(f"  mesh = {mesh_kwargs['nx']}x{mesh_kwargs['ny']}x{mesh_kwargs['nz']}")
    print(
        "  solver | converged | ndof_u | ndof_phi | total_newton | total_linear | "
        "avg_linear_per_newton | cutbacks | final_residual | final_reaction | "
        "final_max_penetration | ksp_reason_histogram"
    )


def print_row(row):
    print(
        "  {solver_label} | {converged} | {ndof_u} | {ndof_phi} | {total_newton_iterations} | "
        "{total_linear_iterations} | {avg_linear_iterations_per_newton:.3f} | {cutback_count} | "
        "{final_residual_norm:.6e} | {final_reaction_norm:.6e} | {final_max_penetration:.6e} | "
        "{ksp_reason_histogram}".format(**row)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-aggressive", action="store_true")
    parser.add_argument("--include-6x6x6", action="store_true")
    parser.add_argument("--skip-additive", action="store_true")
    parser.add_argument("--skip-symmetric", action="store_true")
    parser.add_argument("--skip-schur", action="store_true")
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    runtime = get_petsc_runtime_info()
    cases = build_cases(
        include_aggressive=args.include_aggressive,
        include_6x6x6=args.include_6x6x6,
    )
    solver_configs = build_solver_configs(
        include_additive=not args.skip_additive,
        include_symmetric=not args.skip_symmetric,
        include_schur=not args.skip_schur,
    )

    rows = []
    failures = []

    print("Monolithic block preconditioner sweep")
    print("")
    print(f"petsc_version = {runtime['petsc_version']}")
    print(f"available_block_pc_names = {runtime['block_pc_names']}")
    print(f"backend = {recommended['backend']}")
    print(f"max_newton_iter = {recommended['max_newton_iter']}")
    print(f"line_search = {recommended['line_search']}")
    print(f"initial_damping = {recommended['initial_damping']}")
    print("")

    for case_label, mode, mesh_kwargs in cases:
        print_case_header(case_label, mode, mesh_kwargs)
        for cfg in solver_configs:
            try:
                _, _, summary = run_monolithic_case(
                    mode,
                    backend=recommended["backend"],
                    mesh_scale="small",
                    nx=mesh_kwargs["nx"],
                    ny=mesh_kwargs["ny"],
                    nz=mesh_kwargs["nz"],
                    linear_solver_mode=cfg["linear_solver_mode"],
                    ksp_type=cfg["ksp_type"],
                    pc_type=cfg["pc_type"],
                    block_pc_name=cfg["block_pc_name"],
                    ksp_rtol=recommended["ksp_rtol"],
                    ksp_atol=recommended["ksp_atol"],
                    ksp_max_it=recommended["ksp_max_it"],
                    max_newton_iter=recommended["max_newton_iter"],
                    line_search=recommended["line_search"],
                    damping=recommended["initial_damping"],
                    write_outputs=False,
                    verbose=False,
                )
                row = summarize_row(case_label, mode, mesh_kwargs, cfg, summary)
                rows.append(row)
                print_row(row)
            except Exception as exc:
                failure = {
                    "case_label": case_label,
                    "mode": mode,
                    "mesh_resolution": f"{mesh_kwargs['nx']}x{mesh_kwargs['ny']}x{mesh_kwargs['nz']}",
                    "solver_label": cfg["label"],
                    "linear_solver_mode": cfg["linear_solver_mode"],
                    "ksp_type": cfg["ksp_type"],
                    "pc_type": cfg["pc_type"],
                    "block_pc_name": cfg["block_pc_name"],
                    "error": str(exc),
                }
                failures.append(failure)
                print(
                    "  {solver_label} | FAILED | error = {error}".format(**failure)
                )
        print("")

    csv_path = "monolithic_block_pc_sweep_summary.csv"
    json_path = "monolithic_block_pc_sweep_summary.json"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "runtime": runtime,
                "recommended": recommended,
                "available_block_pc_names": monolithic_block_pc_names(),
                "rows": rows,
                "failures": failures,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"csv_path = {csv_path}")
    print(f"json_path = {json_path}")
    if failures:
        print("failures:")
        for failure in failures:
            print(
                "  {case_label} | {solver_label} | {mesh_resolution} | {error}".format(
                    **failure
                )
            )


if __name__ == "__main__":
    main()

import csv
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import recommended_monolithic_contact_options


def main():
    recommended = recommended_monolithic_contact_options()
    rows = []
    baseline_configs = [
        ("reference_lu", "lu", "preonly", "lu", "global_lu"),
        (
            "scalable_prep_krylov",
            "krylov",
            recommended["ksp_type"],
            recommended["pc_type"],
            recommended["block_pc_name"],
        ),
    ]
    for label, linear_solver_mode, ksp_type, pc_type, block_pc_name in baseline_configs:
        for mode in ("baseline", "aggressive"):
            _, _, summary = run_monolithic_case(
                mode,
                backend=recommended["backend"],
                linear_solver_mode=linear_solver_mode,
                ksp_type=ksp_type,
                pc_type=pc_type,
                block_pc_name=block_pc_name,
                ksp_rtol=recommended["ksp_rtol"],
                ksp_atol=recommended["ksp_atol"],
                ksp_max_it=recommended["ksp_max_it"],
                max_newton_iter=recommended["max_newton_iter"],
                line_search=recommended["line_search"],
                damping=recommended["initial_damping"],
                write_outputs=True,
                verbose=False,
            )
            row = {
                "baseline_label": label,
                "mode": mode,
                "mesh_resolution": summary["mesh_resolution"],
                "ndof_u": summary["ndof_u"],
                "ndof_phi": summary["ndof_phi"],
                "total_dofs": summary["total_dofs"],
                "backend": summary["backend"],
                "linear_solver_mode": summary["linear_solver_mode"],
                "ksp_type": summary["ksp_type"],
                "pc_type": summary["pc_type"],
                "block_pc_name": summary["block_pc_name"],
                "max_newton_iter": summary["max_newton_iter"],
                "line_search": summary["line_search"],
                "damping": summary["damping"],
                "requested_final_target_load": summary["requested_final_target_load"],
                "final_accepted_load": summary["final_accepted_load"],
                "reached_final_target": summary["reached_final_target"],
                "terminated_early": summary["terminated_early"],
                "termination_reason": summary["termination_reason"],
                "attempt_count": summary["attempt_count"],
                "accepted_step_count": summary["accepted_step_count"],
                "total_newton_iterations": summary["total_newton_iterations_accepted"],
                "total_linear_iterations": summary["total_linear_iterations_accepted"],
                "avg_linear_iterations_per_newton": summary["avg_linear_iterations_per_newton"],
                "ksp_reason_histogram": summary["ksp_reason_histogram"],
                "cutback_count": summary["cutback_count"],
                "final_residual_norm": summary["final_residual_norm"],
                "final_reaction_norm": summary["final_reaction_norm"],
                "final_max_penetration": summary["final_max_penetration"],
                "u_norm": summary["u_norm"],
                "phi_delta_norm": summary["phi_delta_norm"],
                "history_path": summary["history_path"],
            }
            rows.append(row)

    with open("monolithic_baseline_summary.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Monolithic baseline regression")
    print("")
    print(f"backend = {recommended['backend']}")
    print(f"max_newton_iter = {recommended['max_newton_iter']}")
    print(f"line_search = {recommended['line_search']}")
    print(f"damping = {recommended['initial_damping']}")
    print(f"formal reference linear solver = lu")
    print(
        "scalable-prep linear solver = "
        f"krylov ({recommended['ksp_type']} + {recommended['pc_type']})"
    )
    print("")
    for row in rows:
        print(f"baseline_label = {row['baseline_label']}")
        print(f"mode = {row['mode']}")
        print(f"  mesh_resolution = {row['mesh_resolution']}")
        print(f"  ndof_u = {row['ndof_u']}")
        print(f"  ndof_phi = {row['ndof_phi']}")
        print(f"  total_dofs = {row['total_dofs']}")
        print(f"  backend = {row['backend']}")
        print(f"  linear_solver_mode = {row['linear_solver_mode']}")
        print(f"  ksp_type = {row['ksp_type']}")
        print(f"  pc_type = {row['pc_type']}")
        print(f"  block_pc_name = {row['block_pc_name']}")
        print(f"  requested_final_target_load = {row['requested_final_target_load']}")
        print(f"  final_accepted_load = {row['final_accepted_load']}")
        print(f"  reached_final_target = {row['reached_final_target']}")
        print(f"  terminated_early = {row['terminated_early']}")
        print(f"  termination_reason = {row['termination_reason']}")
        print(f"  attempt_count = {row['attempt_count']}")
        print(f"  accepted_step_count = {row['accepted_step_count']}")
        print(f"  total_newton_iterations = {row['total_newton_iterations']}")
        print(f"  total_linear_iterations = {row['total_linear_iterations']}")
        print(f"  avg_linear_iterations_per_newton = {row['avg_linear_iterations_per_newton']}")
        print(f"  ksp_reason_histogram = {row['ksp_reason_histogram']}")
        print(f"  cutback_count = {row['cutback_count']}")
        print(f"  final_residual_norm = {row['final_residual_norm']}")
        print(f"  final_reaction_norm = {row['final_reaction_norm']}")
        print(f"  final_max_penetration = {row['final_max_penetration']}")
        print(f"  ||u||_2 = {row['u_norm']}")
        print(f"  ||phi - phi0||_2 = {row['phi_delta_norm']}")
        print(f"  history_path = {row['history_path']}")
        print("")


if __name__ == "__main__":
    main()

import csv
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import recommended_monolithic_contact_options


def main():
    recommended = recommended_monolithic_contact_options()
    rows = []
    for mode in ("baseline", "aggressive"):
        _, _, summary = run_monolithic_case(
            mode,
            backend=recommended["backend"],
            max_newton_iter=recommended["max_newton_iter"],
            line_search=recommended["line_search"],
            damping=recommended["initial_damping"],
            write_outputs=True,
            verbose=False,
        )
        row = {
            "mode": mode,
            "backend": summary["backend"],
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
    print("")
    for row in rows:
        print(f"mode = {row['mode']}")
        print(f"  backend = {row['backend']}")
        print(f"  requested_final_target_load = {row['requested_final_target_load']}")
        print(f"  final_accepted_load = {row['final_accepted_load']}")
        print(f"  reached_final_target = {row['reached_final_target']}")
        print(f"  terminated_early = {row['terminated_early']}")
        print(f"  termination_reason = {row['termination_reason']}")
        print(f"  attempt_count = {row['attempt_count']}")
        print(f"  accepted_step_count = {row['accepted_step_count']}")
        print(f"  total_newton_iterations = {row['total_newton_iterations']}")
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

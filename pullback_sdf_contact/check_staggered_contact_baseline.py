import csv
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import run_case
from coupled_solver.staggered import recommended_staggered_contact_options


def main():
    recommended = recommended_staggered_contact_options()
    rows = []
    for mode in ("baseline", "aggressive"):
        _, _, summary, history_path, summary_path = run_case(
            mode,
            contact_structure_mode=recommended["contact_structure_mode"],
            max_outer_iter=recommended["max_outer_iter"],
            relaxation_u=recommended["relaxation_u"],
        )
        row = {
            "mode": mode,
            "contact_structure_mode": summary["contact_structure_mode"],
            "max_outer_iter": summary["max_outer_iter"],
            "relaxation_u": summary["relaxation_u"],
            "requested_final_target_load": summary["requested_final_target_load"],
            "final_accepted_load": summary["final_accepted_load"],
            "reached_final_target": summary["reached_final_target"],
            "terminated_early": summary["terminated_early"],
            "termination_reason": summary["termination_reason"],
            "accepted_step_count": summary["accepted_step_count"],
            "attempt_count": summary["attempt_count"],
            "total_outer_iterations": summary["total_outer_iterations_accepted"],
            "cutback_count": summary["cutback_count"],
            "final_reaction_norm": summary["final_reaction_norm"],
            "final_max_penetration": summary["final_max_penetration"],
            "history_path": history_path,
            "summary_path": summary_path,
        }
        rows.append(row)

    with open("staggered_baseline_summary.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Staggered contact baseline regression")
    print("")
    print(f"contact_structure_mode = {recommended['contact_structure_mode']}")
    print(f"max_outer_iter = {recommended['max_outer_iter']}")
    print(f"relaxation_u = {recommended['relaxation_u']}")
    print("")
    for row in rows:
        print(f"mode = {row['mode']}")
        print(f"  requested_final_target_load = {row['requested_final_target_load']}")
        print(f"  final_accepted_load = {row['final_accepted_load']}")
        print(f"  reached_final_target = {row['reached_final_target']}")
        print(f"  terminated_early = {row['terminated_early']}")
        print(f"  accepted_step_count = {row['accepted_step_count']}")
        print(f"  attempt_count = {row['attempt_count']}")
        print(f"  total_outer_iterations = {row['total_outer_iterations']}")
        print(f"  cutback_count = {row['cutback_count']}")
        print(f"  final_reaction_norm = {row['final_reaction_norm']}")
        print(f"  final_max_penetration = {row['final_max_penetration']}")
        print(f"  history_path = {row['history_path']}")
        print(f"  summary_path = {row['summary_path']}")
        print("")


if __name__ == "__main__":
    main()

import csv
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import build_indenter_state
from coupled_solver.staggered import solve_staggered_contact_loadpath


def run_case(case_name, mode, load_schedule, *, use_contact_tangent_uu, max_outer_iter, max_cutbacks):
    state, solver_cfg, mode_cfg = build_indenter_state(mode)
    _, result = solve_staggered_contact_loadpath(
        state,
        solver_cfg,
        load_schedule,
        use_contact_tangent_uu=use_contact_tangent_uu,
        max_outer_iter=max_outer_iter,
        max_cutbacks=max_cutbacks,
        min_cutback_increment=mode_cfg["min_cutback_increment"],
        write_outputs=False,
        history_path=None,
        verbose=True,
        step_verbose=False,
    )
    return result


def write_cutback_history(all_rows, csv_path):
    fieldnames = [
        "case",
        "step",
        "target_load",
        "attempt_load",
        "converged",
        "outer_iterations",
        "cutback_level",
        "cutback_triggered",
        "cutback_reason",
        "accepted_load_before_step",
        "load_increment",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_cutback_summary(summary_lines, path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))
        handle.write("\n")


def main():
    cases = [
        (
            "normal_no_cutback",
            "baseline",
            [
                {"step": 1, "load_value": 0.0},
                {"step": 2, "load_value": 0.01},
                {"step": 3, "load_value": 0.02},
            ],
            True,
            12,
            4,
        ),
        (
            "cutback_recovery",
            "baseline",
            [
                {"step": 1, "load_value": 0.0},
                {"step": 2, "load_value": 0.03},
            ],
            False,
            3,
            4,
        ),
        (
            "zero_increment_first_step_failure",
            "baseline",
            [
                {"step": 1, "load_value": 0.0},
            ],
            True,
            1,
            4,
        ),
    ]

    all_rows = []
    summary_lines = ["Cutback Robustness Summary", ""]

    print("Cutback robustness check")
    print("")
    for case_name, mode, load_schedule, use_contact_tangent_uu, max_outer_iter, max_cutbacks in cases:
        print(f"case = {case_name}")
        result = run_case(
            case_name,
            mode,
            load_schedule,
            use_contact_tangent_uu=use_contact_tangent_uu,
            max_outer_iter=max_outer_iter,
            max_cutbacks=max_cutbacks,
        )
        for row in result["attempt_history"]:
            print(
                "  step={step} target={target_load} attempt={attempt_load} "
                "accepted={accepted} converged={converged} cutback_level={cutback_level} "
                "cutback_triggered={cutback_triggered} reason={cutback_reason}".format(
                    **row
                )
            )
            record = dict(row)
            record["case"] = case_name
            all_rows.append(record)
        summary_lines.append(
            f"{case_name}: attempts={result['attempt_count']}, accepted={result['accepted_step_count']}, "
            f"cutbacks={sum(1 for row in result['attempt_history'] if row['cutback_triggered'])}, "
            f"terminated_early={result['terminated_early']}, termination_reason={result['termination_reason']}"
        )
        print("")

    write_cutback_history(all_rows, "cutback_history.csv")
    write_cutback_summary(summary_lines, "cutback_summary.txt")


if __name__ == "__main__":
    main()

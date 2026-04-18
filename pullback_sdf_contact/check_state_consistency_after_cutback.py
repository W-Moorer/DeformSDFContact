import csv
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np

from check_indenter_block_contact import build_indenter_state
from coupled_solver.staggered import solve_staggered_contact_loadpath


def write_attempt_history(path, attempt_history):
    fieldnames = [
        "attempt_state_index",
        "accepted_state_index",
        "accepted",
        "step",
        "target_load",
        "attempt_load",
        "converged",
        "cutback_level",
        "cutback_triggered",
        "restore_u_diff_norm",
        "restore_phi_diff_norm",
        "output_index",
        "termination_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in attempt_history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    state, solver_cfg, mode_cfg = build_indenter_state("aggressive")
    load_schedule = [
        {"step": 1, "load_value": 0.0, "label": "consistency_1"},
        {"step": 2, "load_value": 0.04, "label": "consistency_2"},
    ]
    final_state, result = solve_staggered_contact_loadpath(
        state,
        solver_cfg,
        load_schedule,
        use_contact_tangent_uu=False,
        max_outer_iter=4,
        max_cutbacks=3,
        min_cutback_increment=mode_cfg["min_cutback_increment"],
        write_outputs=False,
        history_path=None,
        verbose=False,
        step_verbose=False,
    )

    attempt_history = result["attempt_history"]
    accepted_history = result["accepted_history"]
    accepted_snapshots = result["accepted_snapshots"]
    failed_attempts = [item for item in attempt_history if not item["accepted"]]

    max_restore_u = max((item["restore_u_diff_norm"] for item in failed_attempts), default=0.0)
    max_restore_phi = max((item["restore_phi_diff_norm"] for item in failed_attempts), default=0.0)

    if accepted_snapshots:
        last_snapshot = accepted_snapshots[-1]
        final_u_diff = float(np.linalg.norm(final_state["u"].vector.array_r - last_snapshot["u"]))
        final_phi_diff = float(np.linalg.norm(final_state["phi"].vector.array_r - last_snapshot["phi"]))
    else:
        final_u_diff = 0.0
        final_phi_diff = 0.0

    accepted_clean = all(item["accepted"] for item in accepted_history)
    failed_not_promoted = all(not item["accepted"] for item in failed_attempts)
    output_index_only_on_accept = True
    expected_output_index = 0
    for item in attempt_history:
        if item["accepted"]:
            if item["output_index"] != expected_output_index:
                output_index_only_on_accept = False
                break
            expected_output_index += 1
        else:
            if item["output_index"] != expected_output_index:
                output_index_only_on_accept = False
                break

    accepted_after_cutback = []
    for item in accepted_history:
        prior_failed = [
            row
            for row in attempt_history
            if row["step"] == item["step"]
            and row["attempt_state_index"] < item["attempt_state_index"]
            and not row["accepted"]
        ]
        if prior_failed:
            accepted_after_cutback.append((item, prior_failed))

    print("State consistency after cutback")
    print("")
    print(f"attempt_count = {result['attempt_count']}")
    print(f"accepted_step_count = {result['accepted_step_count']}")
    print(f"terminated_early = {result['terminated_early']}")
    print(f"termination_reason = {result['termination_reason']}")
    print(f"final_accepted_load = {result['final_accepted_load']}")
    print(f"max restore ||du|| = {max_restore_u}")
    print(f"max restore ||dphi|| = {max_restore_phi}")
    print(f"final state vs last accepted snapshot ||du|| = {final_u_diff}")
    print(f"final state vs last accepted snapshot ||dphi|| = {final_phi_diff}")
    print(f"accepted history clean = {accepted_clean}")
    print(f"failed attempts not promoted = {failed_not_promoted}")
    print(f"output_index only increments on accepted = {output_index_only_on_accept}")
    print(f"accepted_after_cutback_count = {len(accepted_after_cutback)}")

    report_lines = [
        "State Consistency Report",
        "",
        f"attempt_count={result['attempt_count']}",
        f"accepted_step_count={result['accepted_step_count']}",
        f"terminated_early={result['terminated_early']}",
        f"termination_reason={result['termination_reason']}",
        f"final_accepted_load={result['final_accepted_load']}",
        f"max_restore_u_diff_norm={max_restore_u}",
        f"max_restore_phi_diff_norm={max_restore_phi}",
        f"final_u_diff_to_last_accepted_snapshot={final_u_diff}",
        f"final_phi_diff_to_last_accepted_snapshot={final_phi_diff}",
        f"accepted_history_clean={accepted_clean}",
        f"failed_attempts_not_promoted={failed_not_promoted}",
        f"output_index_only_on_accept={output_index_only_on_accept}",
        f"accepted_after_cutback_count={len(accepted_after_cutback)}",
    ]
    if accepted_after_cutback:
        for accepted_item, failed_rows in accepted_after_cutback:
            report_lines.append(
                "accepted_after_cutback: step={step}, accepted_attempt_state_index={attempt_state_index}, "
                "prior_failed_attempts={count}".format(
                    step=accepted_item["step"],
                    attempt_state_index=accepted_item["attempt_state_index"],
                    count=len(failed_rows),
                )
            )

    write_attempt_history("state_consistency_history.csv", attempt_history)
    write_report("state_consistency_report.txt", report_lines)


if __name__ == "__main__":
    main()

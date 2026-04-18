import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np

from check_indenter_block_contact import build_indenter_state, build_load_schedule
from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from coupled_solver.staggered import (
    _restore_state_fields,
    solve_sdf_subproblem,
    solve_staggered_contact_loadpath,
)
from solid.solve import solve_linear_solid_with_contact


MODES = ("rhs_only", "signed_tangent_only", "consistent_linearized")


def accepted_snapshot_at_load(result, load_value, tol=1e-12):
    for row, snapshot in zip(result["accepted_history"], result["accepted_snapshots"]):
        if abs(float(row["load_value"]) - float(load_value)) < tol:
            return row, snapshot
    raise ValueError(f"Could not find accepted snapshot at load {load_value}")


def clone_state_with_snapshot(snapshot):
    state, _, _ = build_indenter_state("aggressive")
    _restore_state_fields(state, snapshot)
    return state


def evaluate_contact_state(state, need_tangent=True):
    return assemble_contact_contributions_surface(
        state["quadrature_points"],
        state,
        state["penalty"],
        need_residual=True,
        need_tangent_uu=need_tangent,
        need_diagnostics=True,
    )


def run_fixed_state_mode(snapshot, contact_structure_mode):
    state = clone_state_with_snapshot(snapshot)
    before = evaluate_contact_state(state, need_tangent=True)
    u_prev = state["u"].vector.array_r.copy()
    solve_linear_solid_with_contact(
        state["u"],
        state["R_u_form"],
        state["solid_bcs"],
        contact_rhs=before["R_u_c"],
        contact_tangent_uu=before["K_uu_c"],
        contact_structure_mode=contact_structure_mode,
        reference_u=u_prev,
        verbose=False,
    )
    du_norm = float(np.linalg.norm(state["u"].vector.array_r - u_prev))
    after = evaluate_contact_state(state, need_tangent=True)

    u_mid = state["u"].vector.array_r.copy()
    solve_linear_solid_with_contact(
        state["u"],
        state["R_u_form"],
        state["solid_bcs"],
        contact_rhs=after["R_u_c"],
        contact_tangent_uu=after["K_uu_c"],
        contact_structure_mode=contact_structure_mode,
        reference_u=u_mid,
        verbose=False,
    )
    followup_du_norm = float(np.linalg.norm(state["u"].vector.array_r - u_mid))
    return {
        "mode": contact_structure_mode,
        "du_norm": du_norm,
        "contact_rhs_norm_before": before["diagnostics"]["reaction_norm"],
        "contact_rhs_norm_after": after["diagnostics"]["reaction_norm"],
        "reaction_norm_after": after["diagnostics"]["reaction_norm"],
        "negative_gap_sum_after": after["diagnostics"]["negative_gap_sum"],
        "followup_du_norm": followup_du_norm,
    }


def replay_attempt(snapshot, attempt_load, contact_structure_mode, max_outer_iter=4):
    state = clone_state_with_snapshot(snapshot)
    offset = np.asarray(state.get("slave_current_offset", np.zeros(3)), dtype=np.float64).copy()
    offset[2] = -float(attempt_load)
    state["slave_current_offset"] = offset
    state["current_load_value"] = float(attempt_load)

    rows = []
    previous_contact_rhs = None
    converged = False
    failure_reason = "Reached max_outer_iter without satisfying convergence tolerances"
    for outer_it in range(1, max_outer_iter + 1):
        u_prev = state["u"].vector.array_r.copy()
        phi_prev = state["phi"].vector.array_r.copy()

        sdf_iterations, sdf_converged = solve_sdf_subproblem(
            state["phi"],
            state["R_phi_form"],
            state["K_phi_phi_form"],
            state["phi_bcs"],
        )
        assembled = evaluate_contact_state(state, need_tangent=True)
        solve_linear_solid_with_contact(
            state["u"],
            state["R_u_form"],
            state["solid_bcs"],
            contact_rhs=assembled["R_u_c"],
            contact_tangent_uu=assembled["K_uu_c"],
            contact_structure_mode=contact_structure_mode,
            reference_u=u_prev,
            verbose=False,
        )

        du_norm = float(np.linalg.norm(state["u"].vector.array_r - u_prev))
        dphi_norm = float(np.linalg.norm(state["phi"].vector.array_r - phi_prev))
        contact_rhs_change = (
            np.inf
            if previous_contact_rhs is None
            else float(np.linalg.norm(assembled["R_u_c"] - previous_contact_rhs))
        )
        previous_contact_rhs = assembled["R_u_c"].copy()

        rows.append(
            {
                "outer_iteration": outer_it,
                "converged_sdf": bool(sdf_converged),
                "sdf_iterations": int(sdf_iterations),
                "du_norm": du_norm,
                "dphi_norm": dphi_norm,
                "contact_rhs_norm": assembled["diagnostics"]["reaction_norm"],
                "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
                "max_penetration": assembled["diagnostics"]["max_penetration"],
                "contact_rhs_change": contact_rhs_change,
            }
        )

        if not sdf_converged:
            failure_reason = "SDF subproblem did not converge"
            break
        if du_norm < 1e-8 and dphi_norm < 1e-8 and contact_rhs_change < 1e-10:
            converged = True
            failure_reason = ""
            break

    return {
        "mode": contact_structure_mode,
        "converged": converged,
        "failure_reason": failure_reason,
        "rows": rows,
    }


def main():
    state_rhs, solver_cfg_rhs, mode_cfg_rhs = build_indenter_state("aggressive")
    _, result_rhs = solve_staggered_contact_loadpath(
        state_rhs,
        solver_cfg_rhs,
        build_load_schedule("aggressive"),
        contact_structure_mode="rhs_only",
        max_outer_iter=mode_cfg_rhs["max_outer_iter"],
        max_cutbacks=mode_cfg_rhs["max_cutbacks"],
        min_cutback_increment=mode_cfg_rhs["min_cutback_increment"],
        write_outputs=False,
        history_path=None,
        verbose=False,
        step_verbose=False,
    )

    _, snapshot_load001 = accepted_snapshot_at_load(result_rhs, 0.01)
    fixed_rows = [run_fixed_state_mode(snapshot_load001, mode) for mode in MODES]

    attempts = result_rhs["attempt_history"]
    earliest_failed = next(
        row for row in attempts if row["step"] == 2 and abs(float(row["attempt_load"]) - 0.04) < 1e-12
    )
    common_accepted_load = 0.005
    _, common_snapshot = accepted_snapshot_at_load(result_rhs, common_accepted_load)
    divergence_attempt_load = 0.01
    replay_rows = [replay_attempt(common_snapshot, divergence_attempt_load, mode) for mode in MODES]

    print("Structure-step contact mode audit")
    print("")
    print("Fixed-state comparison on accepted snapshot (load = 0.01):")
    print(
        "  mode | ||du|| | contact_rhs_norm_before | contact_rhs_norm_after | "
        "reaction_norm_after | negative_gap_sum_after | followup ||du||"
    )
    for row in fixed_rows:
        print(
            f"  {row['mode']} | {row['du_norm']:.6e} | {row['contact_rhs_norm_before']:.6e} | "
            f"{row['contact_rhs_norm_after']:.6e} | {row['reaction_norm_after']:.6e} | "
            f"{row['negative_gap_sum_after']:.6e} | {row['followup_du_norm']:.6e}"
        )

    print("")
    print("Earliest divergence reference:")
    print(f"  baseline rhs_only common accepted load before divergence = {common_accepted_load:.6f}")
    print(f"  replay attempt load = {divergence_attempt_load:.6f}")
    print(
        "  original rhs_only path accepted at load 0.01 after cutback level 2; "
        "previous audit showed the old +Kuu path failed here."
    )
    print("")
    print("Replayed attempt from same input state:")
    for case in replay_rows:
        print(
            f"  mode={case['mode']} converged={case['converged']} "
            f"failure_reason={case['failure_reason']}"
        )
        print(
            "    iter | du_norm | dphi_norm | contact_rhs_norm | "
            "negative_gap_sum | max_penetration | contact_rhs_change"
        )
        for row in case["rows"]:
            print(
                f"    {row['outer_iteration']:>4d} | {row['du_norm']:.6e} | "
                f"{row['dphi_norm']:.6e} | {row['contact_rhs_norm']:.6e} | "
                f"{row['negative_gap_sum']:.6e} | {row['max_penetration']:.6e} | "
                f"{row['contact_rhs_change']:.6e}"
            )


if __name__ == "__main__":
    main()

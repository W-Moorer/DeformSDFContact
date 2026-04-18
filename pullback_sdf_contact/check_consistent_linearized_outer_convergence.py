import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
from petsc4py import PETSc

from check_indenter_block_contact import build_indenter_state, build_load_schedule
from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from coupled_solver.staggered import _restore_state_fields, solve_sdf_subproblem, solve_staggered_contact_loadpath
from solid.solve import solve_linear_solid_with_contact


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


def apply_relaxation(function, previous_values, relaxation):
    if relaxation >= 1.0:
        return
    trial = function.vector.array_r.copy()
    function.vector.array_w[:] = (1.0 - relaxation) * previous_values + relaxation * trial
    function.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )


def replay_attempt(snapshot, attempt_load, contact_structure_mode, max_outer_iter, relaxation_u=1.0, relaxation_phi=1.0):
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
        apply_relaxation(state["phi"], phi_prev, relaxation_phi)

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
        apply_relaxation(state["u"], u_prev, relaxation_u)

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
                "du_norm": du_norm,
                "dphi_norm": dphi_norm,
                "contact_rhs_change": contact_rhs_change,
                "reaction_norm": assembled["diagnostics"]["reaction_norm"],
                "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
                "max_penetration": assembled["diagnostics"]["max_penetration"],
                "sdf_iterations": int(sdf_iterations),
                "sdf_converged": bool(sdf_converged),
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
        "max_outer_iter": max_outer_iter,
        "relaxation_u": relaxation_u,
        "relaxation_phi": relaxation_phi,
        "converged": converged,
        "failure_reason": failure_reason,
        "rows": rows,
    }


def contraction_summary(rows):
    if len(rows) < 2:
        return {
            "last_du_norm": rows[-1]["du_norm"] if rows else 0.0,
            "last_contact_rhs_change": rows[-1]["contact_rhs_change"] if rows else 0.0,
            "mean_du_ratio": np.nan,
            "monotone_du": True,
            "monotone_rhs_change": True,
        }
    du_values = np.array([row["du_norm"] for row in rows], dtype=np.float64)
    rhs_values = np.array(
        [
            row["contact_rhs_change"]
            for row in rows[1:]
            if np.isfinite(row["contact_rhs_change"])
        ],
        dtype=np.float64,
    )
    du_ratios = du_values[1:] / np.maximum(du_values[:-1], 1e-30)
    rhs_ratios = (
        rhs_values[1:] / np.maximum(rhs_values[:-1], 1e-30)
        if rhs_values.size >= 2
        else np.array([], dtype=np.float64)
    )
    return {
        "last_du_norm": float(du_values[-1]),
        "last_contact_rhs_change": float(rows[-1]["contact_rhs_change"]),
        "mean_du_ratio": float(np.mean(du_ratios)),
        "monotone_du": bool(np.all(du_values[1:] <= du_values[:-1] + 1e-30)),
        "monotone_rhs_change": bool(
            rhs_values.size <= 1 or np.all(rhs_values[1:] <= rhs_values[:-1] + 1e-30)
        ),
        "mean_rhs_ratio": float(np.mean(rhs_ratios)) if rhs_ratios.size > 0 else np.nan,
    }


def main():
    state, solver_cfg, mode_cfg = build_indenter_state("aggressive")
    _, result = solve_staggered_contact_loadpath(
        state,
        solver_cfg,
        build_load_schedule("aggressive"),
        contact_structure_mode="rhs_only",
        max_outer_iter=mode_cfg["max_outer_iter"],
        max_cutbacks=mode_cfg["max_cutbacks"],
        min_cutback_increment=mode_cfg["min_cutback_increment"],
        write_outputs=False,
        history_path=None,
        verbose=False,
        step_verbose=False,
    )
    _, common_snapshot = accepted_snapshot_at_load(result, 0.005)
    attempt_load = 0.01

    reference_runs = [
        replay_attempt(common_snapshot, attempt_load, "rhs_only", 4, relaxation_u=1.0),
        replay_attempt(common_snapshot, attempt_load, "signed_tangent_only", 4, relaxation_u=1.0),
    ]
    consistent_runs = [
        replay_attempt(common_snapshot, attempt_load, "consistent_linearized", max_outer_iter, relaxation_u=1.0)
        for max_outer_iter in (4, 6, 8, 10, 12)
    ]
    relaxed_runs = [
        replay_attempt(common_snapshot, attempt_load, "consistent_linearized", 8, relaxation_u=relaxation_u)
        for relaxation_u in (0.7, 0.5)
    ]

    print("Consistent-linearized outer convergence diagnostic")
    print("")
    print("Reference difficult attempt:")
    print("  common accepted load = 0.005")
    print("  attempt load = 0.01")
    print("")
    print("Reference modes at max_outer_iter = 4:")
    print(
        "  mode | converged | last du_norm | last contact_rhs_change | "
        "mean du ratio | monotone du | monotone rhs_change"
    )
    for run in reference_runs:
        summary = contraction_summary(run["rows"])
        print(
            f"  {run['mode']} | {run['converged']} | {summary['last_du_norm']:.6e} | "
            f"{summary['last_contact_rhs_change']:.6e} | {summary['mean_du_ratio']:.6e} | "
            f"{summary['monotone_du']} | {summary['monotone_rhs_change']}"
        )
    print("")
    print("Consistent-linearized scan over max_outer_iter:")
    print(
        "  max_outer_iter | converged | last du_norm | last contact_rhs_change | "
        "mean du ratio | mean rhs ratio | monotone du | monotone rhs_change"
    )
    for run in consistent_runs:
        summary = contraction_summary(run["rows"])
        print(
            f"  {run['max_outer_iter']:>13d} | {run['converged']} | {summary['last_du_norm']:.6e} | "
            f"{summary['last_contact_rhs_change']:.6e} | {summary['mean_du_ratio']:.6e} | "
            f"{summary['mean_rhs_ratio']:.6e} | {summary['monotone_du']} | "
            f"{summary['monotone_rhs_change']}"
        )
    print("")
    print("Consistent-linearized relaxation scan at max_outer_iter = 8:")
    print(
        "  relaxation_u | converged | last du_norm | last contact_rhs_change | "
        "mean du ratio | monotone du | monotone rhs_change"
    )
    for run in relaxed_runs:
        summary = contraction_summary(run["rows"])
        print(
            f"  {run['relaxation_u']:.1f} | {run['converged']} | {summary['last_du_norm']:.6e} | "
            f"{summary['last_contact_rhs_change']:.6e} | {summary['mean_du_ratio']:.6e} | "
            f"{summary['monotone_du']} | {summary['monotone_rhs_change']}"
        )
    print("")
    print("Detailed consistent-linearized curves:")
    for run in consistent_runs:
        print(
            f"  max_outer_iter = {run['max_outer_iter']}, converged = {run['converged']}, "
            f"failure_reason = {run['failure_reason']}"
        )
        print(
            "    iter | du_norm | dphi_norm | contact_rhs_change | "
            "reaction_norm | negative_gap_sum"
        )
        for row in run["rows"]:
            print(
                f"    {row['outer_iteration']:>4d} | {row['du_norm']:.6e} | "
                f"{row['dphi_norm']:.6e} | {row['contact_rhs_change']:.6e} | "
                f"{row['reaction_norm']:.6e} | {row['negative_gap_sum']:.6e}"
            )


if __name__ == "__main__":
    main()

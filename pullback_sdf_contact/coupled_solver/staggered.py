# DOLFINx 0.3 exposes NonlinearProblem/NewtonSolver from dolfinx.fem and
# dolfinx.nls instead of the newer *.petsc modules.
import csv
import numpy as np

from dolfinx import fem, nls
from petsc4py import PETSc

from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from solid.solve import solve_linear_solid_with_contact


def solve_sdf_subproblem(phi, R_phi_form, K_phi_phi_form, bcs):
    """Solve the nonlinear pull-back SDF subproblem."""
    problem = fem.NonlinearProblem(R_phi_form, phi, bcs=bcs, J=K_phi_phi_form)
    solver = nls.NewtonSolver(phi.function_space.mesh.mpi_comm(), problem)
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 30
    n, converged = solver.solve(phi)
    return n, converged


def solve_staggered(state, cfg):
    """Legacy SDF-only staggered entry point kept for compatibility."""
    n, converged = solve_sdf_subproblem(
        state["phi"],
        state["R_phi_form"],
        state["K_phi_phi_form"],
        state["phi_bcs"],
    )
    return {"sdf_iterations": n, "sdf_converged": converged}


def _snapshot_state_fields(state):
    return {
        "u": state["u"].vector.array_r.copy(),
        "phi": state["phi"].vector.array_r.copy(),
        "step_offset": np.asarray(state.get("slave_current_offset", np.zeros(3)), dtype=np.float64).copy(),
        "current_load_value": float(state.get("current_load_value", 0.0)),
    }


def _restore_state_fields(state, snapshot):
    state["u"].vector.array_w[:] = snapshot["u"]
    state["u"].vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    state["phi"].vector.array_w[:] = snapshot["phi"]
    state["phi"].vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    state["slave_current_offset"] = snapshot["step_offset"].copy()
    state["current_load_value"] = snapshot["current_load_value"]


def _apply_relaxation(function, previous_values, relaxation):
    if relaxation >= 1.0:
        return
    trial = function.vector.array_r.copy()
    function.vector.array_w[:] = (1.0 - relaxation) * previous_values + relaxation * trial
    function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def _apply_step_data(state, step_data):
    state["current_step_id"] = int(step_data.get("step", state.get("current_step_id", 0)))
    load_value = float(step_data.get("load_value", state.get("current_load_value", 0.0)))
    state["current_load_value"] = load_value

    if "apply_step_data" in state and callable(state["apply_step_data"]):
        state["apply_step_data"](state, step_data)
        return

    if "slave_current_offset" in step_data:
        state["slave_current_offset"] = np.asarray(step_data["slave_current_offset"], dtype=np.float64)
        return

    offset = np.asarray(state.get("slave_current_offset", np.zeros(3)), dtype=np.float64).copy()
    offset[2] = -load_value
    state["slave_current_offset"] = offset


def _make_step_info(step_data, use_contact_tangent_uu, diagnostics, solid_info, sdf_info, du_norm, dphi_norm):
    return {
        "step_id": int(step_data.get("step", 0)),
        "load_value": float(step_data.get("load_value", 0.0)),
        "outer_iterations": int(step_data.get("outer_iterations", 0)),
        "converged": bool(step_data.get("converged", False)),
        "active_contact_points": int(diagnostics["active_contact_points"]),
        "negative_gap_sum": float(diagnostics["negative_gap_sum"]),
        "contact_rhs_norm": float(diagnostics["reaction_norm"]),
        "du_norm": float(du_norm),
        "dphi_norm": float(dphi_norm),
        "max_penetration": float(diagnostics["max_penetration"]),
        "reaction_norm": float(diagnostics["reaction_norm"]),
        "used_contact_tangent_uu": bool(use_contact_tangent_uu),
        "solid_info": solid_info,
        "sdf_info": sdf_info,
    }


def solve_staggered_contact_step(
    state,
    cfg,
    step_data,
    *,
    use_contact_tangent_uu=False,
    max_outer_iter=20,
    tol_du=1e-8,
    tol_dphi=1e-8,
    tol_contact_rhs=1e-10,
    relaxation_u=1.0,
    relaxation_phi=1.0,
    verbose=True,
):
    """Solve one load step by staggering between SDF and structure/contact."""
    _apply_step_data(state, step_data)
    rank0 = state["domain"].mpi_comm().rank == 0
    step_id = int(step_data.get("step", 0))
    load_value = float(step_data.get("load_value", state.get("current_load_value", 0.0)))
    previous_contact_rhs = None
    failure_reason = ""
    iter_history = []

    for outer_it in range(1, max_outer_iter + 1):
        u_prev = state["u"].vector.array_r.copy()
        phi_prev = state["phi"].vector.array_r.copy()

        sdf_iterations, sdf_converged = solve_sdf_subproblem(
            state["phi"],
            state["R_phi_form"],
            state["K_phi_phi_form"],
            state["phi_bcs"],
        )
        _apply_relaxation(state["phi"], phi_prev, relaxation_phi)
        dphi_norm = float(np.linalg.norm(state["phi"].vector.array_r - phi_prev))

        assembled = assemble_contact_contributions_surface(
            state["quadrature_points"],
            state,
            state["penalty"],
            need_residual=True,
            need_tangent_uu=use_contact_tangent_uu,
            need_diagnostics=True,
        )
        contact_rhs = assembled["R_u_c"]
        contact_tangent_uu = assembled["K_uu_c"] if use_contact_tangent_uu else None
        diagnostics = assembled["diagnostics"]
        point_data = assembled["point_data"]

        solid_info = solve_linear_solid_with_contact(
            state["u"],
            state["R_u_form"],
            state["solid_bcs"],
            contact_rhs=contact_rhs,
            contact_tangent_uu=contact_tangent_uu,
            verbose=verbose,
        )
        _apply_relaxation(state["u"], u_prev, relaxation_u)
        du_norm = float(np.linalg.norm(state["u"].vector.array_r - u_prev))

        contact_rhs_change = (
            np.inf if previous_contact_rhs is None else float(np.linalg.norm(contact_rhs - previous_contact_rhs))
        )
        previous_contact_rhs = contact_rhs.copy()

        sdf_info = {"iterations": sdf_iterations, "converged": bool(sdf_converged)}
        iter_info = _make_step_info(
            {"step": step_id, "load_value": load_value},
            use_contact_tangent_uu,
            diagnostics,
            solid_info,
            sdf_info,
            du_norm,
            dphi_norm,
        )
        iter_info["outer_iteration"] = outer_it
        iter_info["point_data"] = point_data
        iter_info["contact_rhs_change"] = contact_rhs_change
        iter_history.append(iter_info)

        if verbose and rank0:
            print(f"iter = {outer_it}")
            print(f"active contact points = {diagnostics['active_contact_points']}")
            print(f"||R_u^c||_2 = {diagnostics['reaction_norm']}")
            print(f"sum weighted negative gaps = {diagnostics['negative_gap_sum']}")
            print(f"||u^{{k+1}} - u^k||_2 = {du_norm}")

        if not sdf_converged:
            failure_reason = "SDF subproblem did not converge"
        elif not solid_info["converged"]:
            failure_reason = "Solid linear solve did not converge"
        elif du_norm < tol_du and dphi_norm < tol_dphi and contact_rhs_change < tol_contact_rhs:
            failure_reason = ""
            break
    else:
        outer_it = max_outer_iter
        if not failure_reason:
            failure_reason = "Reached max_outer_iter without satisfying convergence tolerances"

    converged = failure_reason == ""
    last = iter_history[-1]
    step_info = {
        "step_id": step_id,
        "load_value": load_value,
        "outer_iterations": outer_it,
        "converged": converged,
        "active_contact_points": last["active_contact_points"],
        "negative_gap_sum": last["negative_gap_sum"],
        "contact_rhs_norm": last["contact_rhs_norm"],
        "du_norm": last["du_norm"],
        "dphi_norm": last["dphi_norm"],
        "max_penetration": last["max_penetration"],
        "reaction_norm": last["reaction_norm"],
        "used_contact_tangent_uu": bool(use_contact_tangent_uu),
        "failure_reason": failure_reason,
        "point_data": last["point_data"],
        "iter_history": iter_history,
    }
    return state, step_info


def _write_history_csv(history, history_path):
    if history_path is None:
        return
    fieldnames = [
        "step",
        "load_value",
        "converged",
        "outer_iterations",
        "active_contact_points",
        "negative_gap_sum",
        "max_penetration",
        "reaction_norm",
        "du_norm",
        "dphi_norm",
        "cutback_level",
        "used_contact_tangent_uu",
        "output_index",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            row = {key: item.get(key, "") for key in fieldnames}
            writer.writerow(row)


def solve_staggered_contact_loadpath(
    state0,
    cfg,
    load_schedule,
    *,
    use_contact_tangent_uu=False,
    max_outer_iter=20,
    max_cutbacks=6,
    tol_du=1e-8,
    tol_dphi=1e-8,
    tol_contact_rhs=1e-10,
    relaxation_u=1.0,
    relaxation_phi=1.0,
    write_outputs=True,
    history_path=None,
    verbose=True,
    step_verbose=None,
):
    """Advance a multi-step load path with warm starts and simple cutbacks."""
    rank0 = state0["domain"].mpi_comm().rank == 0
    steps = []
    for idx, entry in enumerate(load_schedule, start=1):
        if isinstance(entry, dict):
            step_data = dict(entry)
            step_data.setdefault("step", idx)
        else:
            step_data = {"step": idx, "load_value": float(entry)}
        step_data.setdefault("cutback_level", 0)
        step_data.setdefault("label", f"step_{step_data['step']}")
        steps.append(step_data)

    history = []
    output_index = 0
    accepted_load = float(state0.get("current_load_value", 0.0))
    index = 0

    while index < len(steps):
        step_data = dict(steps[index])
        snapshot = _snapshot_state_fields(state0)
        if step_verbose is None:
            use_step_verbose = verbose
        else:
            use_step_verbose = step_verbose
        state0, step_info = solve_staggered_contact_step(
            state0,
            cfg,
            step_data,
            use_contact_tangent_uu=use_contact_tangent_uu,
            max_outer_iter=max_outer_iter,
            tol_du=tol_du,
            tol_dphi=tol_dphi,
            tol_contact_rhs=tol_contact_rhs,
            relaxation_u=relaxation_u,
            relaxation_phi=relaxation_phi,
            verbose=use_step_verbose,
        )

        history_item = {
            "step": step_data["step"],
            "load_value": float(step_data["load_value"]),
            "outer_iterations": int(step_info["outer_iterations"]),
            "converged": bool(step_info["converged"]),
            "active_contact_points": int(step_info["active_contact_points"]),
            "negative_gap_sum": float(step_info["negative_gap_sum"]),
            "contact_rhs_norm": float(step_info["contact_rhs_norm"]),
            "du_norm": float(step_info["du_norm"]),
            "dphi_norm": float(step_info["dphi_norm"]),
            "cutback_level": int(step_data.get("cutback_level", 0)),
            "reaction_norm": float(step_info["reaction_norm"]),
            "max_penetration": float(step_info["max_penetration"]),
            "output_index": output_index,
            "used_contact_tangent_uu": bool(use_contact_tangent_uu),
            "failure_reason": step_info["failure_reason"],
        }

        if step_info["converged"]:
            history.append(history_item)
            accepted_load = float(step_data["load_value"])
            output_index += 1
            index += 1
            continue

        _restore_state_fields(state0, snapshot)
        current_cutback = int(step_data.get("cutback_level", 0))
        if current_cutback >= max_cutbacks:
            history.append(history_item)
            if verbose and rank0:
                print(
                    f"step {step_data['step']} failed at load {step_data['load_value']} "
                    f"after {current_cutback} cutbacks: {step_info['failure_reason']}"
                )
            break

        midpoint = accepted_load + 0.5 * (float(step_data["load_value"]) - accepted_load)
        cutback_step = dict(step_data)
        cutback_step["load_value"] = midpoint
        cutback_step["cutback_level"] = current_cutback + 1
        base_label = step_data.get("label", f"step_{step_data['step']}")
        cutback_step["label"] = f"{base_label}_cutback{cutback_step['cutback_level']}"
        steps.insert(index, cutback_step)
        if verbose and rank0:
            print(
                f"cutback: step {step_data['step']} target {step_data['load_value']} "
                f"-> retry with {midpoint} (level {cutback_step['cutback_level']})"
            )

    if write_outputs and rank0:
        _write_history_csv(history, history_path)

    return state0, history


def solve_staggered_contact(state, cfg):
    """Compatibility wrapper for the original single-step assembled contact solve."""
    state, info = solve_staggered_contact_step(
        state,
        cfg,
        {"step": 1, "load_value": float(state.get("current_load_value", 0.0))},
        use_contact_tangent_uu=False,
        max_outer_iter=5 if cfg is None else cfg.max_it,
        verbose=True if cfg is None else cfg.verbose,
    )
    return {
        "iterations": info["outer_iterations"],
        "history": info["iter_history"],
        "final_active_contact_points": info["active_contact_points"],
        "final_contact_norm": info["contact_rhs_norm"],
    }

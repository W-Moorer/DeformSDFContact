import csv
import numpy as np

from dolfinx import fem
from petsc4py import PETSc

from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from solid.solve import _owned_bc_dofs, assemble_linear_solid_system, dense_array_from_petsc_mat


def _compile_form(expr):
    if hasattr(fem, "form") and callable(fem.form):
        return fem.form(expr)
    return fem.Form(expr)


def _assemble_vector_array(form_expr):
    compiled = _compile_form(form_expr)
    vec = fem.assemble_vector(compiled)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vec.array.copy()


def _assemble_matrix_dense(form_expr):
    compiled = _compile_form(form_expr)
    mat = fem.assemble_matrix(compiled)
    mat.assemble()
    return dense_array_from_petsc_mat(mat)


def _snapshot_state_fields(state):
    return {
        "u": state["u"].vector.array_r.copy(),
        "phi": state["phi"].vector.array_r.copy(),
        "step_offset": np.asarray(
            state.get("slave_current_offset", np.zeros(3)), dtype=np.float64
        ).copy(),
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


def _apply_step_data(state, step_data):
    state["current_step_id"] = int(step_data.get("step", state.get("current_step_id", 0)))
    load_value = float(
        step_data.get("attempt_load", step_data.get("load_value", state.get("current_load_value", 0.0)))
    )
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


def _write_history_csv(history, history_path):
    if history_path is None:
        return
    fieldnames = [
        "attempt_state_index",
        "accepted_state_index",
        "accepted",
        "step",
        "load_value",
        "target_load",
        "attempt_load",
        "accepted_load_before_step",
        "load_increment",
        "converged",
        "newton_iterations",
        "residual_norm",
        "increment_norm",
        "active_contact_points",
        "negative_gap_sum",
        "max_penetration",
        "reaction_norm",
        "cutback_level",
        "cutback_triggered",
        "cutback_reason",
        "terminated_early",
        "termination_reason",
        "requested_final_target_load",
        "final_accepted_load",
        "reached_final_target",
        "output_index",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            row = {key: item.get(key, "") for key in fieldnames}
            writer.writerow(row)


def _summarize_loadpath_result(
    attempt_history,
    accepted_history,
    requested_final_target_load,
    terminated_early,
    termination_reason,
):
    final_accepted_load = (
        float(accepted_history[-1]["load_value"]) if accepted_history else 0.0
    )
    reached_final_target = (
        not terminated_early
        and abs(final_accepted_load - requested_final_target_load) < 1e-12
    )
    return {
        "attempt_history": attempt_history,
        "accepted_history": accepted_history,
        "requested_final_target_load": float(requested_final_target_load),
        "final_accepted_load": final_accepted_load,
        "reached_final_target": bool(reached_final_target),
        "terminated_early": bool(terminated_early),
        "termination_reason": termination_reason,
        "attempt_count": len(attempt_history),
        "accepted_step_count": len(accepted_history),
    }


def _current_state_vector(state):
    return np.concatenate([state["u"].vector.array_r.copy(), state["phi"].vector.array_r.copy()])


def _apply_state_increment(state, delta_u, delta_phi, scale=1.0):
    state["u"].vector.array_w[:] = state["u"].vector.array_r + scale * delta_u
    state["u"].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    state["phi"].vector.array_w[:] = state["phi"].vector.array_r + scale * delta_phi
    state["phi"].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def _apply_block_dirichlet(J, residual, current_values, constrained_dofs):
    if constrained_dofs.size == 0:
        return
    J[constrained_dofs, :] = 0.0
    J[:, constrained_dofs] = 0.0
    J[constrained_dofs, constrained_dofs] = 1.0
    residual[constrained_dofs] = current_values[constrained_dofs]


def assemble_monolithic_contact_system(state, cfg, *, need_jacobian=True):
    """Assemble the dense monolithic block residual and Jacobian."""
    A_struct, b_struct, solid_meta = assemble_linear_solid_system(
        state["u"],
        state["R_u_form"],
        state["solid_bcs"],
        contact_rhs=None,
        contact_tangent_uu=None,
        contact_structure_mode="rhs_only",
        reference_u=state["u"].vector.array_r.copy(),
    )
    J_uu_struct = dense_array_from_petsc_mat(A_struct)
    u_vec = state["u"].vector.array_r.copy()
    R_u_struct = J_uu_struct.dot(u_vec) - b_struct.array.copy()

    contact = assemble_contact_contributions_surface(
        state["quadrature_points"],
        state,
        state["penalty"],
        need_residual=True,
        need_tangent_uu=need_jacobian,
        need_tangent_uphi=need_jacobian,
        need_diagnostics=True,
    )
    R_u_total = R_u_struct - contact["R_u_c"]

    R_phi_total = _assemble_vector_array(state["R_phi_form"])

    out = {
        "R_u_total": R_u_total,
        "R_phi_total": R_phi_total,
        "R_u_struct": R_u_struct,
        "R_u_contact": contact["R_u_c"],
        "J_uu_struct": J_uu_struct,
        "solid_meta": solid_meta,
        "diagnostics": contact["diagnostics"],
        "point_data": contact["point_data"],
        "u_constrained_dofs": _owned_bc_dofs(state["solid_bcs"]),
        "phi_constrained_dofs": _owned_bc_dofs(state["phi_bcs"]),
    }

    if need_jacobian:
        out["J_uu"] = J_uu_struct - contact["K_uu_c"]
        out["J_uphi"] = -contact["K_uphi_c"]
        out["J_phiu"] = _assemble_matrix_dense(state["K_phi_u_form"])
        out["J_phiphi"] = _assemble_matrix_dense(state["K_phi_phi_form"])
    else:
        out["J_uu"] = None
        out["J_uphi"] = None
        out["J_phiu"] = None
        out["J_phiphi"] = None

    return out


def _build_monolithic_dense_system(state, assembled):
    ndof_u = state["u"].vector.getLocalSize()
    ndof_phi = state["phi"].vector.getLocalSize()
    residual = np.concatenate([assembled["R_u_total"], assembled["R_phi_total"]])
    J = np.zeros((ndof_u + ndof_phi, ndof_u + ndof_phi), dtype=np.float64)
    J[:ndof_u, :ndof_u] = assembled["J_uu"]
    J[:ndof_u, ndof_u:] = assembled["J_uphi"]
    J[ndof_u:, :ndof_u] = assembled["J_phiu"]
    J[ndof_u:, ndof_u:] = assembled["J_phiphi"]

    current_values = _current_state_vector(state)
    _apply_block_dirichlet(J, residual, current_values, assembled["u_constrained_dofs"])
    _apply_block_dirichlet(
        J, residual, current_values, ndof_u + assembled["phi_constrained_dofs"]
    )
    return J, residual


def solve_monolithic_contact(
    state,
    cfg,
    *,
    max_newton_iter=15,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=False,
    verbose=True,
):
    """Solve one fixed-load monolithic contact state by dense Newton."""
    rank0 = state["domain"].mpi_comm().rank == 0
    history = []
    failure_reason = ""

    for newton_it in range(1, max_newton_iter + 1):
        assembled = assemble_monolithic_contact_system(state, cfg, need_jacobian=True)
        J, residual = _build_monolithic_dense_system(state, assembled)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < tol_res:
            info = {
                "newton_iterations": newton_it - 1,
                "converged": True,
                "failure_reason": "",
                "residual_norm": residual_norm,
                "increment_norm": 0.0,
                "active_contact_points": assembled["diagnostics"]["active_contact_points"],
                "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
                "reaction_norm": assembled["diagnostics"]["reaction_norm"],
                "max_penetration": assembled["diagnostics"]["max_penetration"],
                "history": history,
            }
            return state, info

        try:
            delta = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError as exc:
            failure_reason = f"dense block solve failed: {exc}"
            break

        ndof_u = state["u"].vector.getLocalSize()
        delta_u = delta[:ndof_u]
        delta_phi = delta[ndof_u:]
        increment_norm = float(np.linalg.norm(delta))

        if line_search:
            alpha = 1.0
            snapshot = _snapshot_state_fields(state)
            accepted = False
            while alpha >= 1.0 / 64.0:
                _restore_state_fields(state, snapshot)
                _apply_state_increment(state, delta_u, delta_phi, scale=alpha)
                trial = assemble_monolithic_contact_system(state, cfg, need_jacobian=False)
                trial_residual = np.concatenate([trial["R_u_total"], trial["R_phi_total"]])
                trial_residual_norm = float(np.linalg.norm(trial_residual))
                if trial_residual_norm < residual_norm:
                    accepted = True
                    residual_after = trial_residual_norm
                    break
                alpha *= 0.5
            if not accepted:
                _restore_state_fields(state, snapshot)
                failure_reason = "line search failed to reduce residual"
                break
        else:
            _apply_state_increment(state, delta_u, delta_phi, scale=1.0)
            residual_after = None

        row = {
            "newton_iteration": newton_it,
            "residual_norm": residual_norm,
            "increment_norm": increment_norm,
            "active_contact_points": assembled["diagnostics"]["active_contact_points"],
            "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
            "reaction_norm": assembled["diagnostics"]["reaction_norm"],
            "max_penetration": assembled["diagnostics"]["max_penetration"],
            "residual_after": residual_after,
        }
        history.append(row)

        if verbose and rank0:
            print(
                f"Newton iter {newton_it}: "
                f"||R||={residual_norm:.6e}, ||d||={increment_norm:.6e}, "
                f"active={row['active_contact_points']}, gap_sum={row['negative_gap_sum']:.6e}"
            )

        if increment_norm < tol_inc:
            info = {
                "newton_iterations": newton_it,
                "converged": True,
                "failure_reason": "",
                "residual_norm": residual_norm,
                "increment_norm": increment_norm,
                "active_contact_points": row["active_contact_points"],
                "negative_gap_sum": row["negative_gap_sum"],
                "reaction_norm": row["reaction_norm"],
                "max_penetration": row["max_penetration"],
                "history": history,
            }
            return state, info

    if not failure_reason:
        failure_reason = "Reached max_newton_iter without satisfying convergence tolerances"

    last = history[-1] if history else {
        "increment_norm": np.inf,
        "active_contact_points": 0,
        "negative_gap_sum": 0.0,
        "reaction_norm": 0.0,
        "max_penetration": 0.0,
        "residual_norm": np.inf,
    }
    info = {
        "newton_iterations": len(history),
        "converged": False,
        "failure_reason": failure_reason,
        "residual_norm": last["residual_norm"],
        "increment_norm": last["increment_norm"],
        "active_contact_points": last["active_contact_points"],
        "negative_gap_sum": last["negative_gap_sum"],
        "reaction_norm": last["reaction_norm"],
        "max_penetration": last["max_penetration"],
        "history": history,
    }
    return state, info


def solve_monolithic_contact_loadpath(
    state0,
    cfg,
    load_schedule,
    *,
    max_newton_iter=15,
    max_cutbacks=6,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=False,
    write_outputs=True,
    history_path=None,
    verbose=True,
    min_cutback_increment=1e-8,
):
    """Advance a load path with the dense monolithic Newton step."""
    rank0 = state0["domain"].mpi_comm().rank == 0
    steps = []
    for idx, entry in enumerate(load_schedule, start=1):
        if isinstance(entry, dict):
            step_data = dict(entry)
            step_data.setdefault("step", idx)
        else:
            step_data = {"step": idx, "load_value": float(entry)}
        target_load = float(step_data["load_value"])
        step_data.setdefault("target_load", target_load)
        step_data.setdefault("attempt_load", target_load)
        step_data.setdefault("cutback_level", 0)
        step_data.setdefault("label", f"step_{step_data['step']}")
        steps.append(step_data)

    attempt_history = []
    accepted_history = []
    accepted_load = float(state0.get("current_load_value", 0.0))
    requested_final_target_load = float(steps[-1]["target_load"]) if steps else accepted_load
    output_index = 0
    attempt_state_index = 0
    accepted_state_index = 0
    index = 0
    terminated_early = False
    termination_reason = "completed_schedule"

    while index < len(steps):
        step_data = dict(steps[index])
        attempt_load = float(step_data.get("attempt_load", step_data["load_value"]))
        target_load = float(step_data.get("target_load", attempt_load))
        accepted_load_before_step = float(accepted_load)
        load_increment = attempt_load - accepted_load_before_step
        attempt_state_index += 1
        snapshot = _snapshot_state_fields(state0)
        _apply_step_data(state0, step_data)

        if verbose and rank0:
            print(
                f"monolithic step {step_data['step']} attempt load = {attempt_load} "
                f"(target = {target_load}, increment = {load_increment}, "
                f"cutback level = {step_data.get('cutback_level', 0)})"
            )

        state0, step_info = solve_monolithic_contact(
            state0,
            cfg,
            max_newton_iter=max_newton_iter,
            tol_res=tol_res,
            tol_inc=tol_inc,
            line_search=line_search,
            verbose=verbose,
        )

        history_item = {
            "attempt_state_index": attempt_state_index,
            "accepted_state_index": accepted_state_index,
            "accepted": False,
            "step": step_data["step"],
            "load_value": attempt_load,
            "target_load": target_load,
            "attempt_load": attempt_load,
            "accepted_load_before_step": accepted_load_before_step,
            "load_increment": load_increment,
            "converged": bool(step_info["converged"]),
            "newton_iterations": int(step_info["newton_iterations"]),
            "residual_norm": float(step_info["residual_norm"]),
            "increment_norm": float(step_info["increment_norm"]),
            "active_contact_points": int(step_info["active_contact_points"]),
            "negative_gap_sum": float(step_info["negative_gap_sum"]),
            "reaction_norm": float(step_info["reaction_norm"]),
            "max_penetration": float(step_info["max_penetration"]),
            "cutback_level": int(step_data.get("cutback_level", 0)),
            "cutback_triggered": False,
            "cutback_reason": "",
            "terminated_early": False,
            "termination_reason": "",
            "requested_final_target_load": requested_final_target_load,
            "final_accepted_load": accepted_load_before_step,
            "reached_final_target": False,
            "output_index": output_index,
        }

        if step_info["converged"]:
            accepted_state_index += 1
            history_item["accepted"] = True
            history_item["accepted_state_index"] = accepted_state_index
            history_item["final_accepted_load"] = attempt_load
            attempt_history.append(history_item)
            accepted_history.append(dict(history_item))
            accepted_load = attempt_load
            output_index += 1
            index += 1
            continue

        _restore_state_fields(state0, snapshot)

        current_cutback = int(step_data.get("cutback_level", 0))
        load_gap = abs(target_load - accepted_load_before_step)
        same_increment = abs(attempt_load - accepted_load_before_step) < min_cutback_increment
        min_increment_reached = load_gap < min_cutback_increment or same_increment

        if current_cutback >= max_cutbacks or min_increment_reached:
            history_item["terminated_early"] = True
            history_item["termination_reason"] = step_info["failure_reason"]
            history_item["cutback_reason"] = (
                "min_increment_reached" if min_increment_reached else "max_cutbacks_reached"
            )
            history_item["final_accepted_load"] = accepted_load_before_step
            attempt_history.append(history_item)
            terminated_early = True
            termination_reason = step_info["failure_reason"]
            break

        midpoint = accepted_load_before_step + 0.5 * (attempt_load - accepted_load_before_step)
        cutback_step = dict(step_data)
        cutback_step["attempt_load"] = midpoint
        cutback_step["target_load"] = target_load
        cutback_step["cutback_level"] = current_cutback + 1
        history_item["cutback_triggered"] = True
        history_item["cutback_reason"] = step_info["failure_reason"]
        history_item["next_attempt_load"] = midpoint
        attempt_history.append(history_item)
        steps.insert(index, cutback_step)

    result = _summarize_loadpath_result(
        attempt_history,
        accepted_history,
        requested_final_target_load,
        terminated_early,
        termination_reason,
    )
    for item in attempt_history:
        item["terminated_early"] = result["terminated_early"]
        item["termination_reason"] = result["termination_reason"]
        item["final_accepted_load"] = result["final_accepted_load"]
        item["reached_final_target"] = result["reached_final_target"]

    if write_outputs and rank0:
        _write_history_csv(attempt_history, history_path)

    return state0, result

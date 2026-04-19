import csv
import numpy as np

from dolfinx import fem
from petsc4py import PETSc

from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from solid.solve import _owned_bc_dofs, assemble_linear_solid_system, dense_array_from_petsc_mat

RECOMMENDED_MONOLITHIC_BACKEND = "petsc_block"
RECOMMENDED_MONOLITHIC_MAX_NEWTON_ITER = 20
RECOMMENDED_MONOLITHIC_LINE_SEARCH = True
RECOMMENDED_MONOLITHIC_INITIAL_DAMPING = 1.0
RECOMMENDED_MONOLITHIC_MAX_BACKTRACKS = 8
RECOMMENDED_MONOLITHIC_BACKTRACK_FACTOR = 0.5
RECOMMENDED_MONOLITHIC_TOL_RES = 1e-8
RECOMMENDED_MONOLITHIC_TOL_INC = 1e-8
RECOMMENDED_MONOLITHIC_LINEAR_SOLVER_MODE = "lu"
RECOMMENDED_MONOLITHIC_KSP_TYPE = "gmres"
RECOMMENDED_MONOLITHIC_PC_TYPE = "fieldsplit"
RECOMMENDED_MONOLITHIC_BLOCK_PC_NAME = "fieldsplit_multiplicative_ilu"
RECOMMENDED_MONOLITHIC_KSP_RTOL = 1e-10
RECOMMENDED_MONOLITHIC_KSP_ATOL = 1e-12
RECOMMENDED_MONOLITHIC_KSP_MAX_IT = 400


def recommended_monolithic_contact_options(**overrides):
    """Return the current project-recommended monolithic settings."""
    options = {
        "backend": RECOMMENDED_MONOLITHIC_BACKEND,
        "max_newton_iter": RECOMMENDED_MONOLITHIC_MAX_NEWTON_ITER,
        "line_search": RECOMMENDED_MONOLITHIC_LINE_SEARCH,
        "initial_damping": RECOMMENDED_MONOLITHIC_INITIAL_DAMPING,
        "max_backtracks": RECOMMENDED_MONOLITHIC_MAX_BACKTRACKS,
        "backtrack_factor": RECOMMENDED_MONOLITHIC_BACKTRACK_FACTOR,
        "tol_res": RECOMMENDED_MONOLITHIC_TOL_RES,
        "tol_inc": RECOMMENDED_MONOLITHIC_TOL_INC,
        "linear_solver_mode": RECOMMENDED_MONOLITHIC_LINEAR_SOLVER_MODE,
        "ksp_type": RECOMMENDED_MONOLITHIC_KSP_TYPE,
        "pc_type": RECOMMENDED_MONOLITHIC_PC_TYPE,
        "block_pc_name": RECOMMENDED_MONOLITHIC_BLOCK_PC_NAME,
        "ksp_rtol": RECOMMENDED_MONOLITHIC_KSP_RTOL,
        "ksp_atol": RECOMMENDED_MONOLITHIC_KSP_ATOL,
        "ksp_max_it": RECOMMENDED_MONOLITHIC_KSP_MAX_IT,
    }
    options.update(overrides)
    return options


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
        "backend",
        "linear_solver_mode",
        "ksp_type",
        "pc_type",
        "block_pc_name",
        "linear_converged",
        "linear_iterations",
        "ksp_reason",
        "newton_iterations",
        "residual_norm",
        "increment_norm",
        "active_contact_points",
        "negative_gap_sum",
        "max_penetration",
        "reaction_norm",
        "cutback_level",
        "step_length",
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


def _create_petsc_vec_from_array(array, comm):
    vec = PETSc.Vec().createSeq(len(array), comm=comm)
    vec.setValues(np.arange(len(array), dtype=np.int32), np.asarray(array, dtype=np.float64))
    vec.assemble()
    return vec


def _create_petsc_aij_from_dense(matrix, comm):
    matrix = np.asarray(matrix, dtype=np.float64)
    nrows, ncols = matrix.shape
    mat = PETSc.Mat().createAIJ([nrows, ncols], comm=comm)
    mat.setUp()
    rows = np.arange(nrows, dtype=np.int32)
    mat.setValues(rows, rows, matrix)
    mat.assemble()
    return mat


def _configure_ksp_for_fieldsplit(solver, A, ndof_u, ndof_phi, *, block_pc_name):
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    if block_pc_name.startswith("fieldsplit_additive_"):
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    elif block_pc_name.startswith("fieldsplit_multiplicative_"):
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
    elif block_pc_name.startswith("fieldsplit_symmetric_multiplicative_"):
        pc.setFieldSplitType(PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE)
    else:
        raise ValueError(f"Unsupported block_pc_name: {block_pc_name}")
    comm = A.getComm()
    u_is = PETSc.IS().createGeneral(np.arange(ndof_u, dtype=np.int32), comm=comm)
    phi_is = PETSc.IS().createGeneral(
        np.arange(ndof_u, ndof_u + ndof_phi, dtype=np.int32), comm=comm
    )
    pc.setFieldSplitIS(("u", u_is), ("phi", phi_is))
    solver.setUp()
    subksps = pc.getFieldSplitSubKSP()
    if len(subksps) != 2:
        raise RuntimeError(f"Expected 2 fieldsplit sub-KSPs, got {len(subksps)}")

    if block_pc_name in {
        "fieldsplit_additive_ilu",
        "fieldsplit_multiplicative_ilu",
        "fieldsplit_symmetric_multiplicative_ilu",
    }:
        for sub in subksps:
            sub.setType("preonly")
            sub.getPC().setType("ilu")
    elif block_pc_name in {
        "fieldsplit_additive_lu",
        "fieldsplit_multiplicative_lu",
        "fieldsplit_symmetric_multiplicative_lu",
    }:
        for sub in subksps:
            sub.setType("preonly")
            sub.getPC().setType("lu")
    else:
        raise ValueError(f"Unsupported block_pc_name: {block_pc_name}")

    return subksps


def _solve_petsc_linear_system(
    A,
    rhs,
    *,
    ndof_u,
    ndof_phi,
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    ksp_rtol=1e-10,
    ksp_atol=1e-12,
    ksp_max_it=400,
):
    x = rhs.duplicate()
    x.set(0.0)
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    if linear_solver_mode == "lu":
        solver.setType("preonly")
        solver.getPC().setType("lu")
        resolved_ksp_type = "preonly"
        resolved_pc_type = "lu"
        resolved_block_pc_name = "global_lu"
    elif linear_solver_mode == "krylov":
        solver.setType(ksp_type)
        solver.setTolerances(rtol=float(ksp_rtol), atol=float(ksp_atol), max_it=int(ksp_max_it))
        pc = solver.getPC()
        resolved_ksp_type = ksp_type
        resolved_pc_type = pc_type
        if pc_type == "fieldsplit":
            resolved_block_pc_name = block_pc_name
            _configure_ksp_for_fieldsplit(
                solver,
                A,
                ndof_u,
                ndof_phi,
                block_pc_name=block_pc_name,
            )
        else:
            pc.setType(pc_type)
            resolved_block_pc_name = f"global_{pc_type}"
    else:
        raise ValueError(f"Unsupported linear_solver_mode: {linear_solver_mode}")
    solver.solve(rhs, x)
    info = {
        "linear_solver_mode": linear_solver_mode,
        "ksp_type": resolved_ksp_type,
        "pc_type": resolved_pc_type,
        "block_pc_name": resolved_block_pc_name,
        "linear_iterations": int(solver.getIterationNumber()),
        "ksp_reason": int(solver.getConvergedReason()),
        "linear_converged": int(solver.getConvergedReason()) > 0,
    }
    return x.getArray().copy(), solver, info


def _dense_block_system_from_blocks(state, assembled):
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


def _assemble_global_backend_objects(state, assembled, backend):
    global_jacobian_dense, global_residual_array = _dense_block_system_from_blocks(state, assembled)
    out = {
        "global_jacobian_dense": global_jacobian_dense,
        "global_residual_array": global_residual_array,
        "global_jacobian_mat": None,
        "global_residual_vec": None,
    }
    if backend == "petsc_block":
        comm = state["domain"].mpi_comm()
        out["global_jacobian_mat"] = _create_petsc_aij_from_dense(global_jacobian_dense, comm)
        out["global_residual_vec"] = _create_petsc_vec_from_array(global_residual_array, comm)
    elif backend != "dense":
        raise ValueError(f"Unsupported monolithic backend: {backend}")
    return out


def assemble_monolithic_contact_system(state, cfg, *, backend="dense", need_jacobian=True):
    """Assemble the monolithic block residual and Jacobian."""
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
        "backend": backend,
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
        out.update(_assemble_global_backend_objects(state, out, backend))
    else:
        out["J_uu"] = None
        out["J_uphi"] = None
        out["J_phiu"] = None
        out["J_phiphi"] = None
        out["global_jacobian_dense"] = None
        out["global_jacobian_mat"] = None
        out["global_residual_array"] = np.concatenate([R_u_total, R_phi_total])
        out["global_residual_vec"] = (
            _create_petsc_vec_from_array(out["global_residual_array"], state["domain"].mpi_comm())
            if backend == "petsc_block"
            else None
        )

    return out


def _evaluate_total_residual_norm(state, cfg, *, backend="dense"):
    assembled = assemble_monolithic_contact_system(state, cfg, backend=backend, need_jacobian=False)
    residual = assembled["global_residual_array"]
    return float(np.linalg.norm(residual)), assembled


def _backtracking_line_search(
    state,
    cfg,
    delta_u,
    delta_phi,
    residual_norm_before,
    *,
    backend="dense",
    initial_damping=1.0,
    max_backtracks=8,
    backtrack_factor=0.5,
):
    snapshot = _snapshot_state_fields(state)
    alpha = float(initial_damping)
    best = None

    for _ in range(max_backtracks + 1):
        _restore_state_fields(state, snapshot)
        _apply_state_increment(state, delta_u, delta_phi, scale=alpha)
        residual_after, assembled_after = _evaluate_total_residual_norm(state, cfg, backend=backend)
        candidate = (alpha, residual_after, assembled_after)
        if best is None or residual_after < best[1]:
            best = candidate
        if residual_after < residual_norm_before:
            return {
                "accepted": True,
                "step_length": alpha,
                "residual_after": residual_after,
                "assembled_after": assembled_after,
                "used_fallback": False,
            }
        alpha *= float(backtrack_factor)

    _restore_state_fields(state, snapshot)
    if best is None:
        return {
            "accepted": False,
            "step_length": 0.0,
            "residual_after": residual_norm_before,
            "assembled_after": None,
            "used_fallback": False,
        }

    alpha_best, residual_best, assembled_best = best
    _apply_state_increment(state, delta_u, delta_phi, scale=alpha_best)
    return {
        "accepted": True,
        "step_length": alpha_best,
        "residual_after": residual_best,
        "assembled_after": assembled_best,
        "used_fallback": True,
    }


def solve_monolithic_contact(
    state,
    cfg,
    *,
    backend="dense",
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    ksp_rtol=1e-10,
    ksp_atol=1e-12,
    ksp_max_it=400,
    max_newton_iter=15,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=False,
    initial_damping=1.0,
    max_backtracks=8,
    backtrack_factor=0.5,
    verbose=True,
):
    """Solve one fixed-load monolithic contact state."""
    rank0 = state["domain"].mpi_comm().rank == 0
    history = []
    failure_reason = ""

    for newton_it in range(1, max_newton_iter + 1):
        assembled = assemble_monolithic_contact_system(state, cfg, backend=backend, need_jacobian=True)
        residual = assembled["global_residual_array"]
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < tol_res:
            total_linear_iterations = int(sum(item["linear_iterations"] for item in history))
            info = {
                "newton_iterations": newton_it - 1,
                "converged": True,
                "failure_reason": "",
                "backend": backend,
                "linear_solver_mode": linear_solver_mode,
                "ksp_type": "preonly" if linear_solver_mode == "lu" else ksp_type,
                "pc_type": "lu" if linear_solver_mode == "lu" else pc_type,
                "block_pc_name": "global_lu" if linear_solver_mode == "lu" else block_pc_name,
                "linear_converged": True,
                "linear_iterations": 0,
                "total_linear_iterations": total_linear_iterations,
                "ksp_reason": 0,
                "residual_norm": residual_norm,
                "increment_norm": 0.0,
                "active_contact_points": assembled["diagnostics"]["active_contact_points"],
                "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
                "reaction_norm": assembled["diagnostics"]["reaction_norm"],
                "max_penetration": assembled["diagnostics"]["max_penetration"],
                "step_length": 0.0,
                "history": history,
            }
            return state, info

        try:
            linear_info = {
                "linear_solver_mode": linear_solver_mode,
                "ksp_type": "",
                "pc_type": "",
                "block_pc_name": "",
                "linear_converged": True,
                "linear_iterations": 0,
                "ksp_reason": 0,
            }
            if backend == "dense":
                delta = np.linalg.solve(assembled["global_jacobian_dense"], -residual)
            elif backend == "petsc_block":
                rhs = _create_petsc_vec_from_array(-residual, state["domain"].mpi_comm())
                ndof_u = state["u"].vector.getLocalSize()
                ndof_phi = state["phi"].vector.getLocalSize()
                delta, _, linear_info = _solve_petsc_linear_system(
                    assembled["global_jacobian_mat"],
                    rhs,
                    ndof_u=ndof_u,
                    ndof_phi=ndof_phi,
                    linear_solver_mode=linear_solver_mode,
                    ksp_type=ksp_type,
                    pc_type=pc_type,
                    block_pc_name=block_pc_name,
                    ksp_rtol=ksp_rtol,
                    ksp_atol=ksp_atol,
                    ksp_max_it=ksp_max_it,
                )
                if not linear_info["linear_converged"]:
                    failure_reason = (
                        "monolithic linear solve failed: "
                        f"ksp_reason={linear_info['ksp_reason']}"
                    )
                    break
            else:
                raise ValueError(f"Unsupported monolithic backend: {backend}")
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            failure_reason = f"monolithic solve failed: {exc}"
            break

        ndof_u = state["u"].vector.getLocalSize()
        delta_u = delta[:ndof_u]
        delta_phi = delta[ndof_u:]
        increment_norm = float(np.linalg.norm(delta))

        used_fallback = False
        if line_search:
            line_search_out = _backtracking_line_search(
                state,
                cfg,
                delta_u,
                delta_phi,
                residual_norm,
                backend=backend,
                initial_damping=initial_damping,
                max_backtracks=max_backtracks,
                backtrack_factor=backtrack_factor,
            )
            if not line_search_out["accepted"]:
                failure_reason = "line search failed to produce a trial step"
                break
            step_length = float(line_search_out["step_length"])
            residual_after = float(line_search_out["residual_after"])
            assembled_after = line_search_out["assembled_after"]
            used_fallback = bool(line_search_out["used_fallback"])
        else:
            _apply_state_increment(state, delta_u, delta_phi, scale=1.0)
            step_length = 1.0
            residual_after, assembled_after = _evaluate_total_residual_norm(state, cfg, backend=backend)

        row = {
            "newton_iteration": newton_it,
            "residual_norm_before": residual_norm,
            "residual_norm_after": residual_after,
            "residual_norm": residual_after,
            "increment_norm": increment_norm,
            "active_contact_points": assembled["diagnostics"]["active_contact_points"],
            "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
            "reaction_norm": assembled["diagnostics"]["reaction_norm"],
            "max_penetration": assembled["diagnostics"]["max_penetration"],
            "active_contact_points_after": assembled_after["diagnostics"]["active_contact_points"],
            "negative_gap_sum_after": assembled_after["diagnostics"]["negative_gap_sum"],
            "reaction_norm_after": assembled_after["diagnostics"]["reaction_norm"],
            "max_penetration_after": assembled_after["diagnostics"]["max_penetration"],
            "step_length": step_length,
            "used_fallback": used_fallback,
            "backend": backend,
            "linear_solver_mode": linear_info["linear_solver_mode"],
            "ksp_type": linear_info["ksp_type"],
            "pc_type": linear_info["pc_type"],
            "block_pc_name": linear_info["block_pc_name"],
            "linear_converged": linear_info["linear_converged"],
            "linear_iterations": linear_info["linear_iterations"],
            "ksp_reason": linear_info["ksp_reason"],
        }
        history.append(row)

        if verbose and rank0:
            print(
                f"Newton iter {newton_it}: "
                f"||R||={residual_norm:.6e} -> {residual_after:.6e}, "
                f"||d||={increment_norm:.6e}, alpha={step_length:.3f}, "
                f"lin_it={row['linear_iterations']}, "
                f"active={row['active_contact_points_after']}, "
                f"gap_sum={row['negative_gap_sum_after']:.6e}"
            )

        if residual_after < tol_res or increment_norm * step_length < tol_inc:
            total_linear_iterations = int(sum(item["linear_iterations"] for item in history))
            info = {
                "newton_iterations": newton_it,
                "converged": True,
                "failure_reason": "",
                "backend": backend,
                "linear_solver_mode": row["linear_solver_mode"],
                "ksp_type": row["ksp_type"],
                "pc_type": row["pc_type"],
                "block_pc_name": row["block_pc_name"],
                "linear_converged": row["linear_converged"],
                "linear_iterations": row["linear_iterations"],
                "total_linear_iterations": total_linear_iterations,
                "ksp_reason": row["ksp_reason"],
                "residual_norm": residual_after,
                "increment_norm": increment_norm * step_length,
                "active_contact_points": row["active_contact_points_after"],
                "negative_gap_sum": row["negative_gap_sum_after"],
                "reaction_norm": row["reaction_norm_after"],
                "max_penetration": row["max_penetration_after"],
                "step_length": step_length,
                "history": history,
            }
            return state, info

    if not failure_reason:
        failure_reason = "Reached max_newton_iter without satisfying convergence tolerances"

    last = history[-1] if history else {
        "increment_norm": np.inf,
        "active_contact_points_after": 0,
        "negative_gap_sum_after": 0.0,
        "reaction_norm_after": 0.0,
        "max_penetration_after": 0.0,
        "residual_norm": np.inf,
        "step_length": 0.0,
        "linear_solver_mode": linear_solver_mode,
        "ksp_type": "preonly" if linear_solver_mode == "lu" else ksp_type,
        "pc_type": "lu" if linear_solver_mode == "lu" else pc_type,
        "block_pc_name": "global_lu" if linear_solver_mode == "lu" else block_pc_name,
        "linear_converged": False,
        "linear_iterations": 0,
        "ksp_reason": 0,
    }
    total_linear_iterations = int(sum(item["linear_iterations"] for item in history))
    info = {
        "newton_iterations": len(history),
        "converged": False,
        "failure_reason": failure_reason,
        "backend": backend,
        "linear_solver_mode": last["linear_solver_mode"],
        "ksp_type": last["ksp_type"],
        "pc_type": last["pc_type"],
        "block_pc_name": last["block_pc_name"],
        "linear_converged": last["linear_converged"],
        "linear_iterations": last["linear_iterations"],
        "total_linear_iterations": total_linear_iterations,
        "ksp_reason": last["ksp_reason"],
        "residual_norm": last["residual_norm"],
        "increment_norm": last["increment_norm"],
        "active_contact_points": last["active_contact_points_after"],
        "negative_gap_sum": last["negative_gap_sum_after"],
        "reaction_norm": last["reaction_norm_after"],
        "max_penetration": last["max_penetration_after"],
        "step_length": last["step_length"],
        "history": history,
    }
    return state, info


def solve_monolithic_contact_loadpath(
    state0,
    cfg,
    load_schedule,
    *,
    backend="dense",
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    ksp_rtol=1e-10,
    ksp_atol=1e-12,
    ksp_max_it=400,
    max_newton_iter=15,
    max_cutbacks=6,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=False,
    initial_damping=1.0,
    max_backtracks=8,
    backtrack_factor=0.5,
    write_outputs=True,
    history_path=None,
    verbose=True,
    min_cutback_increment=1e-8,
):
    """Advance a load path with the monolithic Newton step."""
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
            backend=backend,
            linear_solver_mode=linear_solver_mode,
            ksp_type=ksp_type,
            pc_type=pc_type,
            block_pc_name=block_pc_name,
            ksp_rtol=ksp_rtol,
            ksp_atol=ksp_atol,
            ksp_max_it=ksp_max_it,
            max_newton_iter=max_newton_iter,
            tol_res=tol_res,
            tol_inc=tol_inc,
            line_search=line_search,
            initial_damping=initial_damping,
            max_backtracks=max_backtracks,
            backtrack_factor=backtrack_factor,
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
            "backend": backend,
            "linear_solver_mode": step_info["linear_solver_mode"],
            "ksp_type": step_info["ksp_type"],
            "pc_type": step_info["pc_type"],
            "block_pc_name": step_info["block_pc_name"],
            "linear_converged": bool(step_info["linear_converged"]),
            "linear_iterations": int(step_info["total_linear_iterations"]),
            "ksp_reason": int(step_info["ksp_reason"]),
            "newton_iterations": int(step_info["newton_iterations"]),
            "residual_norm": float(step_info["residual_norm"]),
            "increment_norm": float(step_info["increment_norm"]),
            "active_contact_points": int(step_info["active_contact_points"]),
            "negative_gap_sum": float(step_info["negative_gap_sum"]),
            "reaction_norm": float(step_info["reaction_norm"]),
            "max_penetration": float(step_info["max_penetration"]),
            "cutback_level": int(step_data.get("cutback_level", 0)),
            "step_length": float(step_info.get("step_length", 0.0)),
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

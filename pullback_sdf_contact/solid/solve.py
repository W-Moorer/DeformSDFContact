import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


def _owned_bc_dofs(bcs):
    constrained = []
    for bc in bcs:
        dofs, first_ghost = bc.dof_indices()
        constrained.extend(dofs[:first_ghost].tolist())
    if not constrained:
        return np.array([], dtype=np.int32)
    return np.unique(np.asarray(constrained, dtype=np.int32))


def mask_contact_tangent_for_bcs(contact_tangent_uu, bcs):
    """Return a dense contact tangent with constrained rows and columns zeroed."""
    tangent = np.array(contact_tangent_uu, dtype=np.float64, copy=True)
    if tangent.ndim != 2 or tangent.shape[0] != tangent.shape[1]:
        raise ValueError("contact_tangent_uu must be a square dense matrix")
    constrained_dofs = _owned_bc_dofs(bcs)
    if constrained_dofs.size > 0:
        tangent[constrained_dofs, :] = 0.0
        tangent[:, constrained_dofs] = 0.0
    return tangent, constrained_dofs


def dense_array_from_petsc_mat(A):
    """Extract a dense NumPy copy of a PETSc matrix on the current rank."""
    nrows, ncols = A.getSize()
    rows = np.arange(nrows, dtype=np.int32)
    cols = np.arange(ncols, dtype=np.int32)
    return np.asarray(A.getValues(rows, cols), dtype=np.float64)


def diagnose_dense_operator(matrix):
    """Return basic norm, symmetry, and extreme-entry diagnostics for a dense operator."""
    matrix = np.asarray(matrix, dtype=np.float64)
    fro_norm = float(np.linalg.norm(matrix))
    max_abs_entry = float(np.max(np.abs(matrix))) if matrix.size > 0 else 0.0
    if fro_norm > 0.0:
        symmetry_error = float(np.linalg.norm(matrix - matrix.T) / fro_norm)
    else:
        symmetry_error = 0.0
    return {
        "fro_norm": fro_norm,
        "symmetry_error": symmetry_error,
        "max_abs_entry": max_abs_entry,
    }


def add_dense_operator_to_petsc_mat(A, operator, scale=1.0):
    """Add a dense operator to a PETSc matrix on the current rank."""
    tangent = np.asarray(operator, dtype=np.float64)
    if tangent.ndim != 2 or tangent.shape[0] != tangent.shape[1]:
        raise ValueError("dense operator must be a square matrix")
    if scale != 1.0:
        tangent = scale * tangent
    nrows = tangent.shape[0]
    rows = np.arange(nrows, dtype=np.int32)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    A.setValuesLocal(rows, rows, tangent, addv=PETSc.InsertMode.ADD_VALUES)
    A.assemble()


def build_contact_structure_terms(
    reference_u,
    contact_rhs,
    contact_tangent_uu,
    bcs,
    *,
    contact_structure_mode="rhs_only",
):
    """Build matrix and rhs contact contributions for the requested structure-step mode."""
    valid_modes = {"rhs_only", "signed_tangent_only", "consistent_linearized"}
    if contact_structure_mode not in valid_modes:
        raise ValueError(
            f"Unsupported contact_structure_mode: {contact_structure_mode}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    constrained_dofs = _owned_bc_dofs(bcs)
    rhs = None
    if contact_rhs is not None:
        rhs = np.asarray(contact_rhs, dtype=np.float64).copy()
        if constrained_dofs.size > 0:
            rhs[constrained_dofs] = 0.0

    masked_tangent = None
    matrix_scale = 0.0
    affine_rhs = None
    if contact_tangent_uu is not None:
        masked_tangent, _ = mask_contact_tangent_for_bcs(contact_tangent_uu, bcs)

    if contact_structure_mode == "rhs_only":
        matrix_scale = 0.0
    elif contact_structure_mode == "signed_tangent_only":
        matrix_scale = -1.0
    elif contact_structure_mode == "consistent_linearized":
        matrix_scale = -1.0
        if masked_tangent is None:
            raise ValueError("consistent_linearized mode requires contact_tangent_uu")
        if reference_u is None:
            raise ValueError("consistent_linearized mode requires reference_u")
        affine_rhs = -masked_tangent.dot(np.asarray(reference_u, dtype=np.float64))
        if rhs is None:
            rhs = affine_rhs.copy()
        else:
            rhs = rhs + affine_rhs

    return {
        "rhs": rhs,
        "masked_tangent": masked_tangent,
        "matrix_scale": matrix_scale,
        "affine_rhs": affine_rhs,
        "constrained_dofs": constrained_dofs,
        "contact_structure_mode": contact_structure_mode,
    }


def assemble_linear_solid_system(
    u,
    R_form,
    bcs,
    contact_rhs=None,
    contact_tangent_uu=None,
    contact_structure_mode="rhs_only",
    reference_u=None,
    contact_rhs_sign=1.0,
    contact_tangent_sign=1.0,
):
    """Assemble the linear solid system with optional contact load and tangent."""
    du = ufl.TrialFunction(u.function_space)
    zero_u = fem.Function(u.function_space)
    a_form = ufl.derivative(R_form, u, du)
    L_form = -ufl.replace(R_form, {u: zero_u})

    A = fem.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = fem.assemble_vector(L_form)
    fem.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    if contact_rhs is not None and np.asarray(contact_rhs).shape[0] != b.getLocalSize():
        raise ValueError(f"contact_rhs has size {np.asarray(contact_rhs).shape[0]}, expected {b.getLocalSize()}")
    if reference_u is None:
        reference_u = u.vector.array_r.copy()

    contact_terms = build_contact_structure_terms(
        reference_u,
        contact_rhs,
        contact_tangent_uu,
        bcs,
        contact_structure_mode=contact_structure_mode,
    )
    constrained_dofs = contact_terms["constrained_dofs"]
    used_contact_rhs = contact_terms["rhs"] is not None
    used_contact_tangent_uu = (
        contact_terms["masked_tangent"] is not None and abs(contact_terms["matrix_scale"]) > 0.0
    )

    effective_rhs = contact_terms["rhs"]
    if effective_rhs is not None:
        b.array[:] += float(contact_rhs_sign) * effective_rhs

    masked_contact_tangent = contact_terms["masked_tangent"]
    effective_tangent_scale = float(contact_tangent_sign) * float(contact_terms["matrix_scale"])
    if masked_contact_tangent is not None and abs(effective_tangent_scale) > 0.0:
        add_dense_operator_to_petsc_mat(A, masked_contact_tangent, scale=effective_tangent_scale)

    fem.set_bc(b, bcs)
    constrained_rhs_max = (
        float(np.max(np.abs(b.array[constrained_dofs]))) if constrained_dofs.size > 0 else 0.0
    )
    constrained_tangent_max = 0.0
    if masked_contact_tangent is not None and constrained_dofs.size > 0:
        constrained_tangent_max = float(
            max(
                np.max(np.abs(masked_contact_tangent[constrained_dofs, :])),
                np.max(np.abs(masked_contact_tangent[:, constrained_dofs])),
            )
        )
    meta = {
        "rhs_norm": float(np.linalg.norm(b.array)),
        "used_contact_rhs": used_contact_rhs,
        "used_contact_tangent_uu": used_contact_tangent_uu,
        "contact_structure_mode": contact_structure_mode,
        "contact_rhs_sign": float(contact_rhs_sign),
        "contact_tangent_sign": effective_tangent_scale,
        "constrained_dofs": constrained_dofs,
        "constrained_rhs_max": constrained_rhs_max,
        "constrained_tangent_max": constrained_tangent_max,
        "affine_rhs_norm": (
            float(np.linalg.norm(contact_terms["affine_rhs"]))
            if contact_terms["affine_rhs"] is not None
            else 0.0
        ),
        "effective_contact_rhs_norm": (
            float(np.linalg.norm(effective_rhs)) if effective_rhs is not None else 0.0
        ),
        "contact_tangent_diagnostics": (
            diagnose_dense_operator(effective_tangent_scale * masked_contact_tangent)
            if masked_contact_tangent is not None
            else None
        ),
    }
    return A, b, meta


def solve_linear_solid(u, R_form, bcs):
    return solve_linear_solid_with_contact(u, R_form, bcs, contact_rhs=None)


def solve_linear_solid_with_contact(
    u,
    R_form,
    bcs,
    contact_rhs=None,
    contact_tangent_uu=None,
    contact_structure_mode="rhs_only",
    reference_u=None,
    contact_rhs_sign=1.0,
    contact_tangent_sign=1.0,
    return_system=False,
    verbose=True,
):
    """Solve one linear solid step with optional assembled contact contributions."""
    A, b, meta = assemble_linear_solid_system(
        u,
        R_form,
        bcs,
        contact_rhs=contact_rhs,
        contact_tangent_uu=contact_tangent_uu,
        contact_structure_mode=contact_structure_mode,
        reference_u=reference_u,
        contact_rhs_sign=contact_rhs_sign,
        contact_tangent_sign=contact_tangent_sign,
    )

    solver = PETSc.KSP().create(u.function_space.mesh.mpi_comm())
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.solve(b, u.vector)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    reason = solver.getConvergedReason()
    info = {
        "linear_iterations": solver.getIterationNumber(),
        "converged": reason > 0,
        "ksp_reason": reason,
        "rhs_norm": meta["rhs_norm"],
        "used_contact_rhs": meta["used_contact_rhs"],
        "used_contact_tangent_uu": meta["used_contact_tangent_uu"],
        "contact_structure_mode": meta["contact_structure_mode"],
        "contact_rhs_sign": meta["contact_rhs_sign"],
        "contact_tangent_sign": meta["contact_tangent_sign"],
        "constrained_rhs_max": meta["constrained_rhs_max"],
        "constrained_tangent_max": meta["constrained_tangent_max"],
        "affine_rhs_norm": meta["affine_rhs_norm"],
        "effective_contact_rhs_norm": meta["effective_contact_rhs_norm"],
        "contact_tangent_diagnostics": meta["contact_tangent_diagnostics"],
    }
    if return_system:
        info["A"] = A
        info["b"] = b
    return info

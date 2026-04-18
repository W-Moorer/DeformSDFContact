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


def _apply_contact_tangent(A, contact_tangent_uu, constrained_dofs):
    tangent = np.array(contact_tangent_uu, dtype=np.float64, copy=True)
    if tangent.ndim != 2 or tangent.shape[0] != tangent.shape[1]:
        raise ValueError("contact_tangent_uu must be a square dense matrix")

    if constrained_dofs.size > 0:
        tangent[constrained_dofs, :] = 0.0
        tangent[:, constrained_dofs] = 0.0

    nrows = tangent.shape[0]
    rows = np.arange(nrows, dtype=np.int32)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    A.setValuesLocal(rows, rows, tangent, addv=PETSc.InsertMode.ADD_VALUES)
    A.assemble()


def assemble_linear_solid_system(u, R_form, bcs, contact_rhs=None, contact_tangent_uu=None):
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

    constrained_dofs = _owned_bc_dofs(bcs)
    used_contact_rhs = contact_rhs is not None
    used_contact_tangent_uu = contact_tangent_uu is not None

    if contact_rhs is not None:
        rhs = np.asarray(contact_rhs, dtype=np.float64).copy()
        if rhs.shape[0] != b.getLocalSize():
            raise ValueError(f"contact_rhs has size {rhs.shape[0]}, expected {b.getLocalSize()}")
        if constrained_dofs.size > 0:
            rhs[constrained_dofs] = 0.0
        b.array[:] += rhs

    if contact_tangent_uu is not None:
        _apply_contact_tangent(A, contact_tangent_uu, constrained_dofs)

    fem.set_bc(b, bcs)
    meta = {
        "rhs_norm": float(np.linalg.norm(b.array)),
        "used_contact_rhs": used_contact_rhs,
        "used_contact_tangent_uu": used_contact_tangent_uu,
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
    }
    if return_system:
        info["A"] = A
        info["b"] = b
    return info

from dolfinx import fem
from petsc4py import PETSc
import ufl


def solve_linear_solid(u, R_form, bcs):
    du = ufl.TrialFunction(u.function_space)
    zero_u = fem.Function(u.function_space)
    a_form = ufl.derivative(R_form, u, du)
    L_form = -ufl.replace(R_form, {u: zero_u})

    problem = fem.LinearProblem(
        a_form,
        L_form,
        bcs=bcs,
        u=u,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem.solve()

    ksp = problem.solver
    reason = ksp.getConvergedReason()
    iterations = ksp.getIterationNumber()
    return {
        "linear_iterations": iterations,
        "converged": reason > 0,
        "ksp_reason": reason,
    }

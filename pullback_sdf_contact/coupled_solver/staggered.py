# DOLFINx 0.3 exposes NonlinearProblem/NewtonSolver from dolfinx.fem and
# dolfinx.nls instead of the newer *.petsc modules.
from dolfinx import fem, nls


def solve_sdf_subproblem(phi, R_phi_form, K_phi_phi_form, bcs):
    problem = fem.NonlinearProblem(R_phi_form, phi, bcs=bcs, J=K_phi_phi_form)
    solver = nls.NewtonSolver(phi.function_space.mesh.mpi_comm(), problem)
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 30
    n, converged = solver.solve(phi)
    return n, converged


def solve_staggered(state, cfg):
    """
    第一版：
    只求解 pull-back SDF 子问题。
    结构问题在当前版本中不更新，只作为已知场输入。
    """
    n, converged = solve_sdf_subproblem(
        state["phi"],
        state["R_phi_form"],
        state["K_phi_phi_form"],
        state["phi_bcs"],
    )
    return {"sdf_iterations": n, "sdf_converged": converged}

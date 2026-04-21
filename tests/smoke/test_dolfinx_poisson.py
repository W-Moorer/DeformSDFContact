#!/usr/bin/env python3
"""Smoke test: minimal DOLFINx Poisson assembly and PETSc solve.

DOLFINx 0.3.0 compatible writing:
- uses dolfinx.generation.UnitSquareMesh instead of newer mesh.create_unit_square
- uses fem.FunctionSpace(mesh, ("Lagrange", 1))
- uses fem.LinearProblem from the 0.3.0 API surface
"""

import numpy as np


def run() -> None:
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, generation
    import ufl

    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    boundary_dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        ),
    )
    bc = fem.DirichletBC(u_bc, boundary_dofs)

    source = fem.Constant(mesh, PETSc.ScalarType(1.0))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(source, v) * ufl.dx

    problem = fem.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    cell_count = mesh.topology.index_map(mesh.topology.dim).size_global
    dof_count = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    solution_norm = uh.vector.norm()

    assert cell_count > 0
    assert dof_count > 0
    assert solution_norm > 0.0

    print("cells", cell_count)
    print("dofs", dof_count)
    print("ksp", problem.solver.getType())
    print("pc", problem.solver.getPC().getType())
    print("solution_norm", solution_norm)


def test_dolfinx_poisson() -> None:
    run()


if __name__ == "__main__":
    run()

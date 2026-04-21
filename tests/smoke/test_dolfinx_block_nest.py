#!/usr/bin/env python3
"""Smoke test: minimal DOLFINx block/nest assembly into a PETSc Mat.

DOLFINx 0.3.0 compatible writing:
- uses dolfinx.generation.UnitSquareMesh
- uses FunctionSpace.clone() plus fem.assemble_matrix_nest from the 0.3.0 API
- keeps the example linear and block-structured without mixing in geometry logic
"""


def run() -> None:
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, generation
    import ufl

    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    W = V.clone()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(W)
    q = ufl.TestFunction(W)

    block_forms = [
        [ufl.inner(u, v) * ufl.dx, None],
        [None, ufl.inner(p, q) * ufl.dx],
    ]

    A = fem.assemble_matrix_nest(block_forms)
    A.assemble()

    sub00 = A.getNestSubMatrix(0, 0)
    sub11 = A.getNestSubMatrix(1, 1)

    assert isinstance(A, PETSc.Mat)
    assert A.getType() == "nest"
    assert sub00.getSize()[0] > 0
    assert sub11.getSize()[0] > 0

    print("mat_type", A.getType())
    print("sub00_size", sub00.getSize())
    print("sub11_size", sub11.getSize())


def test_dolfinx_block_nest() -> None:
    run()


if __name__ == "__main__":
    run()

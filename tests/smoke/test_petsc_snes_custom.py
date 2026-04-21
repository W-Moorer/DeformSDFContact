#!/usr/bin/env python3
"""Smoke test: solve a 1x1 nonlinear system with explicit SNES residual/Jacobian."""

import math


def run() -> None:
    from petsc4py import PETSc

    x = PETSc.Vec().createSeq(1)
    x.setValue(0, 1.5)
    x.assemble()

    residual = PETSc.Vec().createSeq(1)
    jacobian = PETSc.Mat().createAIJ([1, 1], nnz=1)
    jacobian.setUp()

    def form_function(snes, x_vec, f_vec):
        f_vec.setValue(0, x_vec.getValue(0) ** 2 - 2.0)
        f_vec.assemble()

    def form_jacobian(snes, x_vec, A, P):
        value = 2.0 * x_vec.getValue(0)

        A.zeroEntries()
        A.setValue(0, 0, value)
        A.assemble()

        if A != P:
            P.zeroEntries()
            P.setValue(0, 0, value)
            P.assemble()

    snes = PETSc.SNES().create()
    snes.setType("newtonls")
    snes.setFunction(form_function, residual)
    snes.setJacobian(form_jacobian, jacobian)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.solve(None, x)

    root = x.getValue(0)
    reason = snes.getConvergedReason()
    iterations = snes.getIterationNumber()

    assert abs(root - math.sqrt(2.0)) < 1.0e-10
    assert reason > 0

    print("SNES_type", snes.getType())
    print("root", root)
    print("reason", reason)
    print("iterations", iterations)


def test_petsc_snes_custom() -> None:
    run()


if __name__ == "__main__":
    run()

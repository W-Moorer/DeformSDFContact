from mpi4py import MPI
from dolfinx import fem
import numpy as np
import ufl


def _global_sqrt(domain, form):
    value = fem.assemble_scalar(form)
    value = domain.mpi_comm().allreduce(value, op=MPI.SUM)
    return float(np.sqrt(max(value, 0.0)))


def l2_error(domain, uh, u_exact, dx):
    diff = uh - u_exact
    return _global_sqrt(domain, ufl.inner(diff, diff) * dx)


def h1_semi_error(domain, uh, u_exact, dx):
    diff = uh - u_exact
    return _global_sqrt(domain, ufl.inner(ufl.grad(diff), ufl.grad(diff)) * dx)

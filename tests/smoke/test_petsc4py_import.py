#!/usr/bin/env python3
"""Smoke test: verify mpi4py/petsc4py import and report PETSc runtime details."""


def run() -> None:
    from mpi4py import MPI
    from petsc4py import PETSc

    mpi_version = MPI.Get_library_version().splitlines()[0]
    petsc_version = PETSc.Sys.getVersion()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    assert isinstance(petsc_version, tuple)
    assert len(petsc_version) == 3
    assert size >= 1

    print("MPI_version", mpi_version)
    print("MPI_rank_size", rank, size)
    print("PETSc_version", petsc_version)


def test_petsc4py_import() -> None:
    run()


if __name__ == "__main__":
    run()

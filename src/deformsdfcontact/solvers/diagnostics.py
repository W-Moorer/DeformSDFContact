"""Prototype diagnostics helpers for the monolithic solver path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend.dolfinx0p3.contact_summary import ContactAssemblySummary


@dataclass(frozen=True)
class ResidualNormSplit:
    """Split residual norms for the monolithic `(u, phi)` system."""

    total: float
    u: float
    phi: float


@dataclass(frozen=True)
class BlockMatrixDescription:
    """Prototype block-structure description for a monolithic Jacobian."""

    shape: tuple[int, int]
    block_shapes: dict[str, tuple[int, int]]
    block_nnz: dict[str, int]
    total_nnz: int


@dataclass(frozen=True)
class NonlinearIterationDiagnostics:
    """Prototype diagnostics entry for one nonlinear iteration."""

    iteration: int
    residual_split: ResidualNormSplit


@dataclass(frozen=True)
class SolverDiagnosticsResult:
    """Collected diagnostics for one prototype nonlinear solve."""

    iterations: tuple[NonlinearIterationDiagnostics, ...]
    final_residual_split: ResidualNormSplit
    jacobian_description: BlockMatrixDescription
    converged_reason: int
    iteration_count: int
    linear_solve_iterations: int
    linear_solve_failures: int
    contact_summary: ContactAssemblySummary | None = None


def _as_vector_array(vector: object) -> np.ndarray:
    if hasattr(vector, "getArray"):
        return np.asarray(vector.getArray(), dtype=float)
    return np.asarray(vector, dtype=float)


def split_residual_norms(layout: object, residual: object) -> ResidualNormSplit:
    """Return total and per-block residual norms."""

    array = _as_vector_array(residual)
    if array.shape != (layout.total_dofs,):
        raise ValueError(f"residual must have shape ({layout.total_dofs},), got {array.shape!r}")
    u_array = array[: layout.ndof_u]
    phi_array = array[layout.phi_offset :]
    return ResidualNormSplit(
        total=float(np.linalg.norm(array)),
        u=float(np.linalg.norm(u_array)),
        phi=float(np.linalg.norm(phi_array)),
    )


def describe_block_matrix(layout: object, matrix: object) -> BlockMatrixDescription:
    """Return a tiny-problem block description of one assembled Jacobian."""

    nrow, ncol = matrix.getSize()
    rows = np.arange(nrow, dtype=np.int32)
    cols = np.arange(ncol, dtype=np.int32)
    dense = np.asarray(matrix.getValues(rows, cols), dtype=float)
    uu = dense[: layout.ndof_u, : layout.ndof_u]
    uphi = dense[: layout.ndof_u, layout.phi_offset :]
    phiu = dense[layout.phi_offset :, : layout.ndof_u]
    phiphi = dense[layout.phi_offset :, layout.phi_offset :]
    block_nnz = {
        "uu": int(np.count_nonzero(uu)),
        "uphi": int(np.count_nonzero(uphi)),
        "phiu": int(np.count_nonzero(phiu)),
        "phiphi": int(np.count_nonzero(phiphi)),
    }
    return BlockMatrixDescription(
        shape=(nrow, ncol),
        block_shapes=dict(layout.block_shapes),
        block_nnz=block_nnz,
        total_nnz=int(np.count_nonzero(dense)),
    )


def monitor_snes_iteration(problem: object, snes: object, iteration: int, residual_norm: float) -> NonlinearIterationDiagnostics:
    """Return one prototype nonlinear-iteration diagnostics entry."""

    del residual_norm  # The split norms are recomputed from the current state.
    residual = problem.assemble_residual(snes.getSolution())
    return NonlinearIterationDiagnostics(
        iteration=int(iteration),
        residual_split=split_residual_norms(problem.layout, residual),
    )


__all__ = [
    "BlockMatrixDescription",
    "NonlinearIterationDiagnostics",
    "ResidualNormSplit",
    "SolverDiagnosticsResult",
    "describe_block_matrix",
    "monitor_snes_iteration",
    "split_residual_norms",
]

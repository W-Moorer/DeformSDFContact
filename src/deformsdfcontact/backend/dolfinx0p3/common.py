"""Shared geometry and PETSc helpers for the DOLFINx 0.3.0 transition adapter."""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc


def function_space_dimension(space: object) -> int:
    """Return the global scalar dof count for a DOLFINx 0.3.0 function space."""

    return int(space.dofmap.index_map.size_global * space.dofmap.index_map_bs)


def expand_block_dofs(block_dofs: object, block_size: int) -> np.ndarray:
    """Expand blocked dof ids into scalar dof ids."""

    block_array = np.asarray(block_dofs, dtype=np.int32)
    if block_array.ndim != 1:
        raise ValueError(f"block_dofs must have shape (n,), got {block_array.shape!r}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size!r}")

    expanded = np.empty(block_array.shape[0] * block_size, dtype=np.int32)
    for i, dof in enumerate(block_array):
        start = i * block_size
        expanded[start : start + block_size] = block_size * int(dof) + np.arange(block_size, dtype=np.int32)
    return expanded


def scalar_function_values(function: object) -> np.ndarray:
    """Return the global local-array view of a DOLFINx 0.3.0 function vector."""

    return np.asarray(function.vector.getArray(), dtype=float)


def barycentric_coordinates(point: object, triangle_coordinates: object) -> np.ndarray:
    """Return P1 triangle shape values at one point in physical coordinates."""

    point_array = np.asarray(point, dtype=float)
    coordinates = np.asarray(triangle_coordinates, dtype=float)
    if point_array.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {point_array.shape!r}")
    if coordinates.shape != (3, 2):
        raise ValueError(
            f"triangle_coordinates must have shape (3, 2), got {coordinates.shape!r}"
        )

    system = np.vstack([coordinates.T, np.ones(3, dtype=float)])
    rhs = np.array([point_array[0], point_array[1], 1.0], dtype=float)
    return np.linalg.solve(system, rhs)


def triangle_area_and_shape_gradients(triangle_coordinates: object) -> tuple[float, np.ndarray]:
    """Return the area and constant P1 shape gradients of one triangle."""

    coordinates = np.asarray(triangle_coordinates, dtype=float)
    if coordinates.shape != (3, 2):
        raise ValueError(
            f"triangle_coordinates must have shape (3, 2), got {coordinates.shape!r}"
        )

    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    x3, y3 = coordinates[2]
    det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(det_j) <= 1.0e-14:
        raise ValueError("triangle_coordinates are degenerate")

    area = 0.5 * abs(det_j)
    gradients = np.array(
        [
            [y2 - y3, x3 - x2],
            [y3 - y1, x1 - x3],
            [y1 - y2, x2 - x1],
        ],
        dtype=float,
    ) / det_j
    return area, gradients


def facet_midpoint_length_and_outward_normal(
    facet_coordinates: object,
    cell_centroid: object,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return facet midpoint, length, and outward normal with respect to one cell."""

    edge = np.asarray(facet_coordinates, dtype=float)
    centroid = np.asarray(cell_centroid, dtype=float)
    if edge.shape != (2, 2):
        raise ValueError(f"facet_coordinates must have shape (2, 2), got {edge.shape!r}")
    if centroid.shape != (2,):
        raise ValueError(f"cell_centroid must have shape (2,), got {centroid.shape!r}")

    tangent = edge[1] - edge[0]
    length = float(np.linalg.norm(tangent))
    if length <= 1.0e-14:
        raise ValueError("facet edge length must be positive")

    midpoint = 0.5 * (edge[0] + edge[1])
    normal = np.array([tangent[1], -tangent[0]], dtype=float) / length
    if float(np.dot(normal, midpoint - centroid)) < 0.0:
        normal *= -1.0
    return midpoint, length, normal


def create_petsc_vector(size: int, comm: object) -> PETSc.Vec:
    """Create a PETSc vector for the current transition adapter."""

    vec = PETSc.Vec().createMPI(size=size, comm=comm)
    vec.set(0.0)
    return vec


def create_petsc_matrix(nrow: int, ncol: int, comm: object) -> PETSc.Mat:
    """Create a generously preallocated AIJ matrix for the dry run."""

    mat = PETSc.Mat().createAIJ(size=(nrow, ncol), nnz=max(1, ncol), comm=comm)
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    mat.zeroEntries()
    return mat


def add_values_to_vector(vec: PETSc.Vec, dofs: object, values: object) -> None:
    """Add one local vector contribution."""

    vec.setValues(
        np.asarray(dofs, dtype=np.int32),
        np.asarray(values, dtype=float),
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def add_values_to_matrix(mat: PETSc.Mat, row_dofs: object, col_dofs: object, values: object) -> None:
    """Add one local matrix contribution."""

    mat.setValues(
        np.asarray(row_dofs, dtype=np.int32),
        np.asarray(col_dofs, dtype=np.int32),
        np.asarray(values, dtype=float),
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def assemble_petsc_objects(*objects: object) -> None:
    """Finalize a list of PETSc matrices or vectors."""

    for obj in objects:
        obj.assemblyBegin()
    for obj in objects:
        obj.assemblyEnd()


__all__ = [
    "add_values_to_matrix",
    "add_values_to_vector",
    "assemble_petsc_objects",
    "barycentric_coordinates",
    "create_petsc_matrix",
    "create_petsc_vector",
    "expand_block_dofs",
    "facet_midpoint_length_and_outward_normal",
    "function_space_dimension",
    "scalar_function_values",
    "triangle_area_and_shape_gradients",
]

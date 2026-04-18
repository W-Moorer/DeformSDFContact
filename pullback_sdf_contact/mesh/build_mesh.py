import numpy as np
from mpi4py import MPI
from dolfinx import generation, mesh
from dolfinx.cpp.mesh import CellType

from . import tags


def _create_sorted_meshtags(domain, dim, indices, values):
    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.int32)
    order = np.argsort(indices)
    return mesh.MeshTags(domain, dim, indices[order], values[order])


def create_reference_box(cfg):
    domain = generation.BoxMesh(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([cfg.Lx, cfg.Ly, cfg.Lz])],
        [cfg.nx, cfg.ny, cfg.nz],
        cell_type=CellType.hexahedron,
    )

    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_top(x):
        return np.isclose(x[2], cfg.Lz)

    def on_bottom(x):
        return np.isclose(x[2], 0.0)

    def on_left(x):
        return np.isclose(x[0], 0.0)

    def on_right(x):
        return np.isclose(x[0], cfg.Lx)

    def on_front(x):
        return np.isclose(x[1], 0.0)

    def on_back(x):
        return np.isclose(x[1], cfg.Ly)

    facet_sets = {
        tags.TOP: mesh.locate_entities_boundary(domain, fdim, on_top),
        tags.BOTTOM: mesh.locate_entities_boundary(domain, fdim, on_bottom),
        tags.LEFT: mesh.locate_entities_boundary(domain, fdim, on_left),
        tags.RIGHT: mesh.locate_entities_boundary(domain, fdim, on_right),
        tags.FRONT: mesh.locate_entities_boundary(domain, fdim, on_front),
        tags.BACK: mesh.locate_entities_boundary(domain, fdim, on_back),
    }

    facet_parts = [v for v in facet_sets.values() if len(v) > 0]
    if facet_parts:
        facet_indices = np.hstack(facet_parts).astype(np.int32)
        facet_values = np.hstack(
            [np.full(len(v), k, dtype=np.int32) for k, v in facet_sets.items() if len(v) > 0]
        ).astype(np.int32)
    else:
        facet_indices = np.array([], dtype=np.int32)
        facet_values = np.array([], dtype=np.int32)
    facet_tags = _create_sorted_meshtags(domain, fdim, facet_indices, facet_values)

    def in_band(x):
        return x[2] >= cfg.band_zmin

    cell_indices = mesh.locate_entities(domain, tdim, in_band)
    cell_values = np.full(len(cell_indices), tags.BAND, dtype=np.int32)
    cell_tags = _create_sorted_meshtags(domain, tdim, cell_indices, cell_values)

    return domain, cell_tags, facet_tags

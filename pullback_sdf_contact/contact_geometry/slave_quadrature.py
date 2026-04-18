from dataclasses import dataclass

import numpy as np


@dataclass
class SlaveQuadraturePoint:
    facet_id: int
    cell_id: int
    local_facet_id: int
    X_ref: np.ndarray
    weight: float
    xi_facet: np.ndarray


def _gauss_legendre_tensor_rule(quadrature_degree):
    npts = max(1, int(np.ceil((quadrature_degree + 1) / 2)))
    pts_1d, w_1d = np.polynomial.legendre.leggauss(npts)
    pts_1d = 0.5 * (pts_1d + 1.0)
    w_1d = 0.5 * w_1d

    rule = []
    for i, xi in enumerate(pts_1d):
        for j, eta in enumerate(pts_1d):
            rule.append((np.array([xi, eta], dtype=np.float64), float(w_1d[i] * w_1d[j])))
    return rule


def _facet_vertices(domain, facet_id):
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, 0)
    vertex_ids = domain.topology.connectivity(fdim, 0).links(facet_id)
    return domain.geometry.x[vertex_ids]


def _facet_cell_data(domain, facet_id):
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    cell_ids = domain.topology.connectivity(fdim, tdim).links(facet_id)
    if len(cell_ids) != 1:
        raise RuntimeError(f"boundary facet {facet_id} should have one adjacent cell, got {cell_ids}")

    cell_id = int(cell_ids[0])
    cell_facets = domain.topology.connectivity(tdim, fdim).links(cell_id)
    matches = np.where(cell_facets == facet_id)[0]
    if len(matches) != 1:
        raise RuntimeError(f"could not determine local facet id for facet {facet_id} in cell {cell_id}")

    return cell_id, int(matches[0])


def _map_facet_point(facet_vertices, xi_facet):
    mins = facet_vertices.min(axis=0)
    maxs = facet_vertices.max(axis=0)
    spans = maxs - mins
    fixed_axis = int(np.argmin(spans))
    varying_axes = [axis for axis in range(3) if axis != fixed_axis]

    X_ref = mins.copy()
    X_ref[varying_axes[0]] = mins[varying_axes[0]] + xi_facet[0] * spans[varying_axes[0]]
    X_ref[varying_axes[1]] = mins[varying_axes[1]] + xi_facet[1] * spans[varying_axes[1]]
    X_ref[fixed_axis] = mins[fixed_axis]

    area_jac = spans[varying_axes[0]] * spans[varying_axes[1]]
    return X_ref, float(area_jac)


def build_slave_quadrature(domain, facet_tags, slave_tag, quadrature_degree=2):
    """
    Build a tensor-product Gauss rule on the tagged slave facets.
    """
    slave_facets = facet_tags.indices[facet_tags.values == slave_tag]
    facet_rule = _gauss_legendre_tensor_rule(quadrature_degree)
    quadrature_points = []

    for facet_id in slave_facets:
        facet_vertices = _facet_vertices(domain, int(facet_id))
        cell_id, local_facet_id = _facet_cell_data(domain, int(facet_id))
        for xi_facet, weight_ref in facet_rule:
            X_ref, area_jac = _map_facet_point(facet_vertices, xi_facet)
            quadrature_points.append(
                SlaveQuadraturePoint(
                    facet_id=int(facet_id),
                    cell_id=cell_id,
                    local_facet_id=local_facet_id,
                    X_ref=X_ref,
                    weight=area_jac * weight_ref,
                    xi_facet=xi_facet.copy(),
                )
            )

    return quadrature_points


def slave_quadrature_stats(quadrature_points):
    facet_ids = {qp.facet_id for qp in quadrature_points}
    return {
        "slave_facets": len(facet_ids),
        "quadrature_points": len(quadrature_points),
        "reference_area": float(sum(qp.weight for qp in quadrature_points)),
    }

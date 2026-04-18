import numpy as np

from dolfinx import fem


def create_sdf_bcs(phi_space, facet_tags, top_tag, phi_value=0.0):
    domain = phi_space.mesh
    fdim = domain.topology.dim - 1
    top_facets = facet_tags.indices[facet_tags.values == top_tag]
    top_dofs = fem.locate_dofs_topological(phi_space, fdim, top_facets)
    bc_val = fem.Function(phi_space)
    bc_val.interpolate(lambda x: np.full(x.shape[1], phi_value))
    bc = fem.DirichletBC(bc_val, top_dofs)
    return [bc]

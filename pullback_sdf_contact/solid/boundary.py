import numpy as np

from dolfinx import fem
from petsc4py import PETSc

from mesh import tags


def create_solid_bcs(V_u, facet_tags, bottom_tag=tags.BOTTOM, top_tag=tags.TOP, top_uz=-0.1):
    domain = V_u.mesh
    fdim = domain.topology.dim - 1

    bottom_facets = facet_tags.indices[facet_tags.values == bottom_tag]
    bottom_dofs = fem.locate_dofs_topological(V_u, fdim, bottom_facets)
    u_bottom = fem.Function(V_u)
    u_bottom.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    bc_bottom = fem.DirichletBC(u_bottom, bottom_dofs)

    bcs = [bc_bottom]
    if top_uz is not None:
        top_facets = facet_tags.indices[facet_tags.values == top_tag]
        V_uz, _ = V_u.sub(2).collapse(True)
        top_dofs = fem.locate_dofs_topological((V_u.sub(2), V_uz), fdim, top_facets)
        u_top_z = fem.Function(V_uz)
        u_top_z.interpolate(lambda x: np.full(x.shape[1], top_uz))
        bc_top = fem.DirichletBC(u_top_z, top_dofs, V_u.sub(2))
        bcs.append(bc_top)

    return bcs


def update_top_displacement_value(u_top_z, top_uz):
    """Update the stored scalar top-displacement value in place."""
    with u_top_z.vector.localForm() as local_vector:
        local_vector.set(float(top_uz))
    u_top_z.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

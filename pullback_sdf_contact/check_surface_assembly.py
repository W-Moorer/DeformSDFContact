import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
import ufl
from dolfinx import fem

from config import MeshConfig, SDFConfig
from contact_geometry.slave_quadrature import build_slave_quadrature, slave_quadrature_stats
from contact_mechanics.assembled_surface import (
    assemble_contact_residual_surface,
    assemble_contact_tangent_uphi_surface,
    assemble_contact_tangent_uu_surface,
)
from mesh import tags
from mesh.build_mesh import create_reference_box
from sdf_field.boundary import create_sdf_bcs
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from solid.spaces import create_displacement_space
from coupled_solver.staggered import solve_staggered


def build_surface_state():
    lambda_z = 0.8
    slave_current_offset = np.array([0.0, 0.0, -0.04], dtype=np.float64)

    mesh_cfg = MeshConfig(nx=4, ny=4, nz=4)
    sdf_cfg = SDFConfig(beta=1e-6)

    domain, cell_tags, facet_tags = create_reference_box(mesh_cfg)
    V_u = create_displacement_space(domain, degree=1)
    V_phi = create_sdf_space(domain, degree=sdf_cfg.degree)

    u = fem.Function(V_u, name="u")
    phi = fem.Function(V_phi, name="phi")
    phi0 = fem.Function(V_phi, name="phi0")

    u.interpolate(
        lambda x: (
            0.0 * x[0],
            0.0 * x[1],
            (lambda_z - 1.0) * x[2],
        )
    )
    phi0.interpolate(lambda x: lambda_z * (x[2] - mesh_cfg.Lz))
    phi.interpolate(lambda x: x[2] - mesh_cfg.Lz)

    dx = ufl.Measure("dx", domain=domain)
    dx_band = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags, subdomain_id=tags.BAND
    )
    eta = ufl.TestFunction(V_phi)
    R_phi_expr, K_phi_phi_expr, K_phi_u_expr = sdf_forms(
        u, phi, eta, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    phi_bcs = create_sdf_bcs(V_phi, facet_tags, tags.TOP, phi_value=0.0)
    solve_staggered(
        {
            "domain": domain,
            "u": u,
            "phi": phi,
            "phi0": phi0,
            "R_phi_form": R_phi_expr,
            "K_phi_phi_form": K_phi_phi_expr,
            "K_phi_u_form": K_phi_u_expr,
            "phi_bcs": phi_bcs,
        },
        None,
    )

    return {
        "domain": domain,
        "facet_tags": facet_tags,
        "u": u,
        "phi": phi,
        "slave_current_offset": slave_current_offset,
    }


def main():
    state = build_surface_state()
    quadrature_points = build_slave_quadrature(
        state["domain"], state["facet_tags"], tags.TOP, quadrature_degree=2
    )
    quad_stats = slave_quadrature_stats(quadrature_points)

    residual_1, active_count, total_gap_measure, point_data = assemble_contact_residual_surface(
        quadrature_points, state, penalty=1.0
    )
    tangent_uphi, _ = assemble_contact_tangent_uphi_surface(quadrature_points, state, penalty=1.0)
    tangent_uu, _ = assemble_contact_tangent_uu_surface(quadrature_points, state, penalty=1.0)
    residual_10, _, _, _ = assemble_contact_residual_surface(quadrature_points, state, penalty=10.0)

    active_points = [data for data in point_data if data["g_n"] < 0.0]
    sample_points = active_points[:3] if active_points else point_data[:3]

    print("Surface assembly benchmark")
    print("")
    print(f"slave facets = {quad_stats['slave_facets']}")
    print(f"slave quadrature points = {quad_stats['quadrature_points']}")
    print(f"reference slave area = {quad_stats['reference_area']}")
    print("")
    print(f"active contact points = {active_count}")
    print(f"sum of weighted negative gaps = {total_gap_measure}")
    print("")
    print(f"||R_u^c||_2 = {np.linalg.norm(residual_1)}")
    print(f"||K_uphi^c||_F = {np.linalg.norm(tangent_uphi)}")
    print(f"||K_uu^c||_F = {np.linalg.norm(tangent_uu)}")
    print("")
    for i, data in enumerate(sample_points, start=1):
        print(f"point #{i}:")
        print(f"  X_slave_ref = {data['X_slave_ref']}")
        print(f"  X_c = {data['X_c']}")
        print(f"  g_n = {data['g_n']}")
        print(f"  ||G_u|| = {np.linalg.norm(data['G_u'])}")
        print(f"  ||G_a|| = {np.linalg.norm(data['G_a'])}")
    print("")
    print("penalty scaling check:")
    print(f"||R(eps=1)|| = {np.linalg.norm(residual_1)}")
    print(f"||R(eps=10)|| = {np.linalg.norm(residual_10)}")
    ratio = np.linalg.norm(residual_10) / (np.linalg.norm(residual_1) + 1e-16)
    print(f"ratio = {ratio}")


if __name__ == "__main__":
    main()

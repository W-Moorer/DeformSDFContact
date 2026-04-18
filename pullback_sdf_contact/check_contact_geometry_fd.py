import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem

from config import MeshConfig, SDFConfig
from contact_geometry.evaluate_phi import cell_bounds, eval_phi_quantities, eval_vector_function_data
from contact_geometry.query_point import solve_query_point
from contact_geometry.sensitivities import compute_gap_sensitivities
from coupled_solver.staggered import solve_staggered
from mesh import tags
from mesh.build_mesh import create_reference_box
from sdf_field.boundary import create_sdf_bcs
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from solid.spaces import create_displacement_space


def set_from_sum(target, first, alpha, second):
    with target.localForm() as local_target:
        local_target.set(0.0)
    target.axpy(1.0, first)
    target.axpy(alpha, second)
    target.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def find_candidate_cell(domain, X):
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    X = np.asarray(X, dtype=np.float64)

    for cell_id in range(num_cells):
        mins, maxs = cell_bounds(domain, cell_id)
        if np.all(X >= mins - 1e-12) and np.all(X <= maxs + 1e-12):
            return cell_id

    raise RuntimeError(f"point {X} is not inside any local cell")


def main():
    lambda_z = 0.8
    X_slave_ref = np.array([0.5, 0.5, 0.95], dtype=np.float64)
    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    mesh_cfg = MeshConfig()
    sdf_cfg = SDFConfig(beta=1e-6)

    domain, cell_tags, facet_tags = create_reference_box(mesh_cfg)
    V_u = create_displacement_space(domain, degree=1)
    V_phi = create_sdf_space(domain, degree=sdf_cfg.degree)

    u = fem.Function(V_u, name="u")
    phi = fem.Function(V_phi, name="phi")
    phi0 = fem.Function(V_phi, name="phi0")
    du_dir = fem.Function(V_u, name="du_dir")
    da_dir = fem.Function(V_phi, name="da_dir")
    u_base = fem.Function(V_u, name="u_base")
    phi_base = fem.Function(V_phi, name="phi_base")

    u.interpolate(
        lambda x: (
            0.0 * x[0],
            0.0 * x[1],
            (lambda_z - 1.0) * x[2],
        )
    )
    u_base.interpolate(
        lambda x: (
            0.0 * x[0],
            0.0 * x[1],
            (lambda_z - 1.0) * x[2],
        )
    )
    du_dir.interpolate(
        lambda x: (
            0.02 * x[0] * (1.0 - x[0]),
            0.01 * x[1] * (1.0 - x[1]),
            -0.03 * x[2],
        )
    )
    da_dir.interpolate(lambda x: 0.02 * x[0] * (1.0 - x[0]) + 0.01 * x[2])

    phi0.interpolate(lambda x: lambda_z * (x[2] - mesh_cfg.Lz))
    phi.interpolate(lambda x: x[2] - mesh_cfg.Lz)
    phi_base.interpolate(lambda x: x[2] - mesh_cfg.Lz)

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
    phi_base.vector.array_w[:] = phi.vector.array_r
    phi_base.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    candidate_cell = find_candidate_cell(domain, X_slave_ref)
    u_slave = eval_vector_function_data(u, X_slave_ref, candidate_cell)["value"]
    x_s = X_slave_ref + u_slave

    X_c, converged, iters = solve_query_point(
        x_s, u, candidate_cell=candidate_cell, X_init=X_slave_ref
    )
    g_n, normal, E, G_u, G_a = compute_gap_sensitivities(X_c, candidate_cell, u, phi)
    query_lin = E @ du_dir.vector.array_r
    gap_lin = float(G_u @ du_dir.vector.array_r + G_a @ da_dir.vector.array_r)

    print("Contact geometry finite-difference check")
    print("")
    print("Query point solve:")
    print(f"X_slave_ref = {X_slave_ref}")
    print(f"X_c = {X_c}")
    print(f"converged = {converged}")
    print(f"iters = {iters}")
    print("")
    print("E check:")
    for eps in eps_list:
        set_from_sum(u.vector, u_base.vector, eps, du_dir.vector)
        X_eps, conv_eps, it_eps = solve_query_point(
            x_s, u, candidate_cell=candidate_cell, X_init=X_c
        )
        fd = (X_eps - X_c) / eps
        diff = fd - query_lin
        abs_err = np.linalg.norm(diff)
        rel_err = abs_err / (np.linalg.norm(query_lin) + 1e-16)
        print(f"eps = {eps}")
        print(f"||fd - lin||_2 = {abs_err}")
        print(f"relative error = {rel_err}")

    set_from_sum(u.vector, u_base.vector, 0.0, du_dir.vector)

    print("")
    print("Gap derivative check:")
    print(f"g_n = {g_n}")
    print(f"||G_u du + G_a da|| = {abs(gap_lin)}")
    for eps in eps_list:
        set_from_sum(u.vector, u_base.vector, eps, du_dir.vector)
        set_from_sum(phi.vector, phi_base.vector, eps, da_dir.vector)

        X_eps, conv_eps, it_eps = solve_query_point(
            x_s, u, candidate_cell=candidate_cell, X_init=X_c
        )
        g_eps, _, _, _, _ = compute_gap_sensitivities(X_eps, candidate_cell, u, phi)
        fd = (g_eps - g_n) / eps
        abs_err = abs(fd - gap_lin)
        rel_err = abs_err / (abs(gap_lin) + 1e-16)
        print(f"eps = {eps}")
        print(f"||fd - lin||_2 = {abs_err}")
        print(f"relative error = {rel_err}")

    set_from_sum(u.vector, u_base.vector, 0.0, du_dir.vector)
    set_from_sum(phi.vector, phi_base.vector, 0.0, da_dir.vector)


if __name__ == "__main__":
    main()

import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from config import MeshConfig, SDFConfig
from contact_geometry.evaluate_phi import cell_bounds, eval_vector_function_data
from contact_geometry.query_point import solve_query_point
from contact_geometry.sensitivities import compute_gap_sensitivities
from contact_mechanics.single_point import (
    contact_residual_single_point,
    contact_tangent_uphi_single_point,
    contact_tangent_uu_single_point,
)
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


def build_base_state():
    lambda_z = 0.8
    X_slave_ref = np.array([0.5, 0.5, 0.95], dtype=np.float64)
    penalty = 1.0

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

    return {
        "domain": domain,
        "u": u,
        "phi": phi,
        "phi0": phi0,
        "du_dir": du_dir,
        "da_dir": da_dir,
        "u_base": u_base,
        "phi_base": phi_base,
        "candidate_cell": candidate_cell,
        "x_s": x_s,
        "X_slave_ref": X_slave_ref,
        "lambda_z": lambda_z,
        "penalty": penalty,
    }


def compute_contact_state(u, phi, candidate_cell, x_s, X_init, penalty):
    X_c, converged, iters = solve_query_point(
        x_s, u, candidate_cell=candidate_cell, X_init=X_init
    )
    g_n, normal, E, G_u, G_a = compute_gap_sensitivities(X_c, candidate_cell, u, phi)
    R_uc, lam, kn = contact_residual_single_point(g_n, G_u, penalty)
    return {
        "X_c": X_c,
        "converged": converged,
        "iters": iters,
        "g_n": g_n,
        "normal": normal,
        "E": E,
        "G_u": G_u,
        "G_a": G_a,
        "R_uc": R_uc,
        "lam": lam,
        "kn": kn,
    }


def main():
    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    state = build_base_state()

    base = compute_contact_state(
        state["u"],
        state["phi"],
        state["candidate_cell"],
        state["x_s"],
        state["X_slave_ref"],
        state["penalty"],
    )

    K_uphi_c, lam_uphi, kn_uphi = contact_tangent_uphi_single_point(
        base["g_n"], base["G_u"], base["G_a"], state["penalty"]
    )
    K_uu_c, lam_uu, kn_uu = contact_tangent_uu_single_point(
        base["g_n"], base["G_u"], state["penalty"]
    )

    da_vec = state["da_dir"].vector.array_r.copy()
    du_vec = state["du_dir"].vector.array_r.copy()
    lin_uphi = K_uphi_c @ da_vec
    lin_uu = K_uu_c @ du_vec

    print("Contact tangent finite-difference check")
    print("")
    print("Base contact state:")
    print(f"g_n = {base['g_n']}")
    print(f"lambda_n = {base['lam']}")
    print(f"k_n = {base['kn']}")
    print(f"||R_u^c||_2 = {np.linalg.norm(base['R_uc'])}")
    print("")
    print("K_uphi^c check:")
    print(f"||K_uphi^c * da_dir||_2 = {np.linalg.norm(lin_uphi)}")
    for eps in eps_list:
        set_from_sum(state["phi"].vector, state["phi_base"].vector, eps, state["da_dir"].vector)
        pert = compute_contact_state(
            state["u"],
            state["phi"],
            state["candidate_cell"],
            state["x_s"],
            base["X_c"],
            state["penalty"],
        )
        fd = (pert["R_uc"] - base["R_uc"]) / eps
        diff = fd - lin_uphi
        abs_err = np.linalg.norm(diff)
        rel_err = abs_err / (np.linalg.norm(lin_uphi) + 1e-16)
        print(f"eps = {eps}")
        print(f"||fd - lin||_2 = {abs_err}")
        print(f"relative error = {rel_err}")

    set_from_sum(state["phi"].vector, state["phi_base"].vector, 0.0, state["da_dir"].vector)

    print("")
    print("K_uu^c check:")
    print(f"||K_uu^c * du_dir||_2 = {np.linalg.norm(lin_uu)}")
    for eps in eps_list:
        set_from_sum(state["u"].vector, state["u_base"].vector, eps, state["du_dir"].vector)
        pert = compute_contact_state(
            state["u"],
            state["phi"],
            state["candidate_cell"],
            state["x_s"],
            base["X_c"],
            state["penalty"],
        )
        fd = (pert["R_uc"] - base["R_uc"]) / eps
        diff = fd - lin_uu
        abs_err = np.linalg.norm(diff)
        rel_err = abs_err / (np.linalg.norm(lin_uu) + 1e-16)
        print(f"eps = {eps}")
        print(f"||fd - lin||_2 = {abs_err}")
        print(f"relative error = {rel_err}")

    set_from_sum(state["u"].vector, state["u_base"].vector, 0.0, state["du_dir"].vector)


if __name__ == "__main__":
    main()

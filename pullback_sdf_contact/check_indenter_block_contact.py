import csv
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
import ufl
from dolfinx import fem

from config import MeshConfig, SDFConfig, SolidConfig, SolverConfig
from contact_geometry.slave_quadrature import build_slave_quadrature
from coupled_solver.staggered import solve_staggered_contact_loadpath
from mesh import tags
from mesh.build_mesh import create_reference_box
from post.xdmf import write_scalar_field
from sdf_field.boundary import create_sdf_bcs
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from solid.boundary import create_solid_bcs
from solid.forms import solid_forms
from solid.spaces import create_displacement_space


def build_indenter_state():
    mesh_cfg = MeshConfig(nx=4, ny=4, nz=4)
    solid_cfg = SolidConfig(E=100.0, nu=0.3, body_force_z=0.0)
    sdf_cfg = SDFConfig(beta=1e-6)
    solver_cfg = SolverConfig(max_it=12, verbose=False)
    penalty = 1.0

    domain, cell_tags, facet_tags = create_reference_box(mesh_cfg)
    V_u = create_displacement_space(domain, degree=1)
    V_phi = create_sdf_space(domain, degree=sdf_cfg.degree)

    u = fem.Function(V_u, name="u_contact_loadpath")
    phi = fem.Function(V_phi, name="phi_contact_loadpath")
    phi0 = fem.Function(V_phi, name="phi0_contact_loadpath")

    phi0.interpolate(lambda x: x[2] - mesh_cfg.Lz)
    phi.interpolate(lambda x: x[2] - mesh_cfg.Lz)

    dx = ufl.Measure("dx", domain=domain)
    dx_band = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags, subdomain_id=tags.BAND
    )

    v_u = ufl.TestFunction(V_u)
    R_u_form, _ = solid_forms(u, v_u, solid_cfg, dx)
    solid_bcs = create_solid_bcs(V_u, facet_tags, top_uz=None)

    eta = ufl.TestFunction(V_phi)
    R_phi_form, K_phi_phi_form, K_phi_u_form = sdf_forms(
        u, phi, eta, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    phi_bcs = create_sdf_bcs(V_phi, facet_tags, tags.TOP, phi_value=0.0)

    quadrature_points = build_slave_quadrature(domain, facet_tags, tags.TOP, quadrature_degree=2)

    state = {
        "domain": domain,
        "u": u,
        "phi": phi,
        "phi0": phi0,
        "R_u_form": R_u_form,
        "solid_bcs": solid_bcs,
        "R_phi_form": R_phi_form,
        "K_phi_phi_form": K_phi_phi_form,
        "K_phi_u_form": K_phi_u_form,
        "phi_bcs": phi_bcs,
        "quadrature_points": quadrature_points,
        "penalty": penalty,
        "slave_current_offset": np.zeros(3, dtype=np.float64),
        "current_load_value": 0.0,
    }
    return state, solver_cfg


def build_load_schedule(max_load=0.05, num_steps=11):
    values = np.linspace(0.0, max_load, num_steps)
    return [
        {"step": i + 1, "load_value": float(value), "label": f"indent_{i + 1}"}
        for i, value in enumerate(values)
    ]


def write_history_summary(history, summary_path):
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Indenter Block Contact Summary\n\n")
        for item in history:
            handle.write(
                "step={step}, load={load_value:.6f}, converged={converged}, "
                "outer_iterations={outer_iterations}, active={active_contact_points}, "
                "negative_gap_sum={negative_gap_sum:.6e}, reaction_norm={reaction_norm:.6e}\n".format(
                    **item
                )
            )


def main():
    state, solver_cfg = build_indenter_state()
    load_schedule = build_load_schedule()
    history_path = "contact_history.csv"
    summary_path = "contact_history_summary.txt"

    final_state, history = solve_staggered_contact_loadpath(
        state,
        solver_cfg,
        load_schedule,
        use_contact_tangent_uu=True,
        max_outer_iter=solver_cfg.max_it,
        max_cutbacks=6,
        write_outputs=True,
        history_path=history_path,
        verbose=True,
        step_verbose=False,
    )

    if final_state["domain"].mpi_comm().rank == 0:
        print("Rigid flat indenter vs elastic block")
        print("")
        for item in history:
            print(
                "step={step:02d} load={load_value:.4f} outer_iterations={outer_iterations} "
                "active_contact_points={active_contact_points} negative_gap_sum={negative_gap_sum:.6e} "
                "reaction_norm={reaction_norm:.6e}".format(**item)
            )

    if final_state["domain"].mpi_comm().rank == 0:
        write_history_summary(history, summary_path)

    write_scalar_field(final_state["domain"], final_state["u"], "output_u_contact_loadpath.xdmf")
    write_scalar_field(final_state["domain"], final_state["phi"], "output_phi_contact_loadpath.xdmf")


if __name__ == "__main__":
    main()

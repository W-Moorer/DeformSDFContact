import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from mpi4py import MPI
from dolfinx import fem
import ufl

from bench_affine_pullback import run_affine_benchmark
from config import MeshConfig, SolidConfig, SDFConfig, SolverConfig
from coupled_solver.staggered import solve_staggered
from mesh.build_mesh import create_reference_box
from mesh import tags
from post.xdmf import write_scalar_and_vector
from sdf_field.assembly_tools import assemble_sdf_objects
from sdf_field.boundary import create_sdf_bcs
from sdf_field.diagnostics import eikonal_residual_form
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from solid.boundary import create_solid_bcs
from solid.forms import solid_forms
from solid.solve import solve_linear_solid
from solid.spaces import create_displacement_space

MODE = "solid_then_sdf"


def solve_solid_then_sdf():
    mesh_cfg = MeshConfig()
    solid_cfg = SolidConfig()
    sdf_cfg = SDFConfig()
    solver_cfg = SolverConfig()

    domain, cell_tags, facet_tags = create_reference_box(mesh_cfg)
    V_u = create_displacement_space(domain, degree=1)
    V_phi = create_sdf_space(domain, degree=sdf_cfg.degree)

    u = fem.Function(V_u, name="u")
    phi = fem.Function(V_phi, name="phi")
    phi0 = fem.Function(V_phi, name="phi0")

    phi0.interpolate(lambda x: x[2] - mesh_cfg.Lz)
    phi.interpolate(lambda x: x[2] - mesh_cfg.Lz)

    dx = ufl.Measure("dx", domain=domain)
    dx_band = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags, subdomain_id=tags.BAND
    )

    v_u = ufl.TestFunction(V_u)
    R_u_form, K_u_form = solid_forms(u, v_u, solid_cfg, dx)
    solid_bcs = create_solid_bcs(V_u, facet_tags, top_uz=-0.1)
    solid_info = solve_linear_solid(u, R_u_form, solid_bcs)
    if domain.mpi_comm().rank == 0:
        print("Solid solve info:", solid_info)

    eta = ufl.TestFunction(V_phi)
    R_phi_form, K_phi_phi_form, K_phi_u_form = sdf_forms(
        u, phi, eta, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    _, _, K_phi_u_form_assembled = assemble_sdf_objects(
        u, phi, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    phi_bcs = create_sdf_bcs(V_phi, facet_tags, tags.TOP, phi_value=0.0)

    sdf_info = solve_staggered(
        {
            "domain": domain,
            "u": u,
            "phi": phi,
            "phi0": phi0,
            "R_phi_form": R_phi_form,
            "K_phi_phi_form": K_phi_phi_form,
            "K_phi_u_form": K_phi_u_form,
            "K_phi_u_form_assembled": K_phi_u_form_assembled,
            "phi_bcs": phi_bcs,
        },
        solver_cfg,
    )
    if domain.mpi_comm().rank == 0:
        print("SDF solve info:", sdf_info)

    q2 = fem.assemble_scalar(eikonal_residual_form(u, phi, dx_band))
    q2 = domain.mpi_comm().allreduce(q2, op=MPI.SUM)
    if domain.mpi_comm().rank == 0:
        print("Integrated squared Eikonal residual on band =", q2)

    write_scalar_and_vector(domain, u, phi, filename_prefix="output")


def main():
    if MODE == "affine_benchmark":
        run_affine_benchmark()
        return

    if MODE == "solid_then_sdf":
        solve_solid_then_sdf()
        return

    raise ValueError(f"Unsupported MODE: {MODE}")


if __name__ == "__main__":
    main()

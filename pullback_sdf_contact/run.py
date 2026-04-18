import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from mpi4py import MPI
from dolfinx import fem
import ufl

from config import MeshConfig, SolidConfig, SDFConfig, SolverConfig
from mesh.build_mesh import create_reference_box
from mesh import tags
from solid.spaces import create_displacement_space
from sdf_field.spaces import create_sdf_space
from sdf_field.forms import sdf_forms
from sdf_field.boundary import create_sdf_bcs
from sdf_field.diagnostics import eikonal_residual_form
from coupled_solver.staggered import solve_staggered
from post.xdmf import write_scalar_and_vector


def main():
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

    eta = ufl.TestFunction(V_phi)
    R_phi_form_expr, K_phi_phi_expr, K_phi_u_expr = sdf_forms(
        u, phi, eta, phi0, sdf_cfg.beta, dx_band
    )

    phi_bcs = create_sdf_bcs(V_phi, facet_tags, tags.TOP, phi_value=0.0)

    state = {
        "domain": domain,
        "u": u,
        "phi": phi,
        "phi0": phi0,
        "R_phi_form": R_phi_form_expr,
        "K_phi_phi_form": K_phi_phi_expr,
        "K_phi_u_form": K_phi_u_expr,
        "phi_bcs": phi_bcs,
    }

    info = solve_staggered(state, solver_cfg)
    if domain.mpi_comm().rank == 0:
        print("SDF solve info:", info)

    q2 = fem.assemble_scalar(eikonal_residual_form(u, phi, dx_band))
    q2 = domain.mpi_comm().allreduce(q2, op=MPI.SUM)
    if domain.mpi_comm().rank == 0:
        print("Integrated squared Eikonal residual on band =", q2)

    write_scalar_and_vector(domain, u, phi, filename_prefix="output")


if __name__ == "__main__":
    main()

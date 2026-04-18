import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from mpi4py import MPI
from dolfinx import fem
import ufl

from config import MeshConfig, SDFConfig
from mesh.build_mesh import create_reference_box
from mesh import tags
from solid.spaces import create_displacement_space
from sdf_field.boundary import create_sdf_bcs
from sdf_field.diagnostics import eikonal_residual_form
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from coupled_solver.staggered import solve_staggered
from post.errors import h1_semi_error, l2_error
from post.xdmf import write_scalar_and_vector, write_scalar_field


def run_affine_benchmark(lambda_z=0.8):
    mesh_cfg = MeshConfig()
    sdf_cfg = SDFConfig(beta=1e-6)

    domain, cell_tags, facet_tags = create_reference_box(mesh_cfg)
    V_u = create_displacement_space(domain, degree=1)
    V_phi = create_sdf_space(domain, degree=sdf_cfg.degree)

    u = fem.Function(V_u, name="u")
    phi = fem.Function(V_phi, name="phi")
    phi0 = fem.Function(V_phi, name="phi0")
    phi_exact = fem.Function(V_phi, name="phi_exact")

    u.interpolate(
        lambda x: (
            0.0 * x[0],
            0.0 * x[1],
            (lambda_z - 1.0) * x[2],
        )
    )
    phi0.interpolate(lambda x: lambda_z * (x[2] - mesh_cfg.Lz))
    phi.interpolate(lambda x: x[2] - mesh_cfg.Lz)
    phi_exact.interpolate(lambda x: lambda_z * (x[2] - mesh_cfg.Lz))

    dx = ufl.Measure("dx", domain=domain)
    dx_band = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags, subdomain_id=tags.BAND
    )
    eta = ufl.TestFunction(V_phi)
    R_phi_form, K_phi_phi_form, K_phi_u_form = sdf_forms(
        u, phi, eta, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    phi_bcs = create_sdf_bcs(V_phi, facet_tags, tags.TOP, phi_value=0.0)

    info = solve_staggered(
        {
            "domain": domain,
            "u": u,
            "phi": phi,
            "phi0": phi0,
            "R_phi_form": R_phi_form,
            "K_phi_phi_form": K_phi_phi_form,
            "K_phi_u_form": K_phi_u_form,
            "phi_bcs": phi_bcs,
        },
        None,
    )

    l2 = l2_error(domain, phi, phi_exact, dx_band)
    h1 = h1_semi_error(domain, phi, phi_exact, dx_band)
    q2 = fem.assemble_scalar(eikonal_residual_form(u, phi, dx_band))
    q2 = domain.mpi_comm().allreduce(q2, op=MPI.SUM)

    if domain.mpi_comm().rank == 0:
        print("Affine benchmark:")
        print(f"lambda_z = {lambda_z}")
        print("SDF solve info:", info)
        print(f"L2 error(phi, phi_exact) = {l2}")
        print(f"H1-semi error(phi, phi_exact) = {h1}")
        print(f"Integrated squared Eikonal residual = {q2}")

    write_scalar_and_vector(domain, u, phi, filename_prefix="output")
    write_scalar_field(domain, phi_exact, "output_phi_exact.xdmf")


def main():
    run_affine_benchmark()


if __name__ == "__main__":
    main()

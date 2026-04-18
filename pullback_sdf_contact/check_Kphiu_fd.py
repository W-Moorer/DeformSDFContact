import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
import ufl

from config import MeshConfig, SDFConfig
from coupled_solver.staggered import solve_staggered
from mesh import tags
from mesh.build_mesh import create_reference_box
from sdf_field.assembly_tools import assemble_sdf_objects
from sdf_field.boundary import create_sdf_bcs
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from solid.spaces import create_displacement_space


def assemble_residual_vector(R_form):
    b = fem.assemble_vector(R_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


def set_from_sum(target, first, alpha, second):
    with target.localForm() as local_target:
        local_target.set(0.0)
    target.axpy(1.0, first)
    target.axpy(alpha, second)
    target.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def main():
    lambda_z = 0.8
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
    u_base = fem.Function(V_u, name="u_base")

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

    R_phi_form, _, K_phi_u_form = assemble_sdf_objects(
        u, phi, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    R_base = assemble_residual_vector(R_phi_form)

    K_phi_u = fem.create_matrix(K_phi_u_form)
    K_phi_u.zeroEntries()
    fem.assemble_matrix(K_phi_u, K_phi_u_form)
    K_phi_u.assemble()

    delta_R_lin = fem.create_vector(R_phi_form)
    K_phi_u.mult(du_dir.vector, delta_R_lin)
    lin_norm = delta_R_lin.norm()

    if domain.mpi_comm().rank == 0:
        print("K_phi_u finite-difference check")
        print(f"lambda_z = {lambda_z}")
        print(f"||K_phi_u * du_dir||_2 = {lin_norm}")
        print("")
        print(f"{'eps':>12} {'||fd - lin||_2':>20} {'relative error':>20}")

    for eps in eps_list:
        set_from_sum(u.vector, u_base.vector, eps, du_dir.vector)

        R_eps = assemble_residual_vector(R_phi_form)
        delta_R_fd = R_eps.copy()
        delta_R_fd.axpy(-1.0, R_base)
        delta_R_fd.scale(1.0 / eps)

        diff = delta_R_fd.copy()
        diff.axpy(-1.0, delta_R_lin)
        abs_err = diff.norm()
        rel_err = abs_err / (lin_norm + 1e-16)

        if domain.mpi_comm().rank == 0:
            print(f"eps = {eps}")
            print(f"||fd - lin||_2 = {abs_err}")
            print(f"relative error = {rel_err}")
            print(f"{eps:12.1e} {abs_err:20.12e} {rel_err:20.12e}")

    set_from_sum(u.vector, u_base.vector, 0.0, du_dir.vector)


if __name__ == "__main__":
    main()

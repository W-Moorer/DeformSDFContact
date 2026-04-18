from dolfinx import io


def write_scalar_and_vector(domain, u, phi, filename_prefix="output"):
    with io.XDMFFile(domain.mpi_comm(), f"{filename_prefix}_u.xdmf", "w") as xdmf_u:
        xdmf_u.write_mesh(domain)
        xdmf_u.write_function(u)

    with io.XDMFFile(domain.mpi_comm(), f"{filename_prefix}_phi.xdmf", "w") as xdmf_phi:
        xdmf_phi.write_mesh(domain)
        xdmf_phi.write_function(phi)


def write_scalar_field(domain, field, filename):
    with io.XDMFFile(domain.mpi_comm(), filename, "w") as xdmf_file:
        xdmf_file.write_mesh(domain)
        xdmf_file.write_function(field)

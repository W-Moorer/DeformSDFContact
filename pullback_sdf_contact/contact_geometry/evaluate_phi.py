from functools import lru_cache

import basix
import numpy as np


def _require_hexahedron(cell_name):
    if cell_name != "hexahedron":
        raise NotImplementedError("only hexahedron cells are supported in this benchmark")


@lru_cache(maxsize=None)
def _scalar_basix_element(cell_name, degree):
    _require_hexahedron(cell_name)
    return basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        degree,
        basix.LatticeType.equispaced,
    )


def cell_bounds(domain, cell_id):
    geom_dofs = domain.geometry.dofmap.links(cell_id)
    x_g = domain.geometry.x[geom_dofs]
    return x_g.min(axis=0), x_g.max(axis=0)


def reference_coordinates(domain, cell_id, X):
    mins, maxs = cell_bounds(domain, cell_id)
    lengths = maxs - mins
    return (np.asarray(X, dtype=np.float64) - mins) / lengths


def eval_scalar_function_data(function, X_c, cell_id):
    V = function.function_space
    domain = V.mesh
    cell_name = domain.ufl_cell().cellname()
    degree = V.ufl_element().degree()
    xi = reference_coordinates(domain, cell_id, X_c)
    element = _scalar_basix_element(cell_name, degree)
    tab = element.tabulate(1, np.asarray([xi], dtype=np.float64))

    basis = tab[0, 0, :]
    grad_ref = np.vstack([tab[1, 0, :], tab[2, 0, :], tab[3, 0, :]])

    mins, maxs = cell_bounds(domain, cell_id)
    jac = np.diag(maxs - mins)
    grad_phys = np.linalg.solve(jac.T, grad_ref)

    cell_dofs = V.dofmap.list.links(cell_id)
    coeffs = function.vector.array_r[cell_dofs]
    value = float(basis @ coeffs)
    grad = grad_phys @ coeffs

    ndofs = function.vector.getLocalSize()
    N_row = np.zeros(ndofs, dtype=np.float64)
    N_row[cell_dofs] = basis

    B_mat = np.zeros((domain.geometry.dim, ndofs), dtype=np.float64)
    B_mat[:, cell_dofs] = grad_phys

    return {
        "value": value,
        "grad": grad,
        "N_row": N_row,
        "B_mat": B_mat,
        "cell_dofs": np.asarray(cell_dofs, dtype=np.int32),
        "basis_local": basis,
        "grad_local": grad_phys,
        "reference_point": xi,
    }


def eval_vector_function_data(function, X_c, cell_id):
    gdim = function.function_space.mesh.geometry.dim
    ndofs = function.vector.getLocalSize()

    value = np.zeros(gdim, dtype=np.float64)
    grad = np.zeros((gdim, gdim), dtype=np.float64)
    N_mat = np.zeros((gdim, ndofs), dtype=np.float64)

    for i in range(gdim):
        data_i = eval_scalar_function_data(function.sub(i), X_c, cell_id)
        value[i] = data_i["value"]
        grad[i, :] = data_i["grad"]
        N_mat[i, :] = data_i["N_row"]

    return {"value": value, "grad": grad, "N_mat": N_mat}


def eval_phi_quantities(X_c, cell_id, phi_function):
    """
    Returns:
        phi_val
        grad_phi
        hess_phi
        Nphi_row
        Bphi_mat
    """
    data = eval_scalar_function_data(phi_function, X_c, cell_id)
    phi_val = data["value"]
    grad_phi = data["grad"]
    hess_phi = None
    Nphi_row = data["N_row"]
    Bphi_mat = data["B_mat"]
    return phi_val, grad_phi, hess_phi, Nphi_row, Bphi_mat

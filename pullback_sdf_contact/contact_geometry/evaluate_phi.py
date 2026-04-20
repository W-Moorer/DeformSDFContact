import time
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


_CELL_GEOMETRY_CACHE = {}
_FUNCTION_CELL_DOF_CACHE = {}
_VECTOR_SUBFUNCTION_CACHE = {}
_GEOMETRY_EVAL_PROFILE = {
    "cell_geometry_call_count": 0,
    "cell_geometry_cache_hit_count": 0,
    "cell_geometry_cache_miss_count": 0,
    "function_cell_dof_call_count": 0,
    "function_cell_dof_cache_hit_count": 0,
    "function_cell_dof_cache_miss_count": 0,
    "vector_subfunction_call_count": 0,
    "vector_subfunction_cache_hit_count": 0,
    "vector_subfunction_cache_miss_count": 0,
    "scalar_eval_call_count": 0,
    "vector_eval_call_count": 0,
    "phi_eval_call_count": 0,
}


def _profile_add(profile, key, dt):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(dt)


def _profile_inc(profile, key, value=1):
    if profile is not None:
        profile[key] = profile.get(key, 0) + int(value)


def reset_geometry_eval_profile():
    for key in _GEOMETRY_EVAL_PROFILE:
        _GEOMETRY_EVAL_PROFILE[key] = 0


def snapshot_geometry_eval_profile():
    return dict(_GEOMETRY_EVAL_PROFILE)


def _cell_geometry_data(domain, cell_id, profile=None):
    _GEOMETRY_EVAL_PROFILE["cell_geometry_call_count"] += 1
    key = (id(domain), int(cell_id))
    data = _CELL_GEOMETRY_CACHE.get(key)
    if data is None:
        _GEOMETRY_EVAL_PROFILE["cell_geometry_cache_miss_count"] += 1
        t0 = time.perf_counter()
        geom_dofs = domain.geometry.dofmap.links(cell_id)
        x_g = domain.geometry.x[geom_dofs]
        _profile_add(profile, "dofmap_lookup_time", time.perf_counter() - t0)
        mins = x_g.min(axis=0)
        maxs = x_g.max(axis=0)
        lengths = maxs - mins
        jac_inv = np.diag(1.0 / lengths)
        data = {
            "mins": mins,
            "maxs": maxs,
            "lengths": lengths,
            "jac_inv": jac_inv,
        }
        _CELL_GEOMETRY_CACHE[key] = data
    else:
        _GEOMETRY_EVAL_PROFILE["cell_geometry_cache_hit_count"] += 1
    return data


def _function_cell_dofs(function, cell_id, profile=None):
    _GEOMETRY_EVAL_PROFILE["function_cell_dof_call_count"] += 1
    key = (id(function.function_space), int(cell_id))
    cell_dofs = _FUNCTION_CELL_DOF_CACHE.get(key)
    if cell_dofs is None:
        _GEOMETRY_EVAL_PROFILE["function_cell_dof_cache_miss_count"] += 1
        t0 = time.perf_counter()
        cell_dofs = np.asarray(function.function_space.dofmap.list.links(cell_id), dtype=np.int32)
        _profile_add(profile, "dofmap_lookup_time", time.perf_counter() - t0)
        _FUNCTION_CELL_DOF_CACHE[key] = cell_dofs
    else:
        _GEOMETRY_EVAL_PROFILE["function_cell_dof_cache_hit_count"] += 1
    return cell_dofs


def _vector_subfunctions(function):
    _GEOMETRY_EVAL_PROFILE["vector_subfunction_call_count"] += 1
    key = id(function)
    subfunctions = _VECTOR_SUBFUNCTION_CACHE.get(key)
    if subfunctions is None:
        _GEOMETRY_EVAL_PROFILE["vector_subfunction_cache_miss_count"] += 1
        gdim = function.function_space.mesh.geometry.dim
        subfunctions = [function.sub(i) for i in range(gdim)]
        _VECTOR_SUBFUNCTION_CACHE[key] = subfunctions
    else:
        _GEOMETRY_EVAL_PROFILE["vector_subfunction_cache_hit_count"] += 1
    return subfunctions


def cell_bounds(domain, cell_id):
    data = _cell_geometry_data(domain, cell_id)
    return data["mins"], data["maxs"]


def reference_coordinates(domain, cell_id, X, profile=None):
    data = _cell_geometry_data(domain, cell_id, profile=profile)
    return (np.asarray(X, dtype=np.float64) - data["mins"]) / data["lengths"]


def _scalar_eval_basis_data(function, X_c, cell_id, profile=None):
    V = function.function_space
    domain = V.mesh
    cell_name = domain.ufl_cell().cellname()
    degree = V.ufl_element().degree()
    xi = reference_coordinates(domain, cell_id, X_c, profile=profile)
    element = _scalar_basix_element(cell_name, degree)
    tab_t0 = time.perf_counter()
    tab = element.tabulate(2, np.asarray([xi], dtype=np.float64))
    _profile_add(profile, "basis_eval_or_tabulation_time", time.perf_counter() - tab_t0)

    basis = tab[0, 0, :]
    grad_ref = np.vstack([tab[1, 0, :], tab[2, 0, :], tab[3, 0, :]])
    hess_ref = np.array(
        [
            [tab[basix.index(2, 0, 0), 0, :], tab[basix.index(1, 1, 0), 0, :], tab[basix.index(1, 0, 1), 0, :]],
            [tab[basix.index(1, 1, 0), 0, :], tab[basix.index(0, 2, 0), 0, :], tab[basix.index(0, 1, 1), 0, :]],
            [tab[basix.index(1, 0, 1), 0, :], tab[basix.index(0, 1, 1), 0, :], tab[basix.index(0, 0, 2), 0, :]],
        ],
        dtype=np.float64,
    )
    geom = _cell_geometry_data(domain, cell_id, profile=profile)
    grad_phys = geom["jac_inv"].T @ grad_ref
    return {
        "domain": domain,
        "xi": xi,
        "basis": basis,
        "grad_phys": grad_phys,
        "hess_ref": hess_ref,
        "jac_inv": geom["jac_inv"],
    }


def eval_scalar_function_data(function, X_c, cell_id, profile=None, globalize=True):
    _GEOMETRY_EVAL_PROFILE["scalar_eval_call_count"] += 1
    basis_data = _scalar_eval_basis_data(function, X_c, cell_id, profile=profile)
    domain = basis_data["domain"]
    xi = basis_data["xi"]
    basis = basis_data["basis"]
    grad_phys = basis_data["grad_phys"]
    hess_ref = basis_data["hess_ref"]
    jac_inv = basis_data["jac_inv"]

    cell_dofs = _function_cell_dofs(function, cell_id, profile=profile)
    coeffs = function.vector.array_r[cell_dofs]
    value = float(basis @ coeffs)
    grad = grad_phys @ coeffs
    hess = np.tensordot(hess_ref, coeffs, axes=(2, 0))
    hess = jac_inv.T @ hess @ jac_inv

    out = {
        "value": value,
        "grad": grad,
        "hess": hess,
        "cell_dofs": np.asarray(cell_dofs, dtype=np.int32),
        "basis_local": basis,
        "grad_local": grad_phys,
        "reference_point": xi,
    }

    if globalize:
        alloc_t0 = time.perf_counter()
        ndofs = function.vector.getLocalSize()
        N_row = np.zeros(ndofs, dtype=np.float64)
        B_mat = np.zeros((domain.geometry.dim, ndofs), dtype=np.float64)
        _profile_add(profile, "numpy_temp_allocation_time", time.perf_counter() - alloc_t0)
        N_row[cell_dofs] = basis
        B_mat[:, cell_dofs] = grad_phys
        out["N_row"] = N_row
        out["B_mat"] = B_mat

    return out


def eval_vector_function_data(function, X_c, cell_id, profile=None, globalize=True):
    _GEOMETRY_EVAL_PROFILE["vector_eval_call_count"] += 1
    gdim = function.function_space.mesh.geometry.dim
    alloc_t0 = time.perf_counter()
    value = np.zeros(gdim, dtype=np.float64)
    grad = np.zeros((gdim, gdim), dtype=np.float64)
    _profile_add(profile, "numpy_temp_allocation_time", time.perf_counter() - alloc_t0)

    subfunctions = _vector_subfunctions(function)
    basis_data = _scalar_eval_basis_data(subfunctions[0], X_c, cell_id, profile=profile)
    basis = basis_data["basis"]
    grad_phys = basis_data["grad_phys"]
    blocked_cell_dofs = _function_cell_dofs(function, cell_id, profile=profile)
    expanded_cell_dofs = np.repeat(blocked_cell_dofs, gdim) * gdim + np.tile(
        np.arange(gdim, dtype=np.int32),
        len(blocked_cell_dofs),
    )

    alloc_t0 = time.perf_counter()
    N_local = np.zeros((gdim, len(expanded_cell_dofs)), dtype=np.float64)
    B_tensor_local = np.zeros((gdim, gdim, len(expanded_cell_dofs)), dtype=np.float64)
    _profile_add(profile, "numpy_temp_allocation_time", time.perf_counter() - alloc_t0)

    N_mat = None
    B_tensor = None
    if globalize:
        ndofs = function.vector.getLocalSize()
        alloc_t0 = time.perf_counter()
        N_mat = np.zeros((gdim, ndofs), dtype=np.float64)
        B_tensor = np.zeros((gdim, gdim, ndofs), dtype=np.float64)
        _profile_add(profile, "numpy_temp_allocation_time", time.perf_counter() - alloc_t0)

    for i, subfunction in enumerate(subfunctions):
        cell_dofs = _function_cell_dofs(subfunction, cell_id, profile=profile)
        coeffs = subfunction.vector.array_r[cell_dofs]
        value[i] = float(basis @ coeffs)
        grad[i, :] = grad_phys @ coeffs
        N_local[i, i::gdim] = basis
        B_tensor_local[i][:, i::gdim] = grad_phys
        if globalize:
            N_mat[i, cell_dofs] = basis
            B_tensor[i][:, cell_dofs] = grad_phys

    out = {
        "value": value,
        "grad": grad,
        "N_local": N_local,
        "B_tensor_local": B_tensor_local,
        "cell_dofs": np.asarray(expanded_cell_dofs, dtype=np.int32),
    }
    if globalize:
        out["N_mat"] = N_mat
        out["B_tensor"] = B_tensor
    return out


def eval_phi_quantities(X_c, cell_id, phi_function, profile=None, globalize=True):
    """
    Returns:
        phi_val
        grad_phi
        hess_phi
        Nphi_row
        Bphi_mat
    """
    _GEOMETRY_EVAL_PROFILE["phi_eval_call_count"] += 1
    data = eval_scalar_function_data(phi_function, X_c, cell_id, profile=profile, globalize=globalize)
    phi_val = data["value"]
    grad_phi = data["grad"]
    hess_phi = data["hess"]
    Nphi_row = data.get("N_row")
    Bphi_mat = data.get("B_mat")
    return phi_val, grad_phi, hess_phi, Nphi_row, Bphi_mat, data

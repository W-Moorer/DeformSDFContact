import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np

from check_indenter_block_contact import build_indenter_state
from coupled_solver.monolithic import (
    _apply_step_data,
    _create_petsc_vec_from_array,
    _solve_petsc_linear_system,
    assemble_monolithic_contact_system,
)


def main():
    state, solver_cfg, _ = build_indenter_state("baseline")
    step_data = {"step": 2, "load_value": 0.02}
    _apply_step_data(state, step_data)

    dense = assemble_monolithic_contact_system(state, solver_cfg, backend="dense", need_jacobian=True)
    sparse = assemble_monolithic_contact_system(state, solver_cfg, backend="petsc_block", need_jacobian=True)

    dense_residual = dense["global_residual_array"]
    sparse_residual = sparse["global_residual_vec"].getArray().copy()
    residual_diff = float(np.linalg.norm(dense_residual - sparse_residual))

    J_uu_diff = float(np.linalg.norm(dense["J_uu"] - sparse["J_uu"]))
    J_uphi_diff = float(np.linalg.norm(dense["J_uphi"] - sparse["J_uphi"]))
    J_phiu_diff = float(np.linalg.norm(dense["J_phiu"] - sparse["J_phiu"]))
    J_phiphi_diff = float(np.linalg.norm(dense["J_phiphi"] - sparse["J_phiphi"]))
    rows = np.arange(sparse["global_jacobian_dense"].shape[0], dtype=np.int32)
    sparse_global = np.asarray(
        sparse["global_jacobian_mat"].getValues(rows, rows), dtype=np.float64
    )
    global_J_diff = float(np.linalg.norm(dense["global_jacobian_dense"] - sparse_global))

    dense_delta = np.linalg.solve(dense["global_jacobian_dense"], -dense_residual)
    rhs = _create_petsc_vec_from_array(-sparse_residual, state["domain"].mpi_comm())
    sparse_delta, _ = _solve_petsc_linear_system(sparse["global_jacobian_mat"], rhs)
    delta_diff = float(np.linalg.norm(dense_delta - sparse_delta))

    print("Monolithic sparse equivalence check")
    print("")
    print("state = baseline load 0.02, zero-start assembled state")
    print(f"||R_dense - R_sparse||_2 = {residual_diff}")
    print(f"||J_uu_dense - J_uu_sparse||_F = {J_uu_diff}")
    print(f"||J_uphi_dense - J_uphi_sparse||_F = {J_uphi_diff}")
    print(f"||J_phiu_dense - J_phiu_sparse||_F = {J_phiu_diff}")
    print(f"||J_phiphi_dense - J_phiphi_sparse||_F = {J_phiphi_diff}")
    print(f"||J_global_dense - J_global_sparse||_F = {global_J_diff}")
    print(f"||delta_dense - delta_sparse||_2 = {delta_diff}")


if __name__ == "__main__":
    main()

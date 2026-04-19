import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import build_indenter_state, build_load_schedule
from coupled_solver.monolithic import (
    recommended_monolithic_contact_options,
    solve_monolithic_contact_loadpath,
)
from post.xdmf import write_scalar_field


def main():
    mode = "aggressive"
    target_loads = [0.04, 0.08, 0.12]
    recommended = recommended_monolithic_contact_options()

    for target_load in target_loads:
        state, solver_cfg, mode_cfg = build_indenter_state(mode)
        full_schedule = build_load_schedule(mode)
        load_schedule = [item for item in full_schedule if float(item["load_value"]) <= target_load + 1e-12]

        final_state, result = solve_monolithic_contact_loadpath(
            state,
            solver_cfg,
            load_schedule,
            backend=recommended["backend"],
            linear_solver_mode="lu",
            ksp_type="preonly",
            pc_type="lu",
            block_pc_name="global_lu",
            ksp_rtol=recommended["ksp_rtol"],
            ksp_atol=recommended["ksp_atol"],
            ksp_max_it=recommended["ksp_max_it"],
            max_newton_iter=recommended["max_newton_iter"],
            max_cutbacks=mode_cfg["max_cutbacks"],
            tol_res=recommended["tol_res"],
            tol_inc=recommended["tol_inc"],
            line_search=recommended["line_search"],
            initial_damping=recommended["initial_damping"],
            max_backtracks=recommended["max_backtracks"],
            backtrack_factor=recommended["backtrack_factor"],
            write_outputs=False,
            history_path=None,
            verbose=False,
            min_cutback_increment=mode_cfg["min_cutback_increment"],
        )

        final_load = float(result["final_accepted_load"])
        if abs(final_load - target_load) > 1e-12:
            raise RuntimeError(
                f"snapshot solve did not reach target load {target_load:.4f}, got {final_load:.4f}"
            )

        tag = f"{int(round(target_load * 1000.0)):03d}"
        write_scalar_field(
            final_state["domain"],
            final_state["u"],
            f"output_u_monolithic_aggressive_snapshot_load{tag}.xdmf",
        )
        write_scalar_field(
            final_state["domain"],
            final_state["phi"],
            f"output_phi_monolithic_aggressive_snapshot_load{tag}.xdmf",
        )
        print(
            f"wrote snapshot for load={target_load:.4f} "
            f"to output_u_monolithic_aggressive_snapshot_load{tag}.xdmf"
        )


if __name__ == "__main__":
    main()

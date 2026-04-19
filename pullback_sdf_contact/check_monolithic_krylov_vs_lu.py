import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import recommended_monolithic_contact_options


def print_comparison(mode, lu_summary, krylov_summary):
    print(f"mode = {mode}")
    print(
        "  solver | converged | final_accepted | total_newton_iterations | total_linear_iterations | "
        "cutbacks | final_residual_norm | final_reaction_norm | final_max_penetration | ||u||_2 | ||phi-phi0||_2"
    )
    print(
        "  lu     | {conv} | {accepted:.6f} | {newton} | {linear} | {cutbacks} | "
        "{residual:.6e} | {reaction:.6e} | {penetration:.6e} | {u_norm:.6e} | {phi_norm:.6e}".format(
            conv=lu_summary["reached_final_target"],
            accepted=lu_summary["final_accepted_load"],
            newton=lu_summary["total_newton_iterations_accepted"],
            linear=lu_summary["total_linear_iterations_accepted"],
            cutbacks=lu_summary["cutback_count"],
            residual=lu_summary["final_residual_norm"],
            reaction=lu_summary["final_reaction_norm"],
            penetration=lu_summary["final_max_penetration"],
            u_norm=lu_summary["u_norm"],
            phi_norm=lu_summary["phi_delta_norm"],
        )
    )
    print(
        "  krylov | {conv} | {accepted:.6f} | {newton} | {linear} | {cutbacks} | "
        "{residual:.6e} | {reaction:.6e} | {penetration:.6e} | {u_norm:.6e} | {phi_norm:.6e}".format(
            conv=krylov_summary["reached_final_target"],
            accepted=krylov_summary["final_accepted_load"],
            newton=krylov_summary["total_newton_iterations_accepted"],
            linear=krylov_summary["total_linear_iterations_accepted"],
            cutbacks=krylov_summary["cutback_count"],
            residual=krylov_summary["final_residual_norm"],
            reaction=krylov_summary["final_reaction_norm"],
            penetration=krylov_summary["final_max_penetration"],
            u_norm=krylov_summary["u_norm"],
            phi_norm=krylov_summary["phi_delta_norm"],
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]

    print("Monolithic Krylov vs LU")
    print("")
    print("backend = petsc_block")
    print(f"reference solver = lu")
    print(
        "krylov solver = "
        f"{recommended['ksp_type']} + {recommended['pc_type']} ({recommended['block_pc_name']})"
    )
    print("")

    for mode in modes:
        _, _, lu_summary = run_monolithic_case(
            mode,
            backend="petsc_block",
            linear_solver_mode="lu",
            ksp_type="preonly",
            pc_type="lu",
            block_pc_name="global_lu",
            ksp_rtol=recommended["ksp_rtol"],
            ksp_atol=recommended["ksp_atol"],
            ksp_max_it=recommended["ksp_max_it"],
            max_newton_iter=recommended["max_newton_iter"],
            line_search=recommended["line_search"],
            damping=recommended["initial_damping"],
            write_outputs=False,
            verbose=False,
        )
        _, _, krylov_summary = run_monolithic_case(
            mode,
            backend="petsc_block",
            linear_solver_mode="krylov",
            ksp_type=recommended["ksp_type"],
            pc_type=recommended["pc_type"],
            block_pc_name=recommended["block_pc_name"],
            ksp_rtol=recommended["ksp_rtol"],
            ksp_atol=recommended["ksp_atol"],
            ksp_max_it=recommended["ksp_max_it"],
            max_newton_iter=recommended["max_newton_iter"],
            line_search=recommended["line_search"],
            damping=recommended["initial_damping"],
            write_outputs=False,
            verbose=False,
        )
        print_comparison(mode, lu_summary, krylov_summary)
        print("")


if __name__ == "__main__":
    main()

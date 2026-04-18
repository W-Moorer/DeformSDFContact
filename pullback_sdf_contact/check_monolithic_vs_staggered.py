import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import run_case as run_staggered_case
from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import recommended_monolithic_contact_options
from coupled_solver.staggered import recommended_staggered_contact_options


def print_comparison(mode, staggered_summary, dense_summary, sparse_summary):
    print(f"mode = {mode}")
    print(
        "  solver           | converged | requested_target | final_accepted | total_iterations | "
        "cutbacks | final_reaction_norm | final_max_penetration | ||u||_2 | ||phi-phi0||_2"
    )
    print(
        "  staggered   | {conv} | {target:.6f} | {accepted:.6f} | {iters} | {cutbacks} | "
        "{reaction:.6e} | {penetration:.6e} | {u_norm:.6e} | {phi_norm:.6e}".format(
            conv=staggered_summary["reached_final_target"],
            target=staggered_summary["requested_final_target_load"],
            accepted=staggered_summary["final_accepted_load"],
            iters=staggered_summary["total_outer_iterations_accepted"],
            cutbacks=staggered_summary["cutback_count"],
            reaction=staggered_summary["final_reaction_norm"],
            penetration=staggered_summary["final_max_penetration"],
            u_norm=staggered_summary["u_norm"],
            phi_norm=staggered_summary["phi_delta_norm"],
        )
    )
    print(
        "  mono-dense       | {conv} | {target:.6f} | {accepted:.6f} | {iters} | {cutbacks} | "
        "{reaction:.6e} | {penetration:.6e} | {u_norm:.6e} | {phi_norm:.6e}".format(
            conv=dense_summary["reached_final_target"],
            target=dense_summary["requested_final_target_load"],
            accepted=dense_summary["final_accepted_load"],
            iters=dense_summary["total_newton_iterations_accepted"],
            cutbacks=dense_summary["cutback_count"],
            reaction=dense_summary["final_reaction_norm"],
            penetration=dense_summary["final_max_penetration"],
            u_norm=dense_summary["u_norm"],
            phi_norm=dense_summary["phi_delta_norm"],
        )
    )
    print(
        "  mono-petsc       | {conv} | {target:.6f} | {accepted:.6f} | {iters} | {cutbacks} | "
        "{reaction:.6e} | {penetration:.6e} | {u_norm:.6e} | {phi_norm:.6e}".format(
            conv=sparse_summary["reached_final_target"],
            target=sparse_summary["requested_final_target_load"],
            accepted=sparse_summary["final_accepted_load"],
            iters=sparse_summary["total_newton_iterations_accepted"],
            cutbacks=sparse_summary["cutback_count"],
            reaction=sparse_summary["final_reaction_norm"],
            penetration=sparse_summary["final_max_penetration"],
            u_norm=sparse_summary["u_norm"],
            phi_norm=sparse_summary["phi_delta_norm"],
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    parser.add_argument("--max-newton-iter", type=int, default=None)
    parser.add_argument("--line-search", choices=["on", "off"], default=None)
    parser.add_argument("--damping", type=float, default=None)
    args = parser.parse_args()

    recommended = recommended_staggered_contact_options()
    recommended_mono = recommended_monolithic_contact_options()
    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]
    resolved_max_newton_iter = (
        recommended_mono["max_newton_iter"] if args.max_newton_iter is None else args.max_newton_iter
    )
    resolved_line_search = (
        recommended_mono["line_search"] if args.line_search is None else args.line_search == "on"
    )
    resolved_damping = (
        recommended_mono["initial_damping"] if args.damping is None else args.damping
    )

    print("Monolithic vs staggered comparison")
    print("")
    print(
        f"staggered recommended path = {recommended['contact_structure_mode']}, "
        f"max_outer_iter = {recommended['max_outer_iter']}, relaxation_u = {recommended['relaxation_u']}"
    )
    print(
        f"monolithic max_newton_iter = {resolved_max_newton_iter}, "
        f"line_search = {resolved_line_search}, damping = {resolved_damping}"
    )
    print("")

    for mode in modes:
        staggered_state, _, staggered_summary, _, _ = run_staggered_case(mode)
        monolithic_state, _, monolithic_summary = run_monolithic_case(
            mode,
            backend="dense",
            max_newton_iter=resolved_max_newton_iter,
            line_search=resolved_line_search,
            damping=resolved_damping,
            write_outputs=False,
            verbose=False,
        )
        sparse_state, _, sparse_summary = run_monolithic_case(
            mode,
            backend="petsc_block",
            max_newton_iter=resolved_max_newton_iter,
            line_search=resolved_line_search,
            damping=resolved_damping,
            write_outputs=False,
            verbose=False,
        )
        staggered_summary["u_norm"] = float((staggered_state["u"].vector.array_r**2).sum() ** 0.5)
        staggered_summary["phi_delta_norm"] = float(
            ((staggered_state["phi"].vector.array_r - staggered_state["phi0"].vector.array_r) ** 2).sum() ** 0.5
        )
        print_comparison(mode, staggered_summary, monolithic_summary, sparse_summary)
        print("")


if __name__ == "__main__":
    main()

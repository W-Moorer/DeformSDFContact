import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import run_case as run_staggered_case
from check_monolithic_contact import run_monolithic_case
from coupled_solver.staggered import recommended_staggered_contact_options


def print_comparison(mode, staggered_summary, monolithic_summary):
    print(f"mode = {mode}")
    print(
        "  solver      | converged | requested_target | final_accepted | total_iterations | "
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
        "  monolithic  | {conv} | {target:.6f} | {accepted:.6f} | {iters} | {cutbacks} | "
        "{reaction:.6e} | {penetration:.6e} | {u_norm:.6e} | {phi_norm:.6e}".format(
            conv=monolithic_summary["reached_final_target"],
            target=monolithic_summary["requested_final_target_load"],
            accepted=monolithic_summary["final_accepted_load"],
            iters=monolithic_summary["total_newton_iterations_accepted"],
            cutbacks=monolithic_summary["cutback_count"],
            reaction=monolithic_summary["final_reaction_norm"],
            penetration=monolithic_summary["final_max_penetration"],
            u_norm=monolithic_summary["u_norm"],
            phi_norm=monolithic_summary["phi_delta_norm"],
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    parser.add_argument("--max-newton-iter", type=int, default=15)
    args = parser.parse_args()

    recommended = recommended_staggered_contact_options()
    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]

    print("Monolithic vs staggered comparison")
    print("")
    print(
        f"staggered recommended path = {recommended['contact_structure_mode']}, "
        f"max_outer_iter = {recommended['max_outer_iter']}, relaxation_u = {recommended['relaxation_u']}"
    )
    print(f"monolithic max_newton_iter = {args.max_newton_iter}")
    print("")

    for mode in modes:
        staggered_state, _, staggered_summary, _, _ = run_staggered_case(mode)
        monolithic_state, _, monolithic_summary = run_monolithic_case(
            mode,
            max_newton_iter=args.max_newton_iter,
            write_outputs=False,
        )
        staggered_summary["u_norm"] = float((staggered_state["u"].vector.array_r**2).sum() ** 0.5)
        staggered_summary["phi_delta_norm"] = float(
            ((staggered_state["phi"].vector.array_r - staggered_state["phi0"].vector.array_r) ** 2).sum() ** 0.5
        )
        print_comparison(mode, staggered_summary, monolithic_summary)
        print("")


if __name__ == "__main__":
    main()

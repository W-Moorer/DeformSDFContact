import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import print_case, run_monolithic_case
from coupled_solver.monolithic import recommended_monolithic_contact_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    parser.add_argument("--max-newton-iter", type=int, default=None)
    parser.add_argument("--line-search", choices=["on", "off"], default=None)
    parser.add_argument("--damping", type=float, default=None)
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    resolved_max_newton_iter = (
        recommended["max_newton_iter"] if args.max_newton_iter is None else args.max_newton_iter
    )
    resolved_line_search = (
        recommended["line_search"] if args.line_search is None else args.line_search == "on"
    )
    resolved_damping = (
        recommended["initial_damping"] if args.damping is None else args.damping
    )
    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]

    print("Sparse/PETSc monolithic contact regression")
    print("")
    print("backend = petsc_block")
    print(f"max_newton_iter = {resolved_max_newton_iter}")
    print(f"line_search = {resolved_line_search}")
    print(f"damping = {resolved_damping}")
    print("")
    for mode in modes:
        _, result, summary = run_monolithic_case(
            mode,
            backend="petsc_block",
            max_newton_iter=resolved_max_newton_iter,
            line_search=resolved_line_search,
            damping=resolved_damping,
            write_outputs=True,
            verbose=False,
        )
        print_case(result, summary)
        print("")


if __name__ == "__main__":
    main()

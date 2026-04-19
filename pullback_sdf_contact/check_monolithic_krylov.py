import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import print_case, run_monolithic_case
from coupled_solver.monolithic import (
    get_petsc_runtime_info,
    monolithic_block_pc_names,
    recommended_monolithic_contact_options,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="baseline")
    parser.add_argument("--mesh-scale", choices=["small", "larger", "xlarger"], default="small")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--linear-solver", choices=["lu", "krylov"], required=True)
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--pc-type", default=None)
    parser.add_argument("--block-pc-name", default=None)
    parser.add_argument("--max-newton-iter", type=int, default=None)
    parser.add_argument("--line-search", choices=["on", "off"], default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--ksp-rtol", type=float, default=None)
    parser.add_argument("--ksp-atol", type=float, default=None)
    parser.add_argument("--ksp-max-it", type=int, default=None)
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    runtime = get_petsc_runtime_info()
    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]
    resolved_max_newton_iter = (
        recommended["max_newton_iter"] if args.max_newton_iter is None else args.max_newton_iter
    )
    resolved_line_search = (
        recommended["line_search"] if args.line_search is None else args.line_search == "on"
    )
    resolved_damping = (
        recommended["initial_damping"] if args.damping is None else args.damping
    )
    resolved_ksp_type = recommended["ksp_type"] if args.ksp_type is None else args.ksp_type
    resolved_pc_type = recommended["pc_type"] if args.pc_type is None else args.pc_type
    resolved_block_pc_name = (
        recommended["block_pc_name"] if args.block_pc_name is None else args.block_pc_name
    )
    if args.linear_solver == "lu":
        resolved_ksp_type = "preonly"
        resolved_pc_type = "lu"
        resolved_block_pc_name = "global_lu"
    elif resolved_pc_type != "fieldsplit" and args.block_pc_name is None:
        resolved_block_pc_name = f"global_{resolved_pc_type}"
    resolved_ksp_rtol = recommended["ksp_rtol"] if args.ksp_rtol is None else args.ksp_rtol
    resolved_ksp_atol = recommended["ksp_atol"] if args.ksp_atol is None else args.ksp_atol
    resolved_ksp_max_it = (
        recommended["ksp_max_it"] if args.ksp_max_it is None else args.ksp_max_it
    )

    print("Monolithic Krylov/LU benchmark")
    print("")
    print(f"petsc_version = {runtime['petsc_version']}")
    print(f"available_block_pc_names = {monolithic_block_pc_names()}")
    print(f"mesh_scale = {args.mesh_scale}")
    if args.nx is not None or args.ny is not None or args.nz is not None:
        print(f"mesh_override = ({args.nx}, {args.ny}, {args.nz})")
    print("backend = petsc_block")
    print(f"linear_solver_mode = {args.linear_solver}")
    print(f"ksp_type = {resolved_ksp_type}")
    print(f"pc_type = {resolved_pc_type}")
    print(f"block_pc_name = {resolved_block_pc_name}")
    print(f"max_newton_iter = {resolved_max_newton_iter}")
    print(f"line_search = {resolved_line_search}")
    print(f"damping = {resolved_damping}")
    print("")

    for mode in modes:
        _, result, summary = run_monolithic_case(
            mode,
            mesh_scale=args.mesh_scale,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            backend="petsc_block",
            linear_solver_mode=args.linear_solver,
            ksp_type=resolved_ksp_type,
            pc_type=resolved_pc_type,
            block_pc_name=resolved_block_pc_name,
            ksp_rtol=resolved_ksp_rtol,
            ksp_atol=resolved_ksp_atol,
            ksp_max_it=resolved_ksp_max_it,
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

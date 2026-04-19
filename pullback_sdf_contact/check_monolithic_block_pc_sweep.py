import argparse
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import recommended_monolithic_contact_options


def build_cases(include_aggressive=False):
    cases = [
        ("small_baseline", "baseline", "small"),
        ("larger_baseline", "baseline", "larger"),
    ]
    if include_aggressive:
        cases.append(("small_aggressive", "aggressive", "small"))
    return cases


def build_solver_configs(recommended, *, include_additive=False):
    configs = [
        {
            "label": "reference_lu",
            "linear_solver_mode": "lu",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "block_pc_name": "global_lu",
        },
        {
            "label": "gmres_ilu",
            "linear_solver_mode": "krylov",
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "block_pc_name": "global_ilu",
        },
        {
            "label": "gmres_fieldsplit_multiplicative_ilu",
            "linear_solver_mode": "krylov",
            "ksp_type": "gmres",
            "pc_type": "fieldsplit",
            "block_pc_name": "fieldsplit_multiplicative_ilu",
        },
    ]
    if include_additive:
        configs.insert(
            2,
            {
                "label": "gmres_fieldsplit_additive_ilu",
                "linear_solver_mode": "krylov",
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "block_pc_name": "fieldsplit_additive_ilu",
            },
        )
    return configs


def print_case_header(case_label, mode, mesh_scale):
    print(f"case = {case_label}")
    print(f"  mode = {mode}")
    print(f"  mesh_scale = {mesh_scale}")
    print(
        "  solver | converged | final_accepted | total_newton_iterations | "
        "total_linear_iterations | avg_linear_per_newton | cutbacks | "
        "final_residual_norm | final_reaction_norm | final_max_penetration | ||u||_2"
    )


def print_row(label, summary):
    total_newton = summary["total_newton_iterations_accepted"]
    total_linear = summary["total_linear_iterations_accepted"]
    avg_linear = 0.0 if total_newton == 0 else total_linear / total_newton
    print(
        "  {label} | {conv} | {accepted:.6f} | {newton} | {linear} | {avg_linear:.3f} | "
        "{cutbacks} | {residual:.6e} | {reaction:.6e} | {penetration:.6e} | {u_norm:.6e}".format(
            label=label,
            conv=summary["reached_final_target"],
            accepted=summary["final_accepted_load"],
            newton=total_newton,
            linear=total_linear,
            avg_linear=avg_linear,
            cutbacks=summary["cutback_count"],
            residual=summary["final_residual_norm"],
            reaction=summary["final_reaction_norm"],
            penetration=summary["final_max_penetration"],
            u_norm=summary["u_norm"],
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-aggressive", action="store_true")
    parser.add_argument("--include-additive", action="store_true")
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    cases = build_cases(include_aggressive=args.include_aggressive)
    solver_configs = build_solver_configs(recommended, include_additive=args.include_additive)

    print("Monolithic block preconditioner sweep")
    print("")
    print(f"backend = {recommended['backend']}")
    print(f"max_newton_iter = {recommended['max_newton_iter']}")
    print(f"line_search = {recommended['line_search']}")
    print(f"damping = {recommended['initial_damping']}")
    print("")

    for case_label, mode, mesh_scale in cases:
        print_case_header(case_label, mode, mesh_scale)
        for cfg in solver_configs:
            _, _, summary = run_monolithic_case(
                mode,
                mesh_scale=mesh_scale,
                backend=recommended["backend"],
                linear_solver_mode=cfg["linear_solver_mode"],
                ksp_type=cfg["ksp_type"],
                pc_type=cfg["pc_type"],
                block_pc_name=cfg["block_pc_name"],
                ksp_rtol=recommended["ksp_rtol"],
                ksp_atol=recommended["ksp_atol"],
                ksp_max_it=recommended["ksp_max_it"],
                max_newton_iter=recommended["max_newton_iter"],
                line_search=recommended["line_search"],
                damping=recommended["initial_damping"],
                write_outputs=False,
                verbose=False,
            )
            print_row(cfg["label"], summary)
        print("")


if __name__ == "__main__":
    main()

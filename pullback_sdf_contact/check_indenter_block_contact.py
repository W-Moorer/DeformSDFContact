import argparse
import os
import time

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
import ufl
from dolfinx import fem

from config import MeshConfig, SDFConfig, SolidConfig, SolverConfig
from contact_geometry.slave_quadrature import build_slave_quadrature
from coupled_solver.staggered import (
    recommended_staggered_contact_options,
    solve_staggered_contact_loadpath,
)
from mesh import tags
from mesh.build_mesh import create_reference_box
from post.xdmf import write_scalar_field
from sdf_field.boundary import create_sdf_bcs
from sdf_field.forms import sdf_forms
from sdf_field.spaces import create_sdf_space
from solid.boundary import create_solid_bcs
from solid.forms import solid_forms
from solid.spaces import create_displacement_space


def get_mode_config(mode, mesh_scale="small", nx=None, ny=None, nz=None):
    if mesh_scale == "small":
        default_nx, default_ny, default_nz = 3, 3, 3
    elif mesh_scale == "larger":
        default_nx, default_ny, default_nz = 4, 4, 4
    else:
        raise ValueError(f"Unsupported mesh_scale: {mesh_scale}")
    if nx is not None:
        default_nx = int(nx)
    if ny is not None:
        default_ny = int(ny)
    if nz is not None:
        default_nz = int(nz)
    mesh_cfg = MeshConfig(nx=default_nx, ny=default_ny, nz=default_nz)

    if mode == "baseline":
        return {
            "mesh_cfg": mesh_cfg,
            "solid_cfg": SolidConfig(E=100.0, nu=0.3, body_force_z=0.0),
            "sdf_cfg": SDFConfig(beta=1e-6),
            "solver_cfg": SolverConfig(max_it=4, verbose=False),
            "penalty": 1.0,
            "load_values": [0.0, 0.02, 0.04, 0.06],
            "max_outer_iter": 4,
            "max_cutbacks": 6,
            "min_cutback_increment": 1e-8,
        }
    if mode == "aggressive":
        return {
            "mesh_cfg": mesh_cfg,
            "solid_cfg": SolidConfig(E=40.0, nu=0.3, body_force_z=0.0),
            "sdf_cfg": SDFConfig(beta=1e-6),
            "solver_cfg": SolverConfig(max_it=4, verbose=False),
            "penalty": 5.0,
            "load_values": [0.0, 0.04, 0.08, 0.12],
            "max_outer_iter": 4,
            "max_cutbacks": 3,
            "min_cutback_increment": 1e-6,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def mesh_resolution_string(mesh_cfg):
    return f"{mesh_cfg.nx}x{mesh_cfg.ny}x{mesh_cfg.nz}"


def build_indenter_state(mode, mesh_scale="small", nx=None, ny=None, nz=None):
    mode_cfg = get_mode_config(mode, mesh_scale=mesh_scale, nx=nx, ny=ny, nz=nz)
    mesh_cfg = mode_cfg["mesh_cfg"]
    solid_cfg = mode_cfg["solid_cfg"]
    sdf_cfg = mode_cfg["sdf_cfg"]
    solver_cfg = mode_cfg["solver_cfg"]
    penalty = mode_cfg["penalty"]

    domain, cell_tags, facet_tags = create_reference_box(mesh_cfg)
    V_u = create_displacement_space(domain, degree=1)
    V_phi = create_sdf_space(domain, degree=sdf_cfg.degree)

    u = fem.Function(V_u, name=f"u_contact_loadpath_{mode}")
    phi = fem.Function(V_phi, name=f"phi_contact_loadpath_{mode}")
    phi0 = fem.Function(V_phi, name=f"phi0_contact_loadpath_{mode}")

    phi0.interpolate(lambda x: x[2] - mesh_cfg.Lz)
    phi.interpolate(lambda x: x[2] - mesh_cfg.Lz)

    dx = ufl.Measure("dx", domain=domain)
    dx_band = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags, subdomain_id=tags.BAND
    )

    v_u = ufl.TestFunction(V_u)
    R_u_form, _ = solid_forms(u, v_u, solid_cfg, dx)
    solid_bcs = create_solid_bcs(V_u, facet_tags, top_uz=None)

    eta = ufl.TestFunction(V_phi)
    R_phi_form, K_phi_phi_form, K_phi_u_form = sdf_forms(
        u, phi, eta, phi0, sdf_cfg.beta, dx_band, dx_reg=dx
    )
    phi_bcs = create_sdf_bcs(V_phi, facet_tags, tags.TOP, phi_value=0.0)
    quadrature_points = build_slave_quadrature(domain, facet_tags, tags.TOP, quadrature_degree=2)

    state = {
        "domain": domain,
        "u": u,
        "phi": phi,
        "phi0": phi0,
        "R_u_form": R_u_form,
        "solid_bcs": solid_bcs,
        "R_phi_form": R_phi_form,
        "K_phi_phi_form": K_phi_phi_form,
        "K_phi_u_form": K_phi_u_form,
        "phi_bcs": phi_bcs,
        "quadrature_points": quadrature_points,
        "penalty": penalty,
        "slave_current_offset": np.zeros(3, dtype=np.float64),
        "current_load_value": 0.0,
        "mesh_resolution": mesh_resolution_string(mesh_cfg),
    }
    return state, solver_cfg, mode_cfg


def build_load_schedule(mode, mesh_scale="small", nx=None, ny=None, nz=None):
    mode_cfg = get_mode_config(mode, mesh_scale=mesh_scale, nx=nx, ny=ny, nz=nz)
    return [
        {"step": i + 1, "load_value": float(value), "label": f"{mode}_indent_{i + 1}"}
        for i, value in enumerate(mode_cfg["load_values"])
    ]


def get_suffix(mode, contact_structure_mode):
    return f"{mode}_{contact_structure_mode}"


def summarize_result(result, mode, contact_structure_mode, max_outer_iter, relaxation_u, elapsed_s):
    accepted_history = result["accepted_history"]
    attempt_history = result["attempt_history"]
    final_accepted = accepted_history[-1] if accepted_history else None
    return {
        "mode": mode,
        "contact_structure_mode": contact_structure_mode,
        "max_outer_iter": max_outer_iter,
        "relaxation_u": relaxation_u,
        "requested_final_target_load": result["requested_final_target_load"],
        "final_accepted_load": result["final_accepted_load"],
        "reached_final_target": result["reached_final_target"],
        "terminated_early": result["terminated_early"],
        "termination_reason": result["termination_reason"],
        "accepted_step_count": result["accepted_step_count"],
        "attempt_count": result["attempt_count"],
        "total_outer_iterations_accepted": int(
            sum(item["outer_iterations"] for item in accepted_history)
        ),
        "total_outer_iterations_attempts": int(
            sum(item["outer_iterations"] for item in attempt_history)
        ),
        "cutback_count": int(sum(1 for item in attempt_history if item["cutback_triggered"])),
        "final_reaction_norm": 0.0 if final_accepted is None else float(final_accepted["reaction_norm"]),
        "final_max_penetration": 0.0 if final_accepted is None else float(final_accepted["max_penetration"]),
        "elapsed_s": float(elapsed_s),
    }


def write_history_summary(result, summary, summary_path):
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(
            "Indenter Block Contact Summary "
            f"({summary['mode']}, mode={summary['contact_structure_mode']})\n\n"
        )
        for key in (
            "contact_structure_mode",
            "max_outer_iter",
            "relaxation_u",
            "requested_final_target_load",
            "final_accepted_load",
            "reached_final_target",
            "terminated_early",
            "termination_reason",
            "accepted_step_count",
            "attempt_count",
            "total_outer_iterations_accepted",
            "total_outer_iterations_attempts",
            "cutback_count",
            "final_reaction_norm",
            "final_max_penetration",
            "elapsed_s",
        ):
            handle.write(f"{key}={summary[key]}\n")
        handle.write("\nAccepted steps:\n")
        for item in result["accepted_history"]:
            handle.write(
                "  step={step}, load={load_value:.6f}, outer_iterations={outer_iterations}, "
                "active={active_contact_points}, negative_gap_sum={negative_gap_sum:.6e}, "
                "reaction_norm={reaction_norm:.6e}\n".format(**item)
            )
        handle.write("\nAttempts:\n")
        for item in result["attempt_history"]:
            handle.write(
                "  attempt={attempt_state_index}, step={step}, attempt_load={attempt_load:.6f}, "
                "accepted={accepted}, contact_structure_mode={contact_structure_mode}, "
                "cutback_level={cutback_level}, cutback_triggered={cutback_triggered}, "
                "converged={converged}, termination_reason={termination_reason}\n".format(**item)
            )


def run_case(mode, contact_structure_mode=None, max_outer_iter=None, relaxation_u=None):
    state, solver_cfg, mode_cfg = build_indenter_state(mode)
    load_schedule = build_load_schedule(mode)
    recommended = recommended_staggered_contact_options()
    contact_structure_mode = (
        recommended["contact_structure_mode"] if contact_structure_mode is None else contact_structure_mode
    )
    max_outer_iter = recommended["max_outer_iter"] if max_outer_iter is None else max_outer_iter
    relaxation_u = recommended["relaxation_u"] if relaxation_u is None else relaxation_u
    suffix = get_suffix(mode, contact_structure_mode)
    history_path = f"contact_history_{suffix}.csv"
    summary_path = f"contact_history_summary_{suffix}.txt"

    tic = time.perf_counter()
    final_state, result = solve_staggered_contact_loadpath(
        state,
        solver_cfg,
        load_schedule,
        contact_structure_mode=contact_structure_mode,
        max_outer_iter=max_outer_iter,
        max_cutbacks=mode_cfg["max_cutbacks"],
        min_cutback_increment=mode_cfg["min_cutback_increment"],
        relaxation_u=relaxation_u,
        relaxation_phi=recommended["relaxation_phi"],
        tol_du=recommended["tol_du"],
        tol_dphi=recommended["tol_dphi"],
        tol_contact_rhs=recommended["tol_contact_rhs"],
        write_outputs=True,
        history_path=history_path,
        verbose=True,
        step_verbose=False,
    )
    elapsed_s = time.perf_counter() - tic
    summary = summarize_result(
        result,
        mode,
        contact_structure_mode,
        max_outer_iter,
        relaxation_u,
        elapsed_s,
    )

    if final_state["domain"].mpi_comm().rank == 0:
        write_history_summary(result, summary, summary_path)

    write_scalar_field(final_state["domain"], final_state["u"], f"output_u_contact_loadpath_{suffix}.xdmf")
    write_scalar_field(final_state["domain"], final_state["phi"], f"output_phi_contact_loadpath_{suffix}.xdmf")

    return final_state, result, summary, history_path, summary_path


def print_attempt_history(result):
    print("Attempt-level summary:")
    for item in result["attempt_history"]:
        print(
            "  attempt={attempt_state_index:02d} step={step:02d} target={target_load:.4f} "
            "attempt={attempt_load:.4f} accepted={accepted} converged={converged} "
            "cutback_level={cutback_level} output_index={output_index}".format(**item)
        )


def print_accepted_history(result):
    print("Accepted-step summary:")
    for item in result["accepted_history"]:
        print(
            "  accepted_state={accepted_state_index:02d} step={step:02d} load={load_value:.4f} "
            "outer_iterations={outer_iterations} active={active_contact_points} "
            "negative_gap_sum={negative_gap_sum:.6e} reaction_norm={reaction_norm:.6e}".format(
                **item
            )
        )


def print_complete_result_table(summaries):
    print("Complete result table:")
    print(
        "  case | contact_structure_mode | max_outer_iter | relaxation_u | "
        "requested_final_target_load | final_accepted_load | reached_final_target | "
        "terminated_early | termination_reason | accepted_step_count | attempt_count | "
        "total_outer_iterations_accepted | total_outer_iterations_attempts | cutback_count | "
        "final_reaction_norm | final_max_penetration"
    )
    for summary in summaries:
        case = summary["mode"]
        print(
            f"  {case} | {summary['contact_structure_mode']} | {summary['max_outer_iter']} | "
            f"{summary['relaxation_u']:.1f} | {summary['requested_final_target_load']:.6f} | "
            f"{summary['final_accepted_load']:.6f} | "
            f"{summary['reached_final_target']} | "
            f"{summary['terminated_early']} | {summary['termination_reason']} | "
            f"{summary['accepted_step_count']} | {summary['attempt_count']} | "
            f"{summary['total_outer_iterations_accepted']} | "
            f"{summary['total_outer_iterations_attempts']} | {summary['cutback_count']} | "
            f"{summary['final_reaction_norm']:.6e} | {summary['final_max_penetration']:.6e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive", "all"], default="all")
    parser.add_argument("--contact-structure-mode", default=None)
    parser.add_argument("--max-outer-iter", type=int, default=None)
    parser.add_argument("--relaxation-u", type=float, default=None)
    args = parser.parse_args()

    recommended = recommended_staggered_contact_options()
    modes = ["baseline", "aggressive"] if args.mode == "all" else [args.mode]
    runs = []
    for mode in modes:
        _, result, summary, history_path, summary_path = run_case(
            mode,
            contact_structure_mode=args.contact_structure_mode,
            max_outer_iter=args.max_outer_iter,
            relaxation_u=args.relaxation_u,
        )
        runs.append((result, summary, history_path, summary_path))

    print("Rigid flat indenter vs elastic block")
    print("")
    print("Recommended solver path:")
    print(
        f"  contact_structure_mode = {args.contact_structure_mode or recommended['contact_structure_mode']}"
    )
    print(f"  max_outer_iter = {args.max_outer_iter or recommended['max_outer_iter']}")
    print(f"  relaxation_u = {args.relaxation_u or recommended['relaxation_u']}")
    print("")
    for result, summary, history_path, summary_path in runs:
        print(f"Mode: {summary['mode']}")
        print(f"  requested final target load = {summary['requested_final_target_load']}")
        print(f"  final accepted load = {summary['final_accepted_load']}")
        print(f"  reached_final_target = {summary['reached_final_target']}")
        print(f"  terminated_early = {summary['terminated_early']}")
        print(f"  termination_reason = {summary['termination_reason']}")
        print(f"  accepted step count = {summary['accepted_step_count']}")
        print(f"  attempt count = {summary['attempt_count']}")
        print(f"  history_path = {history_path}")
        print(f"  summary_path = {summary_path}")
        print_accepted_history(result)
        print("")

    summaries = [item[1] for item in runs]
    print_complete_result_table(summaries)


if __name__ == "__main__":
    main()

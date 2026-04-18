import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

import numpy as np
from dolfinx import fem
from petsc4py import PETSc

from check_indenter_block_contact import build_indenter_state, build_load_schedule
from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from coupled_solver.staggered import _restore_state_fields, solve_staggered_contact_loadpath, solve_staggered_contact_step
from solid.solve import (
    assemble_linear_solid_system,
    dense_array_from_petsc_mat,
    diagnose_dense_operator,
    mask_contact_tangent_for_bcs,
    solve_linear_solid_with_contact,
)


def accepted_snapshot_at_load(result, load_value, tol=1e-12):
    for row, snapshot in zip(result["accepted_history"], result["accepted_snapshots"]):
        if abs(float(row["load_value"]) - float(load_value)) < tol:
            return row, snapshot
    raise ValueError(f"Could not find accepted snapshot at load {load_value}")


def clone_state_with_snapshot(mode, snapshot):
    state, solver_cfg, mode_cfg = build_indenter_state(mode)
    _restore_state_fields(state, snapshot)
    return state, solver_cfg, mode_cfg


def copy_function(function):
    out = fem.Function(function.function_space, name=function.name)
    out.vector.array_w[:] = function.vector.array_r
    out.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return out


def evaluate_state_metrics(state, need_tangent=True):
    out = assemble_contact_contributions_surface(
        state["quadrature_points"],
        state,
        state["penalty"],
        need_residual=True,
        need_tangent_uu=need_tangent,
        need_diagnostics=True,
    )
    return out


def run_fixed_state_structure_test(snapshot, label, use_tangent):
    state, _, _ = clone_state_with_snapshot("aggressive", snapshot)
    before = evaluate_state_metrics(state, need_tangent=True)
    u_before = state["u"].vector.array_r.copy()
    if use_tangent:
        solve_info = solve_linear_solid_with_contact(
            state["u"],
            state["R_u_form"],
            state["solid_bcs"],
            contact_rhs=before["R_u_c"],
            contact_tangent_uu=before["K_uu_c"],
            verbose=False,
        )
    else:
        solve_info = solve_linear_solid_with_contact(
            state["u"],
            state["R_u_form"],
            state["solid_bcs"],
            contact_rhs=before["R_u_c"],
            contact_tangent_uu=None,
            verbose=False,
        )
    du_norm = float(np.linalg.norm(state["u"].vector.array_r - u_before))
    after = evaluate_state_metrics(state, need_tangent=True)

    u_mid = state["u"].vector.array_r.copy()
    if use_tangent:
        solve_linear_solid_with_contact(
            state["u"],
            state["R_u_form"],
            state["solid_bcs"],
            contact_rhs=after["R_u_c"],
            contact_tangent_uu=after["K_uu_c"],
            verbose=False,
        )
    else:
        solve_linear_solid_with_contact(
            state["u"],
            state["R_u_form"],
            state["solid_bcs"],
            contact_rhs=after["R_u_c"],
            contact_tangent_uu=None,
            verbose=False,
        )
    followup_du_norm = float(np.linalg.norm(state["u"].vector.array_r - u_mid))
    return {
        "label": label,
        "use_tangent": use_tangent,
        "solve_info": solve_info,
        "du_norm": du_norm,
        "before_contact_rhs_norm": before["diagnostics"]["reaction_norm"],
        "after_contact_rhs_norm": after["diagnostics"]["reaction_norm"],
        "before_negative_gap_sum": before["diagnostics"]["negative_gap_sum"],
        "after_negative_gap_sum": after["diagnostics"]["negative_gap_sum"],
        "before_reaction_norm": before["diagnostics"]["reaction_norm"],
        "after_reaction_norm": after["diagnostics"]["reaction_norm"],
        "followup_du_norm": followup_du_norm,
        "after_metrics": after,
    }


def branch_step_comparison(common_snapshot, attempt_load):
    outputs = []
    for use_tangent in (False, True):
        state, solver_cfg, mode_cfg = clone_state_with_snapshot("aggressive", common_snapshot)
        step_data = {
            "step": 2,
            "load_value": attempt_load,
            "attempt_load": attempt_load,
            "target_load": attempt_load,
            "cutback_level": 0,
            "label": f"branch_{'tangent' if use_tangent else 'notangent'}",
        }
        _, info = solve_staggered_contact_step(
            state,
            solver_cfg,
            step_data,
            use_contact_tangent_uu=use_tangent,
            max_outer_iter=mode_cfg["max_outer_iter"],
            verbose=False,
        )
        outputs.append(
            {
                "use_tangent": use_tangent,
                "outer_iterations": info["outer_iterations"],
                "converged": info["converged"],
                "contact_rhs_norm": info["contact_rhs_norm"],
                "du_norm": info["du_norm"],
                "reaction_norm": info["reaction_norm"],
                "negative_gap_sum": info["negative_gap_sum"],
                "max_penetration": info["max_penetration"],
                "failure_reason": info["failure_reason"],
            }
        )
    return outputs


def main():
    state_false, solver_cfg_false, mode_cfg_false = build_indenter_state("aggressive")
    _, result_false = solve_staggered_contact_loadpath(
        state_false,
        solver_cfg_false,
        build_load_schedule("aggressive"),
        use_contact_tangent_uu=False,
        max_outer_iter=mode_cfg_false["max_outer_iter"],
        max_cutbacks=mode_cfg_false["max_cutbacks"],
        min_cutback_increment=mode_cfg_false["min_cutback_increment"],
        write_outputs=False,
        history_path=None,
        verbose=False,
        step_verbose=False,
    )
    state_true, solver_cfg_true, mode_cfg_true = build_indenter_state("aggressive")
    _, result_true = solve_staggered_contact_loadpath(
        state_true,
        solver_cfg_true,
        build_load_schedule("aggressive"),
        use_contact_tangent_uu=True,
        max_outer_iter=mode_cfg_true["max_outer_iter"],
        max_cutbacks=mode_cfg_true["max_cutbacks"],
        min_cutback_increment=mode_cfg_true["min_cutback_increment"],
        write_outputs=False,
        history_path=None,
        verbose=False,
        step_verbose=False,
    )

    accepted_false, snapshot_false = accepted_snapshot_at_load(result_false, 0.01)
    accepted_true, snapshot_true = accepted_snapshot_at_load(result_true, 0.01)
    snapshot_u_diff = float(np.linalg.norm(snapshot_false["u"] - snapshot_true["u"]))
    snapshot_phi_diff = float(np.linalg.norm(snapshot_false["phi"] - snapshot_true["phi"]))

    fixed_without = run_fixed_state_structure_test(snapshot_false, "without_Kuu", use_tangent=False)
    fixed_with = run_fixed_state_structure_test(snapshot_false, "with_Kuu", use_tangent=True)

    state_for_diag, _, _ = clone_state_with_snapshot("aggressive", snapshot_false)
    assembled = evaluate_state_metrics(state_for_diag, need_tangent=True)
    K_uu = assembled["K_uu_c"]
    K_uu_masked, constrained_dofs = mask_contact_tangent_for_bcs(K_uu, state_for_diag["solid_bcs"])
    K_uu_diag = diagnose_dense_operator(K_uu)
    K_uu_masked_diag = diagnose_dense_operator(K_uu_masked)
    A_struct, _, meta_struct = assemble_linear_solid_system(
        state_for_diag["u"],
        state_for_diag["R_u_form"],
        state_for_diag["solid_bcs"],
        contact_rhs=assembled["R_u_c"],
        contact_tangent_uu=None,
    )
    K_struct = dense_array_from_petsc_mat(A_struct)
    K_struct_diag = diagnose_dense_operator(K_struct)
    A_plus, _, meta_plus = assemble_linear_solid_system(
        state_for_diag["u"],
        state_for_diag["R_u_form"],
        state_for_diag["solid_bcs"],
        contact_rhs=assembled["R_u_c"],
        contact_tangent_uu=K_uu,
    )
    K_plus = dense_array_from_petsc_mat(A_plus)
    K_plus_diag = diagnose_dense_operator(K_plus)

    false_attempts = result_false["attempt_history"]
    true_attempts = result_true["attempt_history"]
    earliest_divergence = None
    for idx, (row_false, row_true) in enumerate(zip(false_attempts, true_attempts), start=1):
        signature_false = (
            row_false["step"],
            round(float(row_false["attempt_load"]), 12),
            bool(row_false["accepted"]),
            int(row_false["cutback_level"]),
        )
        signature_true = (
            row_true["step"],
            round(float(row_true["attempt_load"]), 12),
            bool(row_true["accepted"]),
            int(row_true["cutback_level"]),
        )
        if signature_false != signature_true:
            earliest_divergence = (idx, row_false, row_true)
            break

    common_prefix_length = 0
    for row_false, row_true in zip(false_attempts, true_attempts):
        signature_false = (
            row_false["step"],
            round(float(row_false["attempt_load"]), 12),
            bool(row_false["accepted"]),
            int(row_false["cutback_level"]),
        )
        signature_true = (
            row_true["step"],
            round(float(row_true["attempt_load"]), 12),
            bool(row_true["accepted"]),
            int(row_true["cutback_level"]),
        )
        if signature_false != signature_true:
            break
        common_prefix_length += 1

    common_accepted_load = 0.0
    for row_false, row_true in zip(false_attempts[:common_prefix_length], true_attempts[:common_prefix_length]):
        if row_false["accepted"] and row_true["accepted"]:
            common_accepted_load = float(row_false["load_value"])

    common_row, common_snapshot = accepted_snapshot_at_load(result_false, common_accepted_load)
    divergence_attempt_load = (
        float(earliest_divergence[1]["attempt_load"]) if earliest_divergence is not None else 0.04
    )
    common_attempt_outputs = branch_step_comparison(common_snapshot, divergence_attempt_load)

    print("Contact tangent uu effect diagnostic")
    print("")
    print("Accepted-state comparison at load 0.01:")
    print("  reference state for fixed-state test = Kuu=False accepted snapshot at load 0.01")
    print(f"  ||u_false(load=0.01) - u_true(load=0.01)||_2 = {snapshot_u_diff}")
    print(f"  ||phi_false(load=0.01) - phi_true(load=0.01)||_2 = {snapshot_phi_diff}")
    print("")
    print("Fixed-state structure-step comparison on the reference snapshot (Kuu=False at load 0.01):")
    print(
        "  case | du_norm | contact_rhs_norm_before | contact_rhs_norm_after | "
        "negative_gap_sum_after | reaction_norm_after | followup_du_norm"
    )
    for row in (fixed_without, fixed_with):
        print(
            f"  {row['label']} | {row['du_norm']:.6e} | "
            f"{row['before_contact_rhs_norm']:.6e} | {row['after_contact_rhs_norm']:.6e} | "
            f"{row['after_negative_gap_sum']:.6e} | {row['after_reaction_norm']:.6e} | "
            f"{row['followup_du_norm']:.6e}"
        )
    print("")
    print("Contact operator sanity at load 0.01:")
    print(f"  ||K_uu^c||_F = {K_uu_diag['fro_norm']:.6e}")
    print(f"  symmetry_error(K_uu^c) = {K_uu_diag['symmetry_error']:.6e}")
    print(f"  max_abs_entry(K_uu^c) = {K_uu_diag['max_abs_entry']:.6e}")
    print(f"  ||K_struct||_F = {K_struct_diag['fro_norm']:.6e}")
    print(f"  ||K_uu^c||_F / ||K_struct||_F = {K_uu_diag['fro_norm'] / K_struct_diag['fro_norm']:.6e}")
    print(f"  constrained dof count = {len(constrained_dofs)}")
    print(f"  constrained_rhs_max after BC = {meta_struct['constrained_rhs_max']:.6e}")
    print(f"  constrained_tangent_max after masking = {meta_plus['constrained_tangent_max']:.6e}")
    print(f"  symmetry_error(masked K_uu^c) = {K_uu_masked_diag['symmetry_error']:.6e}")
    print(f"  ||K_struct + K_uu^c||_F = {K_plus_diag['fro_norm']:.6e}")
    try:
        cond_struct = float(np.linalg.cond(K_struct))
        cond_plus = float(np.linalg.cond(K_plus))
        print(f"  cond(K_struct) ~= {cond_struct:.6e}")
        print(f"  cond(K_struct + K_uu^c) ~= {cond_plus:.6e}")
    except np.linalg.LinAlgError:
        print("  cond estimate unavailable")
    print("")
    print("Aggressive loadpath earliest divergence:")
    if earliest_divergence is None:
        print("  No divergence detected in aligned attempt signatures.")
    else:
        idx, row_false, row_true = earliest_divergence
        print(f"  attempt index = {idx}")
        print(
            "  False: "
            f"step={row_false['step']} attempt_load={row_false['attempt_load']:.6f} "
            f"accepted={row_false['accepted']} cutback_level={row_false['cutback_level']}"
        )
        print(
            "  True : "
            f"step={row_true['step']} attempt_load={row_true['attempt_load']:.6f} "
            f"accepted={row_true['accepted']} cutback_level={row_true['cutback_level']}"
        )
    print("")
    print(
        "Same-input same-attempt comparison from the last common accepted state "
        f"(load = {common_accepted_load:.4f}, attempt load = {divergence_attempt_load:.4f}):"
    )
    print(
        "  case | converged | outer_iterations | contact_rhs_norm | "
        "du_norm | reaction_norm | negative_gap_sum | max_penetration | failure_reason"
    )
    for row in common_attempt_outputs:
        label = "with_Kuu" if row["use_tangent"] else "without_Kuu"
        print(
            f"  {label} | {row['converged']} | {row['outer_iterations']} | "
            f"{row['contact_rhs_norm']:.6e} | {row['du_norm']:.6e} | "
            f"{row['reaction_norm']:.6e} | {row['negative_gap_sum']:.6e} | "
            f"{row['max_penetration']:.6e} | {row['failure_reason']}"
        )


if __name__ == "__main__":
    main()

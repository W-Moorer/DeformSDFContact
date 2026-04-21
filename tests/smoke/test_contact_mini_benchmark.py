#!/usr/bin/env python3
"""Smoke and regression tests for the contact-strip mini benchmark."""

from __future__ import annotations

from deformsdfcontact.backend.dolfinx0p3.problem import build_contact_strip_benchmark
from deformsdfcontact.solvers import solve_with_load_stepping, solve_with_petsc_snes


def _run_strip_benchmark(contact_backend: str, steps: list[float]):
    return solve_with_load_stepping(
        lambda phi_bias: build_contact_strip_benchmark(
            contact_backend=contact_backend,
            phi_bias=phi_bias,
        ),
        steps,
        parameter_name="phi_bias",
        solve_function=solve_with_petsc_snes,
        solve_kwargs={"max_it": 25},
    )


def test_contact_strip_mini_benchmark_runs_for_transition_and_pairing_backends() -> None:
    steps = [0.3, 0.35, 0.4]
    transition = _run_strip_benchmark("transition", steps)
    pairing = _run_strip_benchmark("pairing", steps)

    assert transition.completed_all_steps
    assert pairing.completed_all_steps
    assert transition.failed_step_index is None
    assert pairing.failed_step_index is None
    assert len(transition.step_results) == len(steps)
    assert len(pairing.step_results) == len(steps)
    assert all(step.solve_result.converged_reason > 0 for step in transition.step_results)
    assert all(step.solve_result.converged_reason > 0 for step in pairing.step_results)
    assert all(step.solve_result.final_residual_norm < 1.0e-8 for step in transition.step_results)
    assert all(step.solve_result.final_residual_norm < 1.0e-8 for step in pairing.step_results)


def test_pairing_backend_reports_owned_pairs_and_contact_metrics() -> None:
    stepped = _run_strip_benchmark("pairing", [0.3, 0.35, 0.4])
    final_summary = stepped.step_results[-1].solve_result.diagnostics.contact_summary

    assert final_summary is not None
    assert final_summary.backend_name == "pairing"
    assert final_summary.owned_pair_count > 0
    assert final_summary.candidate_count >= final_summary.owned_pair_count
    assert final_summary.active_point_count > 0
    assert final_summary.gap_mean is not None
    assert final_summary.reaction_sum > 0.0


def test_transition_and_pairing_backends_show_interpretable_benchmark_difference() -> None:
    steps = [0.3, 0.35, 0.4]
    transition = _run_strip_benchmark("transition", steps)
    pairing = _run_strip_benchmark("pairing", steps)
    transition_summary = transition.step_results[-1].solve_result.diagnostics.contact_summary
    pairing_summary = pairing.step_results[-1].solve_result.diagnostics.contact_summary

    assert transition_summary is not None
    assert pairing_summary is not None
    assert transition_summary.backend_name == "transition"
    assert pairing_summary.backend_name == "pairing"
    assert transition_summary.owned_pair_count == 0
    assert pairing_summary.owned_pair_count > 0
    assert transition_summary.gap_mean != pairing_summary.gap_mean
    assert transition_summary.reaction_sum != pairing_summary.reaction_sum


def test_pairing_backend_failure_path_reports_last_converged_step() -> None:
    stepped = _run_strip_benchmark("pairing", [0.3, 0.4, 0.5])

    assert not stepped.completed_all_steps
    assert stepped.failed_step_index == 2
    assert stepped.last_converged_step_index == 1
    assert stepped.last_converged_parameter_value == 0.4

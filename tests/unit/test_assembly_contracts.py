#!/usr/bin/env python3
"""Unit tests for monolithic assembly-neutral contracts."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.assembly import MonolithicBlockLayout, SolidLocalContribution


def test_monolithic_block_layout_offsets_and_shapes_are_consistent() -> None:
    layout = MonolithicBlockLayout(ndof_u=8, ndof_phi=5)

    assert layout.total_dofs == 13
    assert layout.phi_offset == 8
    assert layout.block_shapes["uu"] == (8, 8)
    assert layout.block_shapes["uphi"] == (8, 5)
    assert np.array_equal(
        layout.lift_phi_dofs(np.array([0, 3], dtype=np.int32)),
        np.array([8, 11], dtype=np.int32),
    )


def test_solid_local_contribution_validates_shapes() -> None:
    contribution = SolidLocalContribution(
        u_dofs=np.array([0, 1, 2], dtype=np.int32),
        R_u=np.array([1.0, 2.0, 3.0], dtype=float),
        K_uu=np.eye(3, dtype=float),
    )

    assert contribution.K_uu.shape == (3, 3)

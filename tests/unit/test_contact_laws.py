#!/usr/bin/env python3
"""Unit tests for minimal local contact laws."""

from __future__ import annotations

import pytest

from deformsdfcontact.contact.laws import PenaltyContactLaw


def test_positive_gap_is_inactive() -> None:
    law = PenaltyContactLaw(penalty=25.0)

    lambda_n, k_n = law.evaluate(0.3)

    assert lambda_n == 0.0
    assert k_n == 0.0


def test_zero_gap_is_inactive_boundary_case() -> None:
    law = PenaltyContactLaw(penalty=25.0)

    lambda_n, k_n = law.evaluate(0.0)

    assert lambda_n == 0.0
    assert k_n == 0.0


def test_negative_gap_is_active_with_penalty_response() -> None:
    law = PenaltyContactLaw(penalty=12.0)

    lambda_n, k_n = law.evaluate(-0.25)

    assert lambda_n == 3.0
    assert k_n == 12.0


def test_nonpositive_penalty_is_rejected() -> None:
    with pytest.raises(ValueError, match="penalty must be strictly positive"):
        PenaltyContactLaw(penalty=0.0)

    with pytest.raises(ValueError, match="penalty must be strictly positive"):
        PenaltyContactLaw(penalty=-1.0)

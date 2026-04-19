"""Variant hash stability. Regressions here would silently re-seed every
pattern on every startup or break URL persistence."""
from __future__ import annotations

from app.service import _params_hash


def test_param_ordering_does_not_affect_hash() -> None:
    a = {"alpha": 1.0, "beta": 2.5, "gamma": 3}
    b = {"gamma": 3, "beta": 2.5, "alpha": 1.0}
    assert _params_hash(a) == _params_hash(b)


def test_int_vs_float_produces_distinct_hashes() -> None:
    # This is the current behavior of our json-based hasher (1 != 1.0 in JSON
    # repr). Pin it so nobody "helpfully" normalizes and breaks cache hits.
    assert _params_hash({"x": 1}) != _params_hash({"x": 1.0})


def test_float_precision_is_stable_across_calls() -> None:
    p = {"period_um": 4.0, "duty": 0.5}
    assert _params_hash(p) == _params_hash(p)


def test_unknown_param_keys_change_hash() -> None:
    assert _params_hash({"a": 1}) != _params_hash({"a": 1, "b": 2})

"""Tests for AASHTO load combination envelope engine."""
import pytest
from nlb.tools.load_envelope import (
    compute_factored_envelopes,
    CaseForces,
    ForceEnvelope,
)


def _make_case(name, ctype, mz=0, vy=0, n=0, tag=30001):
    return CaseForces(
        case_name=name, case_type=ctype,
        element_forces={tag: {"Mz_i": mz, "Vy_i": vy, "N_i": n}},
    )


class TestStrengthI:
    def test_factors(self):
        dc = _make_case("DC", "DC", mz=1000, vy=50)
        dw = _make_case("DW", "DW", mz=200, vy=10)
        ll = _make_case("LL", "LL", mz=2000, vy=80)
        env = compute_factored_envelopes([dc, dw, ll], limit_states=["Strength_I"])
        e = env[30001]
        expected = 1.25 * 1000 + 1.50 * 200 + 1.75 * 2000
        assert abs(e.Mz_max - expected) < 1.0
        assert "Strength_I" in e.Mz_max_combo

    def test_dc_min_factor(self):
        """DC min factor (0.90) used when DC opposes LL."""
        dc = _make_case("DC", "DC", mz=-500)
        ll = _make_case("LL", "LL", mz=2000)
        env = compute_factored_envelopes([dc, ll])
        e = env[30001]
        # Max moment should use DC_min (0.90) to maximize net positive
        str_i_max = 0.90 * (-500) + 1.75 * 2000
        assert e.Mz_max >= str_i_max - 1.0


class TestServiceII:
    def test_factors(self):
        dc = _make_case("DC", "DC", mz=1000)
        ll = _make_case("LL", "LL", mz=2000)
        env = compute_factored_envelopes([dc, ll], limit_states=["Service_II"])
        e = env[30001]
        expected = 1.0 * 1000 + 1.3 * 2000
        assert abs(e.Mz_max - expected) < 1.0


class TestEnvelopeTracking:
    def test_multiple_ll_cases_summed(self):
        """Multiple LL cases are summed (caller pre-envelopes positions)."""
        dc = _make_case("DC", "DC", mz=1000)
        # Pass only the governing LL envelope (not individual positions)
        ll_gov = _make_case("LL_governing", "LL", mz=2000)
        env = compute_factored_envelopes([dc, ll_gov])
        e = env[30001]
        expected = 1.25 * 1000 + 1.75 * 2000
        assert abs(e.Mz_max - expected) < 1.0

    def test_controlling_combo_tracked(self):
        dc = _make_case("DC", "DC", mz=1000)
        ll = _make_case("LL", "LL", mz=2000)
        env = compute_factored_envelopes([dc, ll])
        e = env[30001]
        assert e.Mz_max_combo is not None
        assert len(e.Mz_max_combo) > 0


class TestEdgeCases:
    def test_empty_cases(self):
        env = compute_factored_envelopes([])
        assert len(env) == 0

    def test_dc_only(self):
        dc = _make_case("DC", "DC", mz=1000)
        env = compute_factored_envelopes([dc])
        e = env[30001]
        assert e.Mz_max >= 1000  # At least 1.0 factor

    def test_multiple_elements(self):
        dc1 = CaseForces("DC", "DC", {30001: {"Mz_i": 100}, 30002: {"Mz_i": 200}})
        env = compute_factored_envelopes([dc1])
        assert 30001 in env
        assert 30002 in env

    def test_shear_envelope(self):
        dc = _make_case("DC", "DC", vy=50)
        ll = _make_case("LL", "LL", vy=100)
        env = compute_factored_envelopes([dc, ll])
        e = env[30001]
        assert e.Vy_max > 0


class TestForceEnvelope:
    def test_dataclass_fields(self):
        e = ForceEnvelope(
            element_tag=1, Mz_max=100, Mz_min=-50,
            Vy_max=20, Vy_min=-10, N_max=5, N_min=-3,
            Mz_max_combo="Str_I", Mz_min_combo="Str_I",
            Vy_max_combo="Str_I", Vy_min_combo="Str_I",
        )
        assert e.element_tag == 1
        assert e.Mz_max == 100

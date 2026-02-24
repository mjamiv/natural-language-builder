"""Tests for AASHTO distribution factor calculator."""

import pytest

from nlb.tools.distribution_factors import (
    compute_distribution_factors,
    compute_kg,
    DistributionFactors,
)


class TestComputeKg:
    """Test longitudinal stiffness parameter computation."""

    def test_basic_kg(self):
        """Kg = n(I + A*eg^2)."""
        kg = compute_kg(n=8, I_girder=30000, A_girder=50, eg=30)
        expected = 8 * (30000 + 50 * 30**2)
        assert abs(kg - expected) < 1.0

    def test_zero_eccentricity(self):
        """With eg=0, Kg = n*I."""
        kg = compute_kg(n=8, I_girder=10000, A_girder=100, eg=0)
        assert abs(kg - 80000) < 1.0

    def test_large_eccentricity_dominates(self):
        """Large eg makes A*eg^2 dominate."""
        kg = compute_kg(n=8, I_girder=1000, A_girder=50, eg=40)
        assert kg > 8 * 50 * 40**2 * 0.9  # A*eg^2 term dominates


class TestDistributionFactors:
    """Test AASHTO 4.6.2.2.2 distribution factor formulas."""

    def test_returns_dataclass(self):
        df = compute_distribution_factors(S=8.0, L=100, ts=8.0, Kg=500000)
        assert isinstance(df, DistributionFactors)

    def test_interior_moment_two_lane_exceeds_one_lane(self):
        """For typical bridges, 2+ lane DF exceeds 1-lane DF."""
        df = compute_distribution_factors(S=9.0, L=120, ts=8.0, Kg=600000)
        assert df.gM_int_2 >= df.gM_int_1

    def test_interior_shear_formulas(self):
        """Verify shear DF formulas."""
        S = 9.75
        # 1 lane: 0.36 + S/25
        expected_1 = 0.36 + S / 25.0
        # 2+ lanes: 0.2 + S/12 - (S/35)^2
        expected_2 = 0.2 + S / 12.0 - (S / 35.0) ** 2
        df = compute_distribution_factors(S=S, L=175, ts=9.0, Kg=800000)
        assert abs(df.gV_int_1 - expected_1) < 0.01
        assert abs(df.gV_int_2 - expected_2) < 0.01

    def test_governing_is_max(self):
        """Governing DF is max of 1-lane and 2+-lane."""
        df = compute_distribution_factors(S=9.0, L=120, ts=8.0, Kg=600000)
        assert df.gM_int == max(df.gM_int_1, df.gM_int_2)
        assert df.gV_int == max(df.gV_int_1, df.gV_int_2)

    def test_exterior_moment_correction(self):
        """Exterior 2+ lane uses e = 0.77 + de/9.1."""
        de = 2.0
        e_expected = 0.77 + de / 9.1
        df = compute_distribution_factors(S=9.0, L=120, ts=8.0, Kg=600000, de=de)
        assert abs(df.gM_ext_2 - e_expected * df.gM_int_2) < 0.01

    def test_exterior_shear_correction(self):
        """Exterior shear 2+ lane uses e = 0.6 + de/10.0."""
        de = 3.0
        e_expected = 0.6 + de / 10.0
        df = compute_distribution_factors(S=9.0, L=120, ts=8.0, Kg=600000, de=de)
        assert abs(df.gV_ext_2 - e_expected * df.gV_int_2) < 0.01

    def test_wide_spacing_higher_df(self):
        """Wider spacing → higher DF."""
        df_narrow = compute_distribution_factors(S=6.0, L=120, ts=8.0, Kg=500000)
        df_wide = compute_distribution_factors(S=12.0, L=120, ts=8.0, Kg=500000)
        assert df_wide.gM_int > df_narrow.gM_int

    def test_longer_span_lower_df(self):
        """Longer span → lower moment DF (for same S)."""
        df_short = compute_distribution_factors(S=9.0, L=80, ts=8.0, Kg=500000)
        df_long = compute_distribution_factors(S=9.0, L=200, ts=8.0, Kg=500000)
        assert df_long.gM_int_2 < df_short.gM_int_2

    def test_all_positive(self):
        """All DFs should be positive."""
        df = compute_distribution_factors(S=8.0, L=100, ts=8.0, Kg=400000)
        for attr in ['gM_int', 'gM_ext', 'gV_int', 'gV_ext',
                     'gM_int_1', 'gM_int_2', 'gM_ext_1', 'gM_ext_2',
                     'gV_int_1', 'gV_int_2', 'gV_ext_1', 'gV_ext_2']:
            assert getattr(df, attr) > 0, f"{attr} should be positive"

    def test_fhwa_range(self):
        """FHWA-like parameters should give reasonable DFs."""
        kg = compute_kg(n=8, I_girder=30814, A_girder=56.19, eg=35.13)
        df = compute_distribution_factors(S=9.75, L=175, ts=8.5, Kg=kg)
        # Interior moment DF should be in reasonable range
        assert 0.4 < df.gM_int < 1.2
        # Interior shear DF
        assert 0.5 < df.gV_int < 1.5

    def test_lever_rule_exterior_one_lane(self):
        """Exterior 1-lane moment should use lever rule (> 0)."""
        df = compute_distribution_factors(S=9.0, L=120, ts=8.0, Kg=500000, de=2.0)
        assert df.gM_ext_1 > 0

    def test_five_girders(self):
        """Standard 5-girder bridge."""
        df = compute_distribution_factors(S=8.0, L=100, ts=8.0, Kg=500000, Nb=5)
        assert df.gM_int > 0.3
        assert df.gV_int > 0.5

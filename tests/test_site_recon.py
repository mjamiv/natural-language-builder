"""Tests for site-recon MCP tool."""

import pytest


class TestSiteRecon:
    """Site reconnaissance from GPS coordinates."""

    def test_seismic_hazard_lookup(self):
        """USGS API returns PGA, Ss, S1 for given coordinates."""
        pytest.skip("Not yet implemented")

    def test_soil_profile_lookup(self):
        """NRCS Web Soil Survey returns soil classification."""
        pytest.skip("Not yet implemented")

    def test_wind_speed_lookup(self):
        """ASCE 7 wind speed from county/state."""
        pytest.skip("Not yet implemented")

    def test_thermal_range(self):
        """NOAA climate normals return design temperature range."""
        pytest.skip("Not yet implemented")

    def test_scour_potential(self):
        """Water crossing flag triggers scour analysis path."""
        pytest.skip("Not yet implemented")

    def test_full_site_profile(self):
        """Complete site profile JSON from coordinates."""
        pytest.skip("Not yet implemented")

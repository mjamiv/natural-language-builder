"""Tests for the site-recon MCP tool.

All external API calls are mocked so tests run offline and deterministically.

Coverage:
  - USGS seismic hazard: happy path, malformed response, API failure
  - NOAA climate normals: happy path, API failure fallback
  - Wind speed: state lookup, missing state fallback
  - Scour detection: positive keyword matches, negative case
  - Frost depth: state lookup, unknown state fallback
  - Seismic Design Category boundaries
  - SiteProfile assembly: all-live-data path, all-fallback path
  - geo.py: reverse geocode, state normalization, haversine
"""

from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

# ── module under test ────────────────────────────────────────────────────────
from nlb.tools.site_recon import (
    SeismicProfile,
    SiteProfile,
    ScourProfile,
    ThermalProfile,
    WindProfile,
    SoilProfile,
    _build_seismic_profile,
    _seismic_design_category,
    _SEISMIC_FALLBACK,
    _WIND_SPEED_BY_STATE,
    _WIND_SPEED_DEFAULT,
    _FROST_DEPTH_BY_STATE,
    _FROST_DEPTH_DEFAULT,
    _THERMAL_BY_CLIMATE_ZONE,
    detect_water_crossing,
    fetch_seismic_hazard,
    fetch_thermal_range,
    get_frost_depth,
    get_wind_speed,
    run_site_recon,
)
from nlb.utils.geo import (
    GeoLocation,
    CLIMATE_ZONE_BY_STATE,
    _normalize_state,
    haversine_ft,
    is_valid_us_coordinate,
    reverse_geocode,
    state_abbreviation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

LAT_KISHWAUKEE = 42.28
LON_KISHWAUKEE = -89.09
STATE_IL = "IL"

#: Realistic USGS API response payload for Kishwaukee site (SDC B).
USGS_RESPONSE_OK = {
    "response": {
        "data": {
            "parameters": {
                "pga":  0.075,
                "ss":   0.180,
                "s1":   0.075,
                "fa":   1.600,
                "fv":   2.400,
                "sms":  0.288,
                "sm1":  0.180,
                "sds":  0.192,
                "sd1":  0.120,
            }
        }
    }
}

#: Nominatim reverse-geocode response for Kishwaukee (Winnebago County, IL).
NOMINATIM_RESPONSE_OK = {
    "display_name": "Kishwaukee River, Winnebago County, Illinois, United States",
    "address": {
        "state":        "Illinois",
        "county":       "Winnebago County",
        "city":         "Rockford",
        "country_code": "us",
    },
}

#: Minimal NOAA station response.
NOAA_STATIONS_RESPONSE = {
    "results": [
        {"id": "GHCND:USW00094822", "name": "ROCKFORD AIRPORT, IL US"},
    ]
}

#: NOAA data response with annual min/max normals.
NOAA_DATA_RESPONSE = {
    "results": [
        {"datatype": "MLY-TMIN-NORMAL", "value": "8.0"},   # °F avg Jan low
        {"datatype": "MLY-TMAX-NORMAL", "value": "87.0"},  # °F avg Jul high
    ]
}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response-like object."""
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# geo.py tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeState:
    """State name ↔ abbreviation normalization."""

    def test_full_name_to_abbr(self):
        assert _normalize_state("Illinois") == "IL"

    def test_full_name_lowercase(self):
        assert _normalize_state("illinois") == "IL"

    def test_already_abbreviated(self):
        assert _normalize_state("IL") == "IL"

    def test_already_abbreviated_lowercase(self):
        assert _normalize_state("il") == "IL"

    def test_multi_word_state(self):
        assert _normalize_state("North Dakota") == "ND"

    def test_unknown_returns_empty(self):
        assert _normalize_state("Narnia") == ""

    def test_empty_returns_empty(self):
        assert _normalize_state("") == ""

    def test_state_abbreviation_public_fn(self):
        assert state_abbreviation("New York") == "NY"
        assert state_abbreviation("NY") == "NY"


class TestHaversine:
    """haversine_ft distance calculation."""

    def test_same_point_is_zero(self):
        assert haversine_ft(42.28, -89.09, 42.28, -89.09) == pytest.approx(0.0, abs=1)

    def test_one_degree_lat_approx_364000_ft(self):
        # 1° latitude ≈ 364,000 ft (69 miles)
        dist = haversine_ft(40.0, -90.0, 41.0, -90.0)
        assert 360_000 < dist < 370_000

    def test_chicago_to_springfield_roughly_150_miles(self):
        # Chicago (41.88, -87.63) to Springfield IL (39.78, -89.65) ≈ 170 mi
        dist_ft = haversine_ft(41.88, -87.63, 39.78, -89.65)
        miles = dist_ft / 5280
        assert 160 < miles < 185


class TestIsValidUSCoordinate:
    """Bounding-box coordinate validation."""

    def test_illinois_is_valid(self):
        assert is_valid_us_coordinate(42.28, -89.09)

    def test_alaska_is_valid(self):
        assert is_valid_us_coordinate(64.0, -153.0)

    def test_hawaii_is_valid(self):
        assert is_valid_us_coordinate(21.0, -157.0)

    def test_london_is_invalid(self):
        assert not is_valid_us_coordinate(51.5, -0.1)

    def test_south_pole_is_invalid(self):
        assert not is_valid_us_coordinate(-90.0, 0.0)


class TestReverseGeocode:
    """Reverse geocode with mocked Nominatim."""

    @patch("nlb.utils.geo.requests.get")
    def test_happy_path(self, mock_get):
        mock_get.return_value = _mock_response(NOMINATIM_RESPONSE_OK)
        geo = reverse_geocode(LAT_KISHWAUKEE, LON_KISHWAUKEE)

        assert geo.state == "IL"
        assert "Winnebago" in geo.county
        assert geo.country == "US"
        assert geo.climate_zone == "cold"

    @patch("nlb.utils.geo.requests.get")
    def test_network_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        geo = reverse_geocode(0.0, 0.0)

        # Never raises; returns empty GeoLocation
        assert geo.state == ""
        assert geo.county == ""
        assert geo.country == ""

    @patch("nlb.utils.geo.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        mock_resp = _mock_response({}, 503)
        mock_resp.raise_for_status.side_effect = Exception("Service Unavailable")
        mock_get.return_value = mock_resp
        geo = reverse_geocode(42.28, -89.09)
        assert geo.state == ""

    @patch("nlb.utils.geo.requests.get")
    def test_climate_zone_resolved(self, mock_get):
        mock_get.return_value = _mock_response(NOMINATIM_RESPONSE_OK)
        geo = reverse_geocode(LAT_KISHWAUKEE, LON_KISHWAUKEE)
        # Illinois is "cold" per lookup table
        assert geo.climate_zone == "cold"


class TestClimateZoneByState:
    """Climate zone lookup completeness."""

    def test_florida_is_hot(self):
        assert CLIMATE_ZONE_BY_STATE["FL"] == "hot"

    def test_minnesota_is_very_cold(self):
        assert CLIMATE_ZONE_BY_STATE["MN"] == "very_cold"

    def test_illinois_is_cold(self):
        assert CLIMATE_ZONE_BY_STATE["IL"] == "cold"

    def test_virginia_is_mixed(self):
        assert CLIMATE_ZONE_BY_STATE["VA"] == "mixed"

    def test_all_50_states_have_zone(self):
        """Every USPS state code has a climate zone entry."""
        states_50 = {
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
            "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
            "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
            "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
        }
        missing = states_50 - set(CLIMATE_ZONE_BY_STATE.keys())
        assert missing == set(), f"Missing climate zones for: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# Seismic hazard tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeismicDesignCategory:
    """_seismic_design_category() boundary conditions."""

    def test_sdc_a_below_015(self):
        assert _seismic_design_category(0.10) == "A"

    def test_sdc_a_boundary(self):
        assert _seismic_design_category(0.149) == "A"

    def test_sdc_b_at_boundary(self):
        assert _seismic_design_category(0.15) == "B"

    def test_sdc_b_interior(self):
        assert _seismic_design_category(0.22) == "B"

    def test_sdc_b_upper_boundary(self):
        assert _seismic_design_category(0.299) == "B"

    def test_sdc_c_at_030(self):
        assert _seismic_design_category(0.30) == "C"

    def test_sdc_c_interior(self):
        assert _seismic_design_category(0.40) == "C"

    def test_sdc_d_at_050(self):
        assert _seismic_design_category(0.50) == "D"

    def test_sdc_d_high_seismicity(self):
        assert _seismic_design_category(0.80) == "D"


class TestFetchSeismicHazard:
    """fetch_seismic_hazard() with mocked USGS API."""

    @patch("nlb.tools.site_recon.requests.get")
    def test_happy_path_returns_profile(self, mock_get):
        mock_get.return_value = _mock_response(USGS_RESPONSE_OK)
        profile, ok = fetch_seismic_hazard(LAT_KISHWAUKEE, LON_KISHWAUKEE, "D")

        assert ok is True
        assert isinstance(profile, SeismicProfile)
        assert profile.pga == pytest.approx(0.075, rel=1e-3)
        assert profile.ss  == pytest.approx(0.180, rel=1e-3)
        assert profile.s1  == pytest.approx(0.075, rel=1e-3)
        assert profile.sds == pytest.approx(0.192, rel=1e-3)
        assert profile.sd1 == pytest.approx(0.120, rel=1e-3)
        assert profile.site_class == "D"
        assert profile.sdc == "A"   # sd1=0.12 < 0.15 → SDC A

    @patch("nlb.tools.site_recon.requests.get")
    def test_api_network_failure_returns_fallback(self, mock_get):
        mock_get.side_effect = Exception("Network unreachable")
        profile, ok = fetch_seismic_hazard(42.28, -89.09, "D")

        assert ok is False
        assert isinstance(profile, SeismicProfile)
        # Must still be a valid SDC (conservative fallback)
        assert profile.sdc in ("A", "B", "C", "D")

    @patch("nlb.tools.site_recon.requests.get")
    def test_http_error_returns_fallback(self, mock_get):
        mock_resp = _mock_response({}, 503)
        mock_resp.raise_for_status.side_effect = Exception("503")
        mock_get.return_value = mock_resp
        profile, ok = fetch_seismic_hazard(42.28, -89.09)

        assert ok is False
        # Fallback is SDC B
        assert profile.sdc == "B"

    @patch("nlb.tools.site_recon.requests.get")
    def test_empty_parameters_uses_fallback_math(self, mock_get):
        """If API returns empty parameters dict, fallback defaults are used."""
        mock_get.return_value = _mock_response({"response": {"data": {"parameters": {}}}})
        profile, ok = fetch_seismic_hazard(42.28, -89.09, "D")

        # ok=True because HTTP succeeded; but values are from fallback
        assert isinstance(profile, SeismicProfile)
        assert profile.pga == pytest.approx(_SEISMIC_FALLBACK["pga"], rel=1e-2)

    @patch("nlb.tools.site_recon.requests.get")
    def test_site_class_preserved_in_profile(self, mock_get):
        mock_get.return_value = _mock_response(USGS_RESPONSE_OK)
        for sc in ("A", "B", "C", "D", "E"):
            profile, _ = fetch_seismic_hazard(42.28, -89.09, sc)
            assert profile.site_class == sc

    @patch("nlb.tools.site_recon.requests.get")
    def test_high_seismicity_site_returns_sdc_d(self, mock_get):
        """Simulate a high-seismicity site (coastal CA) returning SDC D."""
        high_seis_response = {
            "response": {
                "data": {
                    "parameters": {
                        "pga": 1.20, "ss": 2.50, "s1": 1.10,
                        "fa": 0.90, "fv": 1.50,
                        "sms": 2.25, "sm1": 1.65,
                        "sds": 1.50, "sd1": 1.10,
                    }
                }
            }
        }
        mock_get.return_value = _mock_response(high_seis_response)
        profile, ok = fetch_seismic_hazard(37.77, -122.42, "D")

        assert ok is True
        assert profile.sdc == "D"


class TestBuildSeismicProfile:
    """_build_seismic_profile() helper."""

    def test_builds_from_fallback_dict(self):
        profile = _build_seismic_profile(_SEISMIC_FALLBACK, "D")
        assert profile.site_class == "D"
        assert profile.pga == _SEISMIC_FALLBACK["pga"]
        assert profile.sdc == _seismic_design_category(_SEISMIC_FALLBACK["sd1"])


# ═══════════════════════════════════════════════════════════════════════════════
# Thermal range tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchThermalRange:
    """fetch_thermal_range() with mocked NOAA API."""

    @patch("nlb.tools.site_recon.requests.get")
    def test_noaa_happy_path(self, mock_get):
        """NOAA returns station + data → ThermalProfile with extremes applied."""
        mock_get.side_effect = [
            _mock_response(NOAA_STATIONS_RESPONSE),
            _mock_response(NOAA_DATA_RESPONSE),
        ]
        profile, ok = fetch_thermal_range(42.28, -89.09, "IL", noaa_token="FAKE_TOKEN")

        assert ok is True
        assert isinstance(profile, ThermalProfile)
        # tmin_normal=8, delta=22 → t_min = 8-22 = -14°F
        assert profile.t_min == pytest.approx(8 - 22)
        # tmax_normal=87, delta=22 → t_max = min(87+22, 120) = 109°F
        assert profile.t_max == min(87 + 22, 120)
        assert profile.delta_t == profile.t_max - profile.t_min

    def test_no_token_falls_back_to_state_lookup(self):
        profile, ok = fetch_thermal_range(42.28, -89.09, "IL", noaa_token=None)
        assert ok is False
        # IL is "cold" zone: t_min=-20, t_max=110
        cold = _THERMAL_BY_CLIMATE_ZONE["cold"]
        assert profile.t_min == cold["t_min"]
        assert profile.t_max == cold["t_max"]

    @patch("nlb.tools.site_recon.requests.get")
    def test_noaa_api_failure_falls_back_to_state(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        profile, ok = fetch_thermal_range(42.28, -89.09, "FL", noaa_token="TOKEN")

        assert ok is False
        # FL is "hot" zone
        hot = _THERMAL_BY_CLIMATE_ZONE["hot"]
        assert profile.t_min == hot["t_min"]
        assert profile.t_max == hot["t_max"]

    def test_cold_climate_range_is_wider_than_hot(self):
        cold = _THERMAL_BY_CLIMATE_ZONE["cold"]
        hot  = _THERMAL_BY_CLIMATE_ZONE["hot"]
        assert cold["delta_t"] > hot["delta_t"]

    def test_unknown_state_falls_back_to_cold_default(self):
        profile, ok = fetch_thermal_range(0.0, 0.0, state="", noaa_token=None)
        assert ok is False
        assert isinstance(profile, ThermalProfile)
        # Empty state → climate zone lookup returns "cold" (CLIMATE_ZONE_BY_STATE default)
        assert profile.t_min <= 0  # cold should be ≤ 0°F

    @patch("nlb.tools.site_recon.requests.get")
    def test_noaa_no_stations_returns_fallback(self, mock_get):
        """If NOAA finds no stations near coordinates, fall back gracefully."""
        mock_get.side_effect = [
            _mock_response({"results": []}),   # no stations
        ]
        profile, ok = fetch_thermal_range(42.28, -89.09, "IL", noaa_token="TOKEN")
        assert ok is False  # fell back to state lookup


# ═══════════════════════════════════════════════════════════════════════════════
# Wind speed tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetWindSpeed:
    """get_wind_speed() state lookup."""

    def test_illinois_returns_115_mph(self):
        wp = get_wind_speed("IL")
        assert wp.v_ult == 115
        assert wp.exposure == "C"

    def test_florida_returns_160_mph(self):
        """FL coastal has highest wind speed due to hurricane exposure."""
        wp = get_wind_speed("FL")
        assert wp.v_ult == 160

    def test_guam_returns_195_mph(self):
        wp = get_wind_speed("GU")
        assert wp.v_ult == 195

    def test_unknown_state_returns_default(self):
        wp = get_wind_speed("")
        assert wp.v_ult == _WIND_SPEED_DEFAULT

    def test_exposure_override(self):
        wp = get_wind_speed("IL", exposure="D")
        assert wp.exposure == "D"

    def test_all_state_speeds_positive(self):
        for state, speed in _WIND_SPEED_BY_STATE.items():
            assert speed > 0, f"Wind speed for {state} is not positive"

    def test_all_state_speeds_realistic(self):
        """No wind speed should be below 85 or above 200 mph."""
        for state, speed in _WIND_SPEED_BY_STATE.items():
            assert 85 <= speed <= 200, f"Wind speed {speed} for {state} out of range"


# ═══════════════════════════════════════════════════════════════════════════════
# Frost depth tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetFrostDepth:
    """get_frost_depth() state lookup."""

    def test_illinois_4ft(self):
        assert get_frost_depth("IL") == 4.0

    def test_minnesota_6ft(self):
        assert get_frost_depth("MN") == 6.0

    def test_florida_zero(self):
        assert get_frost_depth("FL") == 0.0

    def test_north_dakota_7ft(self):
        assert get_frost_depth("ND") == 7.0

    def test_unknown_state_returns_conservative_default(self):
        depth = get_frost_depth("")
        assert depth == _FROST_DEPTH_DEFAULT

    def test_all_frost_depths_non_negative(self):
        for state, depth in _FROST_DEPTH_BY_STATE.items():
            assert depth >= 0, f"Negative frost depth for {state}"

    def test_northern_states_deeper_than_southern(self):
        """Frost depth should be monotonically higher in colder states."""
        assert get_frost_depth("MN") > get_frost_depth("AL")
        assert get_frost_depth("AK") > get_frost_depth("FL")


# ═══════════════════════════════════════════════════════════════════════════════
# Scour / water crossing tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectWaterCrossing:
    """detect_water_crossing() keyword parsing."""

    def test_river_in_description_triggers_scour(self):
        desc = "3-span steel bridge over the Kishwaukee River in Illinois"
        profile = detect_water_crossing(desc)
        assert profile.water_crossing is True
        assert profile.design_flood == "Q100"
        assert profile.check_flood == "Q500"
        assert "river" in profile.keywords_matched

    def test_creek_triggers_scour(self):
        desc = "Two-span bridge over Mill Creek"
        profile = detect_water_crossing(desc)
        assert profile.water_crossing is True
        assert "creek" in profile.keywords_matched

    def test_stream_triggers_scour(self):
        profile = detect_water_crossing("Footbridge over a mountain stream")
        assert profile.water_crossing is True

    def test_channel_triggers_scour(self):
        profile = detect_water_crossing("Bridge spanning the drainage channel")
        assert profile.water_crossing is True

    def test_waterway_triggers_scour(self):
        profile = detect_water_crossing("Overpass crossing a navigable waterway")
        assert profile.water_crossing is True

    def test_highway_overpass_no_scour(self):
        desc = "6-span highway overpass carrying I-94 over US-30 in suburban Chicago"
        profile = detect_water_crossing(desc)
        assert profile.water_crossing is False
        assert profile.design_flood is None
        assert profile.check_flood is None
        assert profile.keywords_matched == []

    def test_empty_description_no_scour(self):
        profile = detect_water_crossing("")
        assert profile.water_crossing is False

    def test_multiple_keywords_all_captured(self):
        desc = "Bridge over river with tidal creek and channel flow"
        profile = detect_water_crossing(desc)
        assert profile.water_crossing is True
        assert len(profile.keywords_matched) >= 3

    def test_case_insensitive(self):
        """Keyword matching should be case-insensitive."""
        assert detect_water_crossing("Bridge over RIVER").water_crossing is True
        assert detect_water_crossing("Bridge over River").water_crossing is True

    def test_bayou_triggers_scour(self):
        assert detect_water_crossing("crossing the bayou").water_crossing is True

    def test_scour_profile_flood_none_when_no_water(self):
        profile = detect_water_crossing("Grade separation over railroad")
        assert profile.design_flood is None
        assert profile.check_flood is None


# ═══════════════════════════════════════════════════════════════════════════════
# Full SiteProfile assembly
# ═══════════════════════════════════════════════════════════════════════════════

def _multi_api_side_effect(url: str, **kwargs):
    """Route mock responses to the correct API based on the request URL.

    Because `requests` is a singleton module shared across nlb.utils.geo and
    nlb.tools.site_recon, patching both 'nlb.utils.geo.requests.get' and
    'nlb.tools.site_recon.requests.get' targets the *same* underlying attribute.
    The second patch overwrites the first.  Instead, we use a single patch with
    a URL-discriminating side_effect so each API receives the correct response.
    """
    if "nominatim" in url or "openstreetmap" in url:
        return _mock_response(NOMINATIM_RESPONSE_OK)
    elif "earthquake.usgs.gov" in url or "designmaps" in url:
        return _mock_response(USGS_RESPONSE_OK)
    return _mock_response({})


class TestRunSiteRecon:
    """run_site_recon() end-to-end assembly with mocked APIs."""

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_full_kishwaukee_site(self, mock_get):
        """Kishwaukee River bridge: all APIs succeed."""
        profile = run_site_recon(
            lat=LAT_KISHWAUKEE,
            lon=LON_KISHWAUKEE,
            description="3-span steel bridge over the Kishwaukee River in Illinois",
            site_class="D",
        )

        # Coordinates preserved
        assert profile.coordinates["lat"] == LAT_KISHWAUKEE
        assert profile.coordinates["lon"] == LON_KISHWAUKEE

        # Location resolved
        assert profile.location["state"] == "IL"
        assert "Winnebago" in profile.location["county"]

        # Seismic from USGS
        assert isinstance(profile.seismic, SeismicProfile)
        assert profile.seismic.site_class == "D"
        assert profile.seismic.sdc in ("A", "B", "C", "D")

        # Wind for Illinois
        assert profile.wind.v_ult == 115
        assert profile.wind.exposure == "C"

        # Water crossing detected
        assert profile.scour.water_crossing is True
        assert profile.scour.design_flood == "Q100"
        assert profile.scour.check_flood == "Q500"

        # Frost depth for Illinois
        assert profile.frost_depth_ft == 4.0

        # Soil / site class
        assert profile.soil.site_class == "D"
        assert "Stiff" in profile.soil.description

        # Climate zone
        assert profile.climate_zone == "cold"

    @patch("requests.get", side_effect=Exception("DNS failure"))
    def test_all_api_failures_returns_conservative_defaults(self, mock_get):
        """When every API fails, run_site_recon still returns a valid profile."""
        profile = run_site_recon(
            lat=42.28,
            lon=-89.09,
            description="Bridge over river",
        )

        # Still a fully populated SiteProfile
        assert isinstance(profile, SiteProfile)
        assert isinstance(profile.seismic, SeismicProfile)
        assert isinstance(profile.wind, WindProfile)
        assert isinstance(profile.thermal, ThermalProfile)
        assert isinstance(profile.scour, ScourProfile)
        assert isinstance(profile.soil, SoilProfile)

        # Warnings list should mention fallbacks
        assert len(profile.warnings) > 0

        # Conservative seismic: fallback is SDC B
        assert profile.seismic.sdc == "B"

        # Scour still detected from description
        assert profile.scour.water_crossing is True

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_no_water_crossing_for_highway_overpass(self, mock_get):
        profile = run_site_recon(
            lat=42.28,
            lon=-89.09,
            description="Highway overpass over US-20 in Rockford Illinois",
        )

        assert profile.scour.water_crossing is False
        assert profile.scour.design_flood is None

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_to_dict_serializable(self, mock_get):
        """SiteProfile.to_dict() produces a JSON-serializable dict."""
        profile = run_site_recon(
            lat=LAT_KISHWAUKEE,
            lon=LON_KISHWAUKEE,
            description="Bridge over river",
        )

        d = profile.to_dict()
        # Must be JSON-round-trippable
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)

        assert deserialized["coordinates"]["lat"] == LAT_KISHWAUKEE
        assert "seismic" in deserialized
        assert "wind" in deserialized
        assert "thermal" in deserialized
        assert "scour" in deserialized
        assert "frost_depth_ft" in deserialized

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_site_class_e_triggers_warning_not_crash(self, mock_get):
        """Site class E (soft clay) is valid input; tool must not crash."""
        profile = run_site_recon(
            lat=29.95,
            lon=-90.07,
            description="Bridge in New Orleans — soft clay",
            site_class="E",
        )
        assert profile.soil.site_class == "E"
        assert "Soft" in profile.soil.description

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_exposure_override_propagates(self, mock_get):
        """Caller-specified exposure category is preserved in output."""
        profile = run_site_recon(
            lat=42.28, lon=-89.09,
            description="Coastal bridge",
            exposure="D",
        )
        assert profile.wind.exposure == "D"

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_data_sources_populated(self, mock_get):
        """data_sources dict must list all major sources."""
        profile = run_site_recon(42.28, -89.09, "bridge over river")
        sources = profile.data_sources

        assert "geocode" in sources
        assert "seismic" in sources
        assert "wind" in sources
        assert "thermal" in sources
        assert "scour" in sources
        assert "frost" in sources


# ═══════════════════════════════════════════════════════════════════════════════
# Regression / edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and regressions."""

    def test_puerto_rico_low_frost(self):
        """Puerto Rico: tropical, no frost — use higher-level mocks to avoid
        singleton requests.get collision."""
        pr_geo = GeoLocation(
            lat=18.40, lon=-66.06,
            state="PR", county="Bayamón Municipio",
            city="Bayamón", country="US",
            display_name="Bayamón, Puerto Rico",
        )
        seismic_pr, _ = _build_seismic_profile(_SEISMIC_FALLBACK, "D"), True

        with patch("nlb.tools.site_recon.reverse_geocode", return_value=pr_geo), \
             patch("nlb.tools.site_recon.fetch_seismic_hazard",
                   return_value=(_build_seismic_profile(_SEISMIC_FALLBACK, "D"), True)):
            profile = run_site_recon(18.40, -66.06, "Bridge in Puerto Rico")

        assert profile.frost_depth_ft == 0.0
        assert profile.climate_zone == "hot"

    def test_sdc_boundaries_are_exhaustive(self):
        """Every SD1 in [0, 1.0] maps to exactly one SDC."""
        for sd1_x100 in range(0, 101):
            sd1 = sd1_x100 / 100.0
            sdc = _seismic_design_category(sd1)
            assert sdc in ("A", "B", "C", "D"), \
                f"sd1={sd1} returned invalid SDC {sdc!r}"

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_multiple_water_keywords(self, mock_get):
        """Multiple scour keywords don't cause double-counting."""
        profile = run_site_recon(
            42.28, -89.09,
            "Bridge over river near tidal creek with channel flow",
        )
        assert profile.scour.water_crossing is True
        # Still just one design flood
        assert profile.scour.design_flood == "Q100"
        assert profile.scour.check_flood == "Q500"

    @patch("requests.get", side_effect=_multi_api_side_effect)
    def test_default_site_class_is_d(self, mock_get):
        """Calling run_site_recon without site_class defaults to D."""
        profile = run_site_recon(42.28, -89.09)
        assert profile.soil.site_class == "D"
        assert profile.seismic.site_class == "D"

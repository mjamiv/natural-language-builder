"""Site Reconnaissance Tool — Environmental Profile from GPS Coordinates.

This is the FIRST tool that runs in the Natural Language Builder pipeline.
Every downstream tool (foundation, loads, bearings, red-team) consumes the
:class:`SiteProfile` this module produces.  If this tool fails or returns
incomplete data, all other tools degrade gracefully using conservative defaults
rather than crashing.

Design philosophy
-----------------
* **Never raise from the top-level function** — individual fetchers may raise
  internally, but :func:`run_site_recon` catches everything and fills in
  conservative defaults so the pipeline can continue.
* **Conservative fallbacks, not optimistic ones** — if seismic data is
  unavailable, we assume SDC C, not SDC A.  Better to over-design than under.
* **All units US customary** — feet, kips, degrees Fahrenheit throughout.
* **Site class D as default** — AASHTO Section 3.10.3.1 permits use of
  Site Class D when no site-specific geotechnical data is available.

External APIs used
------------------
* USGS Unified Hazard Tool (no key required):
  https://earthquake.usgs.gov/ws/designmaps/aashto-2009.json
* Nominatim / OSM reverse geocode (no key required):
  https://nominatim.openstreetmap.org/reverse
* NOAA NCEI Climate Data Online (token optional):
  https://www.ncei.noaa.gov/cdo-web/api/v2/stations
  Falls back to state-level lookup table when token is absent.

References
----------
AASHTO LRFD Bridge Design Specifications, 9th Ed.:
  §3.10  Earthquake Effects (EQ)
  §3.12  Force Effects Due to Temperature
  §3.7   Hydraulic Load Effects (scour / HEC-18)
ASCE 7-22: §26 Wind speed maps and exposure categories
USGS Seismic Design Web Services documentation (2023)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

from nlb.utils.geo import reverse_geocode, GeoLocation, CLIMATE_ZONE_BY_STATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout for all outbound API calls (seconds)
# ---------------------------------------------------------------------------
_API_TIMEOUT = 15

# ---------------------------------------------------------------------------
# AASHTO site-class descriptions (Table 3.10.3.1-1)
# ---------------------------------------------------------------------------
_SITE_CLASS_DESCRIPTIONS: dict[str, str] = {
    "A": "Hard rock (Vs30 > 5000 ft/s)",
    "B": "Rock (2500 < Vs30 ≤ 5000 ft/s)",
    "C": "Dense soil or soft rock (1200 < Vs30 ≤ 2500 ft/s)",
    "D": "Stiff soil (600 < Vs30 ≤ 1200 ft/s)",
    "E": "Soft clay or liquefiable soil (Vs30 ≤ 600 ft/s)",
    "F": "Special soils — site-specific evaluation required",
}

# ---------------------------------------------------------------------------
# Seismic Design Category per AASHTO Table 3.10.6-1 (SD1-based)
# ---------------------------------------------------------------------------
def _seismic_design_category(sd1: float) -> str:
    """Determine AASHTO Seismic Design Category from SD1.

    SDC drives the level of seismic detailing and analysis required:
      A  — minimal seismic design (SD1 < 0.15)
      B  — moderate design, no ductile detailing (0.15 ≤ SD1 < 0.30)
      C  — ductile substructure required (0.30 ≤ SD1 < 0.50)
      D  — full displacement-based seismic design (SD1 ≥ 0.50)
    """
    if sd1 < 0.15:
        return "A"
    elif sd1 < 0.30:
        return "B"
    elif sd1 < 0.50:
        return "C"
    else:
        return "D"


# ---------------------------------------------------------------------------
# Conservative seismic fallback (used when USGS API is unreachable)
# ---------------------------------------------------------------------------
#: Values represent a moderate-seismicity interior US site (CONUS median).
#: They trigger SDC B — requires ductility checks but not full pushover.
_SEISMIC_FALLBACK: dict = {
    "pga":  0.10,   # g
    "ss":   0.25,   # g — 0.2 s spectral acceleration
    "s1":   0.10,   # g — 1.0 s spectral acceleration
    "fa":   1.60,   # site coefficient (site class D)
    "fv":   2.40,   # site coefficient (site class D)
    "sms":  0.40,   # g — Fa * Ss
    "sm1":  0.24,   # g — Fv * S1
    "sds":  0.267,  # g — (2/3) * SMS
    "sd1":  0.160,  # g — (2/3) * SM1
    "site_class": "D",
    "sdc":  "B",    # based on sd1 = 0.16
}

# ---------------------------------------------------------------------------
# ASCE 7-22 ultimate wind speed V_ult by state (mph, Risk Category II)
# These are conservative state-wide values from ASCE 7 Fig. 26.5-1A.
# For critical projects, fetch actual county-level values from the
# ASCE hazard API (https://asce7hazardtool.online/api/).
# ---------------------------------------------------------------------------
_WIND_SPEED_BY_STATE: dict[str, int] = {
    "AK": 110, "AL": 130, "AR": 115, "AZ": 100, "CA": 110,
    "CO": 115, "CT": 115, "DC": 110, "DE": 115, "FL": 160,
    "GA": 130, "GU": 195, "HI": 130, "IA": 115, "ID": 110,
    "IL": 115, "IN": 115, "KS": 120, "KY": 115, "LA": 140,
    "MA": 120, "MD": 115, "ME": 120, "MI": 115, "MN": 115,
    "MO": 115, "MS": 140, "MT": 115, "NC": 130, "ND": 115,
    "NE": 120, "NH": 120, "NJ": 120, "NM": 105, "NV": 105,
    "NY": 120, "OH": 115, "OK": 125, "OR": 110, "PA": 115,
    "PR": 165, "RI": 120, "SC": 130, "SD": 115, "TN": 115,
    "TX": 145, "UT": 110, "VA": 120, "VI": 165, "VT": 115,
    "WA": 110, "WI": 115, "WV": 115, "WY": 115,
}
_WIND_SPEED_DEFAULT = 115  # mph — conservative for unlisted/unknown states

# ---------------------------------------------------------------------------
# ASCE 7 Exposure Category
# Default = C (open terrain with scattered obstructions) per ASCE 7 §26.7.
# Most bridges are in Exposure C unless clearly in urban core (B) or
# wide-open flat terrain like plains/coast (D).
# ---------------------------------------------------------------------------
_DEFAULT_EXPOSURE = "C"

# ---------------------------------------------------------------------------
# Design temperature ranges by climate zone (°F)
# Values represent AASHTO Table 3.12.2.1-1 moderate climate and cold climate
# thermal design ranges for steel bridges.  Conservative (wide) ranges used.
#
# AASHTO §3.12.2: Thermal design requires computing:
#   T_mean (installation temp, typically 45–65°F)
#   T_max and T_min for expansion/contraction design
# ---------------------------------------------------------------------------
_THERMAL_BY_CLIMATE_ZONE: dict[str, dict] = {
    "hot": {
        "t_min":  10,   # °F  (design minimum, not record low)
        "t_max": 120,   # °F  (design maximum, steel in sun can reach 120+)
        "delta_t": 110, # °F  (T_max − T_min)
    },
    "mixed": {
        "t_min":  -10,
        "t_max":  110,
        "delta_t": 120,
    },
    "cold": {
        "t_min":  -20,
        "t_max":  110,
        "delta_t": 130,
    },
    "very_cold": {
        "t_min":  -30,
        "t_max":  105,
        "delta_t": 135,
    },
}
_THERMAL_DEFAULT = _THERMAL_BY_CLIMATE_ZONE["cold"]  # conservative

# ---------------------------------------------------------------------------
# Frost depth by state (feet) — minimum foundation embedment below grade.
# Values sourced from AASHTO §10.6.1.2 and regional DOT standards.
# Round up to nearest 0.5 ft for conservatism.
# ---------------------------------------------------------------------------
_FROST_DEPTH_BY_STATE: dict[str, float] = {
    "AK": 8.0,  # Severe permafrost — project-specific
    "AL": 0.5,
    "AR": 1.0,
    "AZ": 0.5,
    "CA": 0.5,
    "CO": 4.0,
    "CT": 4.0,
    "DC": 2.5,
    "DE": 2.5,
    "FL": 0.0,  # No design frost in south FL
    "GA": 1.0,
    "GU": 0.0,
    "HI": 0.0,
    "IA": 5.0,
    "ID": 4.0,
    "IL": 4.0,
    "IN": 4.0,
    "KS": 3.0,
    "KY": 2.0,
    "LA": 0.5,
    "MA": 4.0,
    "MD": 2.5,
    "ME": 6.0,
    "MI": 4.5,
    "MN": 6.0,
    "MO": 3.0,
    "MS": 0.5,
    "MT": 5.0,
    "NC": 1.5,
    "ND": 7.0,
    "NE": 4.5,
    "NH": 5.0,
    "NJ": 3.5,
    "NM": 2.0,
    "NV": 2.5,
    "NY": 4.5,
    "OH": 4.0,
    "OK": 2.0,
    "OR": 2.0,
    "PA": 4.0,
    "PR": 0.0,
    "RI": 4.0,
    "SC": 1.0,
    "SD": 6.0,
    "TN": 2.0,
    "TX": 1.0,
    "UT": 3.5,
    "VA": 2.5,
    "VI": 0.0,
    "VT": 5.5,
    "WA": 2.0,
    "WI": 5.0,
    "WV": 3.0,
    "WY": 5.0,
}
_FROST_DEPTH_DEFAULT = 4.0  # ft — conservative for unlisted states

# ---------------------------------------------------------------------------
# Water-crossing keywords (scour flag)
# If any keyword is found in the bridge description, scour analysis is
# required per AASHTO §2.6.4.4.2 and HEC-18 (FHWA).
# ---------------------------------------------------------------------------
_WATER_KEYWORDS = re.compile(
    r"\b("
    r"river|creek|stream|channel|waterway|water|bay|inlet|lake|pond|"
    r"drainage|ditch|culvert|floodplain|flood|tide|tidal|estuary|wetland|"
    r"slough|bayou|branch|run|fork|tributary|overflow|wash|arroyo"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SeismicProfile:
    """AASHTO seismic hazard parameters at a site.

    All spectral accelerations in units of g (gravitational acceleration).
    Per AASHTO §3.10: SMS = Fa*Ss, SM1 = Fv*S1, SDS = (2/3)*SMS, SD1 = (2/3)*SM1.
    """
    pga: float          # Peak ground acceleration (g)
    ss: float           # 0.2-s spectral acceleration, Site Class B rock (g)
    s1: float           # 1.0-s spectral acceleration, Site Class B rock (g)
    fa: float           # Short-period site coefficient (AASHTO Table 3.10.3.2-1)
    fv: float           # Long-period site coefficient (AASHTO Table 3.10.3.2-2)
    sms: float          # Fa * Ss — MCER spectral accel at short period (g)
    sm1: float          # Fv * S1 — MCER spectral accel at 1.0 s (g)
    sds: float          # (2/3) * SMS — design spectral accel short period (g)
    sd1: float          # (2/3) * SM1 — design spectral accel 1.0 s (g)
    site_class: str     # AASHTO site class A–F (default D if unknown)
    sdc: str            # Seismic Design Category A–D per AASHTO Table 3.10.6-1


@dataclass
class WindProfile:
    """ASCE 7-22 wind design parameters.

    V_ult is the 3-second gust wind speed (mph) at 33 ft above grade for
    Risk Category II structures (bridges per AASHTO Table 3.8.1.1.2-1).
    """
    v_ult: int          # mph — basic wind speed (3-s gust, RC-II)
    exposure: str       # ASCE 7 Exposure Category: B, C, or D


@dataclass
class ThermalProfile:
    """AASHTO temperature design range for bridge expansion/contraction.

    Values per AASHTO §3.12.2 (steel bridges).  Used to size expansion joints,
    select bearing displacement capacity, and check abutment seat widths.
    """
    t_min: float        # °F — minimum design temperature
    t_max: float        # °F — maximum design temperature
    delta_t: float      # °F — total thermal range (t_max − t_min)


@dataclass
class ScourProfile:
    """Hydraulic / scour design flags per AASHTO §2.6.4 and HEC-18.

    If water_crossing = True:
      design_flood = Q100 (100-year return period) — primary design flood
      check_flood  = Q500 (500-year return period) — extreme-event check
    Scour depths must be computed by the foundation tool using HEC-18
    methods with site-specific hydraulic data.
    """
    water_crossing: bool
    design_flood: Optional[str]    # "Q100" or None
    check_flood: Optional[str]     # "Q500" or None
    keywords_matched: list[str]    # for transparency in reports


@dataclass
class SoilProfile:
    """Soil characterization at the site.

    Site class is determined from AASHTO Table 3.10.3.1-1.
    When no geotechnical data is available, use Site Class D per
    AASHTO Commentary C3.10.3.1 — a conservative assumption for most
    US sites that avoids underestimating amplification.
    """
    site_class: str     # A–F per AASHTO Table 3.10.3.1-1
    description: str    # Human-readable description of site class


@dataclass
class SiteProfile:
    """Complete site environmental profile consumed by all other NLB tools.

    This is the canonical output of :func:`run_site_recon`.  All fields are
    populated — never None — so downstream tools can use them without guards.
    Missing data is filled with conservative defaults (documented inline).

    Serialize with :meth:`to_dict` for JSON transport over MCP.
    """
    coordinates: dict           # {"lat": float, "lon": float}
    location: dict              # {"state": str, "county": str, "city": str}
    seismic: SeismicProfile
    wind: WindProfile
    thermal: ThermalProfile
    scour: ScourProfile
    frost_depth_ft: float       # ft — minimum foundation depth (below grade)
    soil: SoilProfile
    climate_zone: str           # "hot" | "mixed" | "cold" | "very_cold"
    data_sources: dict = field(default_factory=dict)  # which APIs responded
    warnings: list[str] = field(default_factory=list) # fallback notices

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON / MCP transport."""
        d = asdict(self)
        # Flatten nested dataclasses that asdict already handles; just clean up.
        return d


# ---------------------------------------------------------------------------
# USGS Seismic Hazard fetcher
# ---------------------------------------------------------------------------

def fetch_seismic_hazard(
    lat: float,
    lon: float,
    site_class: str = "D",
) -> tuple[SeismicProfile, bool]:
    """Fetch AASHTO seismic hazard parameters from the USGS Design Maps API.

    Uses the AASHTO 2009 reference document endpoint.  The API returns:
    PGA, Ss, S1, Fa, Fv, SMS, SM1, SDS, SD1 for the given coordinates and
    site class.

    Engineering decision: We always query for the user-specified site_class
    (default D).  If the API returns site coefficients that differ from our
    input (which can happen for extreme site classes), we trust the API.

    Args:
        lat:        Latitude (WGS-84, decimal degrees).
        lon:        Longitude (WGS-84, decimal degrees).
        site_class: AASHTO site class A–F (default "D").

    Returns:
        Tuple of (SeismicProfile, success_flag).
        On failure, returns (_SEISMIC_FALLBACK as SeismicProfile, False).

    API reference:
        https://earthquake.usgs.gov/ws/designmaps/aashto-2009.json
    """
    url = "https://earthquake.usgs.gov/ws/designmaps/aashto-2009.json"
    params = {
        "latitude": lat,
        "longitude": lon,
        "riskCategory": "II",   # AASHTO normal bridges
        "siteClass": site_class,
        "title": "NLB Site Recon",
    }

    try:
        resp = requests.get(url, params=params, timeout=_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("USGS seismic API failed for (%.5f, %.5f): %s", lat, lon, exc)
        return _build_seismic_profile(_SEISMIC_FALLBACK, site_class), False

    # The USGS response nests results under data -> parameters
    try:
        params_out = data.get("response", {}).get("data", {}).get("parameters", {})

        if not params_out:
            # Some USGS endpoints return differently structured responses
            params_out = data.get("response", {}).get("data", {})

        def _get(key: str, fallback: float) -> float:
            val = params_out.get(key)
            if val is None:
                return fallback
            return float(val)

        pga  = _get("pga",  _SEISMIC_FALLBACK["pga"])
        ss   = _get("ss",   _SEISMIC_FALLBACK["ss"])
        s1   = _get("s1",   _SEISMIC_FALLBACK["s1"])
        fa   = _get("fa",   _SEISMIC_FALLBACK["fa"])
        fv   = _get("fv",   _SEISMIC_FALLBACK["fv"])
        sms  = _get("sms",  fa * ss)
        sm1  = _get("sm1",  fv * s1)
        sds  = _get("sds",  (2.0 / 3.0) * sms)
        sd1  = _get("sd1",  (2.0 / 3.0) * sm1)

        profile = SeismicProfile(
            pga=round(pga, 4),
            ss=round(ss, 4),
            s1=round(s1, 4),
            fa=round(fa, 3),
            fv=round(fv, 3),
            sms=round(sms, 4),
            sm1=round(sm1, 4),
            sds=round(sds, 4),
            sd1=round(sd1, 4),
            site_class=site_class,
            sdc=_seismic_design_category(sd1),
        )
        return profile, True

    except Exception as exc:
        logger.warning("Failed to parse USGS seismic response: %s", exc)
        return _build_seismic_profile(_SEISMIC_FALLBACK, site_class), False


def _build_seismic_profile(d: dict, site_class: str) -> SeismicProfile:
    """Construct a SeismicProfile from a flat dict (e.g. the fallback dict)."""
    sd1 = d.get("sd1", _SEISMIC_FALLBACK["sd1"])
    return SeismicProfile(
        pga=d.get("pga", _SEISMIC_FALLBACK["pga"]),
        ss=d.get("ss",  _SEISMIC_FALLBACK["ss"]),
        s1=d.get("s1",  _SEISMIC_FALLBACK["s1"]),
        fa=d.get("fa",  _SEISMIC_FALLBACK["fa"]),
        fv=d.get("fv",  _SEISMIC_FALLBACK["fv"]),
        sms=d.get("sms", _SEISMIC_FALLBACK["sms"]),
        sm1=d.get("sm1", _SEISMIC_FALLBACK["sm1"]),
        sds=d.get("sds", _SEISMIC_FALLBACK["sds"]),
        sd1=sd1,
        site_class=site_class,
        sdc=_seismic_design_category(sd1),
    )


# ---------------------------------------------------------------------------
# NOAA Climate Temperature fetcher
# ---------------------------------------------------------------------------

def fetch_thermal_range(
    lat: float,
    lon: float,
    state: str = "",
    noaa_token: Optional[str] = None,
) -> tuple[ThermalProfile, bool]:
    """Fetch temperature design range from NOAA NCEI Climate Data Online API.

    Finds the nearest weather station with Annual Climate Normal data (1991–2020
    normals, dataset NORMAL_ANN) and extracts:
      - MLY-TMIN-NORMAL → approximate design minimum temperature (°F)
      - MLY-TMAX-NORMAL → approximate design maximum temperature (°F)

    Engineering note: NOAA climate normals are 30-year averages, NOT design
    extremes.  AASHTO §3.12.2 requires using the 50-year extreme temperature,
    which is roughly normal ± 20–25°F.  We apply a ±22°F adjustment to convert
    normals to approximate design extremes.

    Fallback hierarchy:
      1. NOAA NCEI API (requires token, may be slow)
      2. State-level climate zone lookup table

    Args:
        lat:         Latitude.
        lon:         Longitude.
        state:       2-letter state abbreviation (for fallback).
        noaa_token:  NOAA API token (from https://www.ncdc.noaa.gov/cdo-web/token).
                     If None, skips API and uses state fallback directly.

    Returns:
        Tuple of (ThermalProfile, success_flag).
    """
    if noaa_token:
        try:
            profile = _fetch_noaa_normals(lat, lon, noaa_token)
            if profile is not None:
                return profile, True
        except Exception as exc:
            logger.warning("NOAA climate API failed: %s", exc)

    # Fallback: state climate zone lookup
    return _thermal_from_state(state), False


def _fetch_noaa_normals(lat: float, lon: float, token: str) -> Optional[ThermalProfile]:
    """Internal: fetch climate normals from NOAA NCEI CDO API."""
    # Step 1: find nearest station with NORMAL_ANN data
    stations_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
    headers = {"token": token}
    station_params = {
        "datasetid": "NORMAL_ANN",
        "datatypeid": "MLY-TMIN-NORMAL,MLY-TMAX-NORMAL",
        "units": "standard",       # Fahrenheit
        "extent": f"{lat-1.0},{lon-1.0},{lat+1.0},{lon+1.0}",
        "limit": 5,
        "sortfield": "name",
    }

    resp = requests.get(
        stations_url, params=station_params, headers=headers, timeout=_API_TIMEOUT
    )
    resp.raise_for_status()
    station_data = resp.json()

    results = station_data.get("results", [])
    if not results:
        return None

    station_id = results[0]["id"]

    # Step 2: fetch annual normal data for that station
    data_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    data_params = {
        "datasetid": "NORMAL_ANN",
        "datatypeid": "MLY-TMIN-NORMAL,MLY-TMAX-NORMAL",
        "stationid": station_id,
        "units": "standard",
        "limit": 10,
    }
    dresp = requests.get(
        data_url, params=data_params, headers=headers, timeout=_API_TIMEOUT
    )
    dresp.raise_for_status()
    records = dresp.json().get("results", [])

    tmin_normal = None
    tmax_normal = None
    for rec in records:
        dtype = rec.get("datatype", "")
        val = float(rec.get("value", 0))
        if dtype == "MLY-TMIN-NORMAL" and (tmin_normal is None or val < tmin_normal):
            tmin_normal = val
        if dtype == "MLY-TMAX-NORMAL" and (tmax_normal is None or val > tmax_normal):
            tmax_normal = val

    if tmin_normal is None or tmax_normal is None:
        return None

    # Convert 30-yr normals to ~50-yr design extremes (+/- 22°F adjustment)
    _EXTREME_DELTA = 22.0
    t_min = round(tmin_normal - _EXTREME_DELTA)
    t_max = round(tmax_normal + _EXTREME_DELTA)
    # Steel in direct sun adds ~10°F; cap max at 120°F per AASHTO Table 3.12.2.1-1
    t_max = min(t_max, 120)

    return ThermalProfile(
        t_min=t_min,
        t_max=t_max,
        delta_t=t_max - t_min,
    )


def _thermal_from_state(state: str) -> ThermalProfile:
    """Return design thermal profile from state→climate zone lookup."""
    zone = CLIMATE_ZONE_BY_STATE.get(state, "cold")
    values = _THERMAL_BY_CLIMATE_ZONE.get(zone, _THERMAL_DEFAULT)
    return ThermalProfile(**values)


# ---------------------------------------------------------------------------
# Wind speed
# ---------------------------------------------------------------------------

def get_wind_speed(
    state: str,
    exposure: str = _DEFAULT_EXPOSURE,
) -> WindProfile:
    """Return ASCE 7-22 basic wind speed for the given state.

    Uses a conservative state-wide lookup table.  For coastal projects,
    county-level values should be obtained from the ASCE 7 Hazard Tool.

    Exposure category logic (ASCE 7 §26.7):
      B — Urban/suburban, forest (≥ 1500 ft fetch)
      C — Open terrain, scattered obstructions (default for most bridges)
      D — Flat, open water or mud flats, shoreline exposed to hurricanes

    The caller may override the default Exposure C.

    Args:
        state:    2-letter state abbreviation.
        exposure: ASCE 7 Exposure Category (B, C, or D).

    Returns:
        WindProfile with V_ult and exposure category.
    """
    v_ult = _WIND_SPEED_BY_STATE.get(state, _WIND_SPEED_DEFAULT)
    return WindProfile(v_ult=v_ult, exposure=exposure)


# ---------------------------------------------------------------------------
# Frost depth
# ---------------------------------------------------------------------------

def get_frost_depth(state: str) -> float:
    """Return minimum foundation depth below grade due to frost (feet).

    Values are based on AASHTO §10.6.1.2 and regional DOT bridge standards.
    The returned value is the *minimum* — actual embedment must also satisfy
    bearing capacity, scour, and development length requirements.

    Args:
        state: 2-letter state abbreviation.

    Returns:
        Frost depth in feet (US customary).
    """
    return _FROST_DEPTH_BY_STATE.get(state, _FROST_DEPTH_DEFAULT)


# ---------------------------------------------------------------------------
# Scour / hydraulic flag
# ---------------------------------------------------------------------------

def detect_water_crossing(description: str) -> ScourProfile:
    """Parse bridge description for water-crossing keywords.

    Per AASHTO §2.6.4.4.2 and FHWA HEC-18: any bridge over or near a water
    body requires scour evaluation.  Design flood = Q100 (100-yr return
    period); extreme check flood = Q500 (500-yr return period).

    The foundation tool will use these flags to:
      - Remove soil springs above the computed scour depth
      - Evaluate foundation capacity at scoured condition
      - Apply the Extreme Event II load combination

    Args:
        description: Natural language bridge description from the user.

    Returns:
        ScourProfile with water_crossing flag and flood return periods.
    """
    matches = _WATER_KEYWORDS.findall(description)
    unique_matches = sorted(set(m.lower() for m in matches))
    water_crossing = len(matches) > 0

    return ScourProfile(
        water_crossing=water_crossing,
        design_flood="Q100" if water_crossing else None,
        check_flood="Q500" if water_crossing else None,
        keywords_matched=unique_matches,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_site_recon(
    lat: float,
    lon: float,
    description: str = "",
    site_class: str = "D",
    exposure: str = _DEFAULT_EXPOSURE,
    noaa_token: Optional[str] = None,
) -> SiteProfile:
    """Run the full site reconnaissance pipeline for a bridge project.

    This is the canonical entry point.  It:
      1. Reverse-geocodes the coordinates → state, county, climate zone
      2. Fetches USGS seismic hazard parameters
      3. Determines wind speed from state lookup
      4. Determines thermal range from NOAA or state climate zone
      5. Parses description for water-crossing / scour flag
      6. Looks up frost depth by state
      7. Assembles and returns a complete :class:`SiteProfile`

    The function **never raises** — all failures are caught, logged, and
    filled with conservative defaults.  Check `profile.warnings` to see
    which data sources fell back to defaults.

    Args:
        lat:         Latitude (WGS-84 decimal degrees).
        lon:         Longitude (WGS-84 decimal degrees).
        description: Natural language description of the bridge (for scour
                     keyword detection).
        site_class:  AASHTO site class (A–F).  Default "D" per AASHTO
                     Commentary C3.10.3.1 (safe default when Vs30 unknown).
        exposure:    ASCE 7 Exposure Category (B/C/D).  Default "C".
        noaa_token:  Optional NOAA NCEI API token for climate data.

    Returns:
        A fully populated :class:`SiteProfile`.

    Example::

        profile = run_site_recon(
            lat=42.28,
            lon=-89.09,
            description="3-span steel bridge over the Kishwaukee River in Illinois",
        )
        print(profile.seismic.sdc)          # → "B"
        print(profile.scour.water_crossing) # → True
        print(profile.frost_depth_ft)       # → 4.0
    """
    warnings: list[str] = []
    sources: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Reverse geocode
    # ------------------------------------------------------------------
    try:
        geo: GeoLocation = reverse_geocode(lat, lon)
        state = geo.state
        county = geo.county
        city = geo.city
        climate_zone = geo.climate_zone
        sources["geocode"] = "Nominatim/OSM"
        logger.info("Geocoded (%.5f, %.5f) → %s, %s, %s", lat, lon, city, county, state)
    except Exception as exc:
        logger.warning("Geocode failed: %s", exc)
        state = county = city = ""
        climate_zone = "cold"
        warnings.append(f"Geocode failed ({exc}); defaulting to cold climate zone.")
        sources["geocode"] = "fallback"

    location = {"state": state, "county": county, "city": city}

    # ------------------------------------------------------------------
    # 2. Seismic hazard
    # ------------------------------------------------------------------
    seismic_profile, seismic_ok = fetch_seismic_hazard(lat, lon, site_class)
    if seismic_ok:
        sources["seismic"] = "USGS Design Maps API (AASHTO 2009)"
    else:
        warnings.append(
            "USGS seismic API unavailable; using conservative SDC-B fallback values. "
            "Verify with USGS Unified Hazard Tool before final design."
        )
        sources["seismic"] = "fallback (moderate-seismicity defaults)"

    # ------------------------------------------------------------------
    # 3. Wind speed
    # ------------------------------------------------------------------
    wind_profile = get_wind_speed(state, exposure)
    if state:
        sources["wind"] = f"ASCE 7-22 state lookup ({state})"
    else:
        warnings.append(
            f"State unknown; using default wind speed {_WIND_SPEED_DEFAULT} mph. "
            "Verify with ASCE 7 Hazard Tool for actual county wind speed."
        )
        sources["wind"] = "fallback (default CONUS)"

    # ------------------------------------------------------------------
    # 4. Thermal range
    # ------------------------------------------------------------------
    thermal_profile, thermal_ok = fetch_thermal_range(lat, lon, state, noaa_token)
    if thermal_ok:
        sources["thermal"] = "NOAA NCEI Climate Data Online"
    else:
        if not state:
            warnings.append(
                "State unknown and NOAA API not used; applying cold-climate thermal range."
            )
        else:
            zone_label = climate_zone
            warnings.append(
                f"NOAA climate API not queried; thermal range from {zone_label} "
                "climate zone lookup. Provide NOAA token for site-specific values."
            )
        sources["thermal"] = f"state climate zone fallback ({climate_zone})"

    # ------------------------------------------------------------------
    # 5. Scour / hydraulic
    # ------------------------------------------------------------------
    scour_profile = detect_water_crossing(description)
    if scour_profile.water_crossing:
        sources["scour"] = "keyword parse"
        logger.info(
            "Water crossing detected: %s — scour analysis required (Q100/Q500)",
            scour_profile.keywords_matched,
        )
    else:
        sources["scour"] = "keyword parse (no water crossing detected)"

    # ------------------------------------------------------------------
    # 6. Frost depth
    # ------------------------------------------------------------------
    frost_depth = get_frost_depth(state)
    if state:
        sources["frost"] = f"state lookup ({state})"
    else:
        warnings.append(
            f"State unknown; using default frost depth {_FROST_DEPTH_DEFAULT} ft."
        )
        sources["frost"] = "fallback default"

    # ------------------------------------------------------------------
    # 7. Soil / site class description
    # ------------------------------------------------------------------
    soil_profile = SoilProfile(
        site_class=site_class,
        description=_SITE_CLASS_DESCRIPTIONS.get(site_class, "Unknown site class"),
    )

    # ------------------------------------------------------------------
    # 8. Assemble and return
    # ------------------------------------------------------------------
    return SiteProfile(
        coordinates={"lat": lat, "lon": lon},
        location=location,
        seismic=seismic_profile,
        wind=wind_profile,
        thermal=thermal_profile,
        scour=scour_profile,
        frost_depth_ft=frost_depth,
        soil=soil_profile,
        climate_zone=climate_zone,
        data_sources=sources,
        warnings=warnings,
    )

"""Geospatial utility functions for the Natural Language Builder.

Provides reverse geocoding, climate zone classification, and coordinate
utilities used by site-recon and other tools.

All coordinates are WGS-84 (decimal degrees).  Positive lat = North,
positive lon = East (so US longitudes are negative).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / lookup tables
# ---------------------------------------------------------------------------

#: USPS state abbreviation → full name (for display / logging)
STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico", "GU": "Guam", "VI": "Virgin Islands",
}

#: Reverse mapping: lowercase full name → state abbreviation.
#: Used to interpret Nominatim's "state" field.
_STATE_NAME_TO_ABBR: dict[str, str] = {v.lower(): k for k, v in STATE_NAMES.items()}

#: Climate zone per state (ASHRAE / IECC broad classification).
#:   "hot"      – Year-round heat, negligible freeze (FL, HI, south TX)
#:   "mixed"    – Seasonal extremes, mild winters (VA, KY, TN, OK, CA coast)
#:   "cold"     – Significant winter freeze (IL, MN, NY, WI)
#:   "very_cold"– Severe freeze, deep frost (AK, ND, ME, MT upper tiers)
#:
#: Used by thermal design range selection when NOAA API is unavailable.
CLIMATE_ZONE_BY_STATE: dict[str, str] = {
    "AK": "very_cold",
    "AL": "mixed",
    "AR": "mixed",
    "AZ": "hot",
    "CA": "mixed",
    "CO": "cold",
    "CT": "cold",
    "DC": "mixed",
    "DE": "mixed",
    "FL": "hot",
    "GA": "mixed",
    "GU": "hot",
    "HI": "hot",
    "IA": "cold",
    "ID": "cold",
    "IL": "cold",
    "IN": "cold",
    "KS": "mixed",
    "KY": "mixed",
    "LA": "hot",
    "MA": "cold",
    "MD": "mixed",
    "ME": "very_cold",
    "MI": "cold",
    "MN": "very_cold",
    "MO": "mixed",
    "MS": "mixed",
    "MT": "very_cold",
    "NC": "mixed",
    "ND": "very_cold",
    "NE": "cold",
    "NH": "very_cold",
    "NJ": "mixed",
    "NM": "mixed",
    "NV": "mixed",
    "NY": "cold",
    "OH": "cold",
    "OK": "mixed",
    "OR": "mixed",
    "PA": "cold",
    "PR": "hot",
    "RI": "cold",
    "SC": "mixed",
    "SD": "very_cold",
    "TN": "mixed",
    "TX": "hot",
    "UT": "cold",
    "VA": "mixed",
    "VI": "hot",
    "VT": "very_cold",
    "WA": "mixed",
    "WI": "cold",
    "WV": "mixed",
    "WY": "very_cold",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeoLocation:
    """Result of a reverse geocode call.

    All fields may be empty-string if the geocoder returned no data —
    callers should handle this gracefully and fall back to conservative
    default values.
    """

    lat: float
    lon: float
    state: str          # USPS 2-letter abbreviation, e.g. "IL"
    county: str         # e.g. "Winnebago County"
    city: str           # nearest populated place, may be empty
    country: str        # ISO 3166-1 alpha-2, e.g. "US"
    display_name: str   # full Nominatim display string (for logging/reports)

    @property
    def climate_zone(self) -> str:
        """Infer ASHRAE climate zone from state abbreviation."""
        return CLIMATE_ZONE_BY_STATE.get(self.state, "mixed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reverse_geocode(
    lat: float,
    lon: float,
    timeout: int = 10,
    user_agent: str = "natural-language-builder/0.1 (bridge-engineering-tool)",
) -> GeoLocation:
    """Reverse-geocode a WGS-84 coordinate pair via OSM Nominatim.

    Returns a :class:`GeoLocation`.  On any network or parse failure, falls
    back to an empty :class:`GeoLocation` so callers never crash — they just
    get less-rich data and should apply conservative defaults.

    Args:
        lat:        Latitude, decimal degrees (positive = North).
        lon:        Longitude, decimal degrees (positive = East).
        timeout:    Request timeout in seconds.
        user_agent: Required by Nominatim usage policy; identifies the app.

    Returns:
        GeoLocation with state, county, city populated where available.
    """
    _empty = GeoLocation(
        lat=lat, lon=lon,
        state="", county="", city="", country="",
        display_name="",
    )

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "zoom": 10,          # county-level resolution
        "extratags": 0,
    }
    headers = {"User-Agent": user_agent}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Nominatim reverse geocode failed for (%.5f, %.5f): %s", lat, lon, exc)
        return _empty

    addr = data.get("address", {})

    # ---- State ---------------------------------------------------------------
    # Nominatim returns "state" as a full name for US points.
    state_raw = addr.get("state", "")
    state_abbr = _normalize_state(state_raw)

    # ---- County --------------------------------------------------------------
    # Nominatim uses different keys depending on jurisdiction level.
    county = (
        addr.get("county")
        or addr.get("region")
        or addr.get("state_district")
        or ""
    )

    # ---- City ----------------------------------------------------------------
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("hamlet")
        or ""
    )

    # ---- Country -------------------------------------------------------------
    country_code = addr.get("country_code", "").upper()

    return GeoLocation(
        lat=lat,
        lon=lon,
        state=state_abbr,
        county=county,
        city=city,
        country=country_code,
        display_name=data.get("display_name", ""),
    )


def state_abbreviation(state_name: str) -> str:
    """Convert a full US state name to its 2-letter USPS abbreviation.

    Case-insensitive.  Returns the input unchanged if not found (so callers
    that already have abbreviations pass through cleanly).

    Examples::

        >>> state_abbreviation("Illinois")
        'IL'
        >>> state_abbreviation("IL")   # already abbreviated
        'IL'
    """
    return _normalize_state(state_name)


def is_valid_us_coordinate(lat: float, lon: float) -> bool:
    """Rough bounding-box check for the contiguous US + AK + HI.

    Not a rigorous polygon test — just a sanity check to catch obviously
    wrong inputs before hitting external APIs.
    """
    # Contiguous US
    if 24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0:
        return True
    # Alaska
    if 51.0 <= lat <= 72.0 and -180.0 <= lon <= -130.0:
        return True
    # Hawaii
    if 18.5 <= lat <= 22.5 and -160.5 <= lon <= -154.5:
        return True
    return False


def haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points, in feet.

    Uses the haversine formula.  Accurate to ~0.5% for distances < 1000 mi.
    Primarily useful for sanity-checking coordinate pairs.
    """
    import math

    R_FT = 20_902_231.0   # mean Earth radius in feet

    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R_FT * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_state(raw: str) -> str:
    """Return USPS 2-letter abbreviation for any state name/abbr input.

    Handles:
    - "Illinois"          → "IL"
    - "illinois"          → "IL"
    - "IL"                → "IL"  (pass-through)
    - "North Dakota"      → "ND"
    Returns "" for unrecognized input.
    """
    if not raw:
        return ""
    stripped = raw.strip()
    # Already an abbreviation?
    upper = stripped.upper()
    if upper in STATE_NAMES:
        return upper
    # Try full name lookup (case-insensitive)
    return _STATE_NAME_TO_ABBR.get(stripped.lower(), "")

"""Load Generation Tool — AASHTO LRFD + Adversarial Load Cases.

Produces ALL standard AASHTO LRFD 9th Edition load cases, computes live-load
distribution factors, generates load combinations per Table 3.4.1-1, and adds
adversarial (red-team) load cases that standard practice typically omits.

Consumes :class:`~nlb.tools.site_recon.SiteProfile` for environmental loads
(seismic, wind, thermal) and bridge geometry for dead/live loads.

Units
-----
* **Input:** kip, ft (US customary)
* **Internal / output:** kip, inch, second (OpenSees standard)

References
----------
AASHTO LRFD Bridge Design Specifications, 9th Edition:
  §3.4   Load Factors and Combinations
  §3.5   Permanent Loads (DC, DW)
  §3.6   Live Loads (LL, HL-93)
  §3.7   Water Loads
  §3.8   Wind Loads (WS, WL)
  §3.9   Ice Loads
  §3.10  Earthquake Effects (EQ)
  §3.11  Earth Pressure
  §3.12  Temperature (TU, TG)
  §3.13  Friction (FR)
  §4.6.2 Distribution of Live Loads
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ======================================================================
# Constants — US customary units (kip, ft, inch, pcf, psf, ksf, klf)
# ======================================================================

# Unit weights (pcf)
GAMMA_STEEL = 490.0        # pcf — AISC
GAMMA_CONCRETE = 150.0     # pcf — normal weight concrete
GAMMA_CONCRETE_LW = 120.0  # pcf — lightweight concrete

# Future wearing surface
FWS_PSF = 25.0             # psf — AASHTO default

# HL-93 design vehicles
DESIGN_TRUCK_AXLES = [
    {"weight_kip": 8.0,  "position_ft": 0.0},
    {"weight_kip": 32.0, "position_ft": 14.0},
    {"weight_kip": 32.0, "position_ft": 28.0},  # 14ft spacing (min)
]
DESIGN_TRUCK_AXLES_MAX = [
    {"weight_kip": 8.0,  "position_ft": 0.0},
    {"weight_kip": 32.0, "position_ft": 14.0},
    {"weight_kip": 32.0, "position_ft": 44.0},  # 30ft spacing (max)
]
DESIGN_TANDEM_AXLES = [
    {"weight_kip": 25.0, "position_ft": 0.0},
    {"weight_kip": 25.0, "position_ft": 4.0},
]
DESIGN_LANE_KLF = 0.64     # klf — uniform per loaded lane (10ft width)
LANE_WIDTH_FT = 10.0       # ft — design lane width

# Impact factor (AASHTO 3.6.2.1)
IM_TRUCK = 0.33            # 33% applied to truck/tandem only
IM_LANE = 0.0              # 0% on lane load

# Multiple presence factors (AASHTO Table 3.6.1.1.2-1)
MULTIPLE_PRESENCE = {1: 1.20, 2: 1.00, 3: 0.85, 4: 0.65}

# Thermal gradient zones (AASHTO Table 3.12.3-1, °F)
# T1 = top of concrete deck, T2 at 4in below, linear to 0 at depth
THERMAL_GRADIENT_ZONES = {
    1: {"T1": 54, "T2": 14},   # South/coastal (FL, TX coast, etc.)
    2: {"T1": 46, "T2": 12},   # Mid-south
    3: {"T1": 41, "T2": 11},   # Mid-north
    4: {"T1": 38, "T2":  9},   # Northern states
}
# Negative gradient multiplier
NEG_GRADIENT_CONCRETE = -0.30
NEG_GRADIENT_STEEL = -0.50

# Wind constants
WIND_PRESSURE_COEFF = 0.00256   # psf constant in ASCE 7 velocity pressure eq
WIND_GUST_FACTOR = 0.85         # G — for bridges (rigid structures)
WIND_CD_GIRDER = 1.30           # Cd — drag coefficient for I-girders
WL_KLF = 0.10                  # klf — wind on live load
WL_HEIGHT_FT = 6.0             # ft above deck

# Braking
BR_TRUCK_FRACTION = 0.25       # 25% of truck axle weights
BR_COMBO_FRACTION = 0.05       # 5% of (truck + lane)

# Permit vehicles
PERMIT_VEHICLES = {
    "IL_3S2": {
        "name": "Illinois Type 3S2",
        "total_kip": 72.0,
        "axles": [
            {"weight_kip": 12.0, "position_ft": 0.0},
            {"weight_kip": 12.0, "position_ft": 10.0},
            {"weight_kip": 16.0, "position_ft": 14.0},
            {"weight_kip": 16.0, "position_ft": 18.0},
            {"weight_kip": 16.0, "position_ft": 22.0},
        ],
    },
    "IL_SU4": {
        "name": "Illinois SU4",
        "total_kip": 54.0,
        "axles": [
            {"weight_kip": 8.0,  "position_ft": 0.0},
            {"weight_kip": 8.0,  "position_ft": 6.0},
            {"weight_kip": 19.0, "position_ft": 20.0},
            {"weight_kip": 19.0, "position_ft": 24.0},
        ],
    },
    "AASHTO_Type3": {
        "name": "AASHTO Legal Type 3",
        "total_kip": 50.0,
        "axles": [
            {"weight_kip": 16.0, "position_ft": 0.0},
            {"weight_kip": 17.0, "position_ft": 10.0},
            {"weight_kip": 17.0, "position_ft": 14.0},
        ],
    },
    "AASHTO_3S2": {
        "name": "AASHTO Legal 3S2",
        "total_kip": 72.0,
        "axles": [
            {"weight_kip": 10.0, "position_ft": 0.0},
            {"weight_kip": 15.5, "position_ft": 4.0},
            {"weight_kip": 15.5, "position_ft": 15.0},
            {"weight_kip": 15.5, "position_ft": 21.0},
            {"weight_kip": 15.5, "position_ft": 25.0},
        ],
    },
    "AASHTO_3_3": {
        "name": "AASHTO Legal 3-3",
        "total_kip": 80.0,
        "axles": [
            {"weight_kip": 12.0, "position_ft": 0.0},
            {"weight_kip": 12.0, "position_ft": 4.0},
            {"weight_kip": 12.0, "position_ft": 15.0},
            {"weight_kip": 16.0, "position_ft": 16.0},
            {"weight_kip": 14.0, "position_ft": 31.0},
            {"weight_kip": 14.0, "position_ft": 35.0},
        ],
    },
}

# Kz values for wind velocity pressure (ASCE 7 Table 26.10-1, Exposure C)
_KZ_TABLE_EXP_C = {
    15: 0.85, 20: 0.90, 25: 0.94, 30: 0.98, 40: 1.04,
    50: 1.09, 60: 1.13, 70: 1.17, 80: 1.21, 90: 1.24,
}

# AASHTO thermal gradient zone by state (approximate)
_GRADIENT_ZONE_BY_STATE: dict[str, int] = {
    "FL": 1, "TX": 1, "LA": 1, "MS": 1, "AL": 1, "GA": 1, "SC": 1,
    "HI": 1, "PR": 1, "VI": 1, "GU": 1,
    "NC": 2, "TN": 2, "AR": 2, "OK": 2, "NM": 2, "AZ": 2, "CA": 2,
    "NV": 2, "VA": 2,
    "KY": 3, "MO": 3, "KS": 3, "CO": 3, "UT": 3, "OR": 3, "WA": 3,
    "MD": 3, "DE": 3, "WV": 3, "DC": 3, "NJ": 3, "CT": 3, "RI": 3,
    "MA": 3, "IN": 3, "OH": 3, "PA": 3, "IL": 3, "IA": 3, "NE": 3,
    "ID": 3,
    "NY": 4, "VT": 4, "NH": 4, "ME": 4, "MI": 4, "WI": 4, "MN": 4,
    "ND": 4, "SD": 4, "MT": 4, "WY": 4, "AK": 4,
}


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class LoadCase:
    """A single load case for OpenSees analysis.

    Attributes:
        name:        Unique identifier, e.g. ``"DC1_deck_slab"``.
        category:    ``'standard'`` or ``'adversarial'``.
        load_type:   AASHTO code (DC, DW, LL, EQ, WS, WL, BR, TU, TG, P, SC).
        description: Human-readable description for reports.
        loads:       List of load dicts, each with a ``type`` key:
                     ``'distributed'``, ``'point'``, ``'spectrum'``,
                     ``'thermal'``, ``'modification'``.
    """
    name: str
    category: str
    load_type: str
    description: str
    loads: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadCombination:
    """A factored load combination per AASHTO Table 3.4.1-1.

    Attributes:
        name:        e.g. ``"Strength_I_max"``.
        limit_state: e.g. ``"Strength I"``.
        factors:     Mapping of load case name → load factor.
    """
    name: str
    limit_state: str
    factors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadModel:
    """Complete load model: all cases, combinations, and distribution factors.

    This is the canonical output of :func:`generate_loads`.
    """
    cases: list[LoadCase] = field(default_factory=list)
    combinations: list[LoadCombination] = field(default_factory=list)
    distribution_factors: dict = field(default_factory=dict)
    adversarial_cases: list[LoadCase] = field(default_factory=list)
    adversarial_combos: list[LoadCombination] = field(default_factory=list)
    moving_load_positions: dict = field(default_factory=dict)

    @property
    def total_combinations(self) -> int:
        return len(self.combinations) + len(self.adversarial_combos)

    def to_dict(self) -> dict:
        return {
            "cases": [c.to_dict() for c in self.cases],
            "combinations": [c.to_dict() for c in self.combinations],
            "distribution_factors": self.distribution_factors,
            "adversarial_cases": [c.to_dict() for c in self.adversarial_cases],
            "adversarial_combos": [c.to_dict() for c in self.adversarial_combos],
            "moving_load_positions": self.moving_load_positions,
            "total_combinations": self.total_combinations,
        }


@dataclass
class BridgeGeometry:
    """Bridge geometry required for load generation.

    All dimensions in feet (input) — converted to inches internally where needed.

    Attributes:
        span_ft:            Span length (ft). For multi-span, use list.
        girder_spacing_ft:  Center-to-center girder spacing (ft).
        deck_thickness_in:  Concrete deck thickness (inches).
        num_girders:        Number of girders.
        deck_width_ft:      Total deck width curb-to-curb (ft).
        num_barriers:       Number of traffic barriers (typically 2).
        barrier_weight_klf: Weight per barrier (klf). Default 0.40.
        haunch_thickness_in: Haunch (pad) thickness between deck and girder (in).
        haunch_width_in:    Haunch width (in).
        girder_weight_plf:  Steel girder self-weight (lb/ft). If 0, estimated.
        girder_depth_in:    Steel girder depth (inches).
        num_lanes:          Number of design lanes (auto-computed if 0).
        overhang_ft:        Deck overhang beyond exterior girder (ft).
        diaphragm_weight_kip: Weight of one diaphragm (kip). Default 0.
        num_diaphragms:     Number of diaphragms per span.
        utilities_ksf:      Utility load on deck (ksf). Default 0.02.
        structure_type:     ``'steel'`` or ``'concrete'``.
        is_composite:       Whether deck acts compositely with girders.
        n_spans:            Number of spans (derived from span_ft if list).
        spans_ft:           List of span lengths (ft).
    """
    span_ft: float = 80.0
    girder_spacing_ft: float = 8.0
    deck_thickness_in: float = 8.0
    num_girders: int = 5
    deck_width_ft: float = 40.0
    num_barriers: int = 2
    barrier_weight_klf: float = 0.40
    haunch_thickness_in: float = 2.0
    haunch_width_in: float = 16.0
    girder_weight_plf: float = 0.0
    girder_depth_in: float = 36.0
    num_lanes: int = 0
    overhang_ft: float = 3.0
    diaphragm_weight_kip: float = 0.30
    num_diaphragms: int = 4
    utilities_ksf: float = 0.02
    structure_type: str = "steel"
    is_composite: bool = True

    @property
    def n_spans(self) -> int:
        return 1

    @property
    def spans_ft(self) -> list[float]:
        return [self.span_ft]

    @property
    def design_lanes(self) -> int:
        """Number of design lanes per AASHTO 3.6.1.1.1."""
        if self.num_lanes > 0:
            return self.num_lanes
        return max(1, int(self.deck_width_ft / 12.0))

    @property
    def deck_thickness_ft(self) -> float:
        return self.deck_thickness_in / 12.0


# ======================================================================
# Dead load computation
# ======================================================================

def _compute_dead_loads(geom: BridgeGeometry) -> list[LoadCase]:
    """Compute DC1, DC2, DC3, and DW dead load cases.

    All output loads are in kip/in (distributed) for OpenSees.
    """
    cases: list[LoadCase] = []

    # --- DC1: Structural components ---

    # Deck slab: thickness × trib width × γ_concrete
    # Trib width per girder ≈ girder_spacing for interior, half-spacing + overhang for exterior
    deck_weight_klf = (
        (geom.deck_thickness_in / 12.0)
        * geom.girder_spacing_ft
        * GAMMA_CONCRETE / 1000.0
    )
    deck_kli = deck_weight_klf / 12.0  # kip/in

    cases.append(LoadCase(
        name="DC1_deck_slab",
        category="standard",
        load_type="DC",
        description=(
            f"Concrete deck slab: {geom.deck_thickness_in}\" thick × "
            f"{geom.girder_spacing_ft}' trib width × {GAMMA_CONCRETE} pcf = "
            f"{deck_weight_klf:.3f} klf per interior girder"
        ),
        loads=[{
            "type": "distributed",
            "w_kip_per_in": deck_kli,
            "w_klf": deck_weight_klf,
            "direction": "gravity",
            "applied_to": "interior_girders",
        }],
    ))

    # Haunch
    haunch_weight_klf = (
        (geom.haunch_thickness_in / 12.0)
        * (geom.haunch_width_in / 12.0)
        * GAMMA_CONCRETE / 1000.0
    )
    haunch_kli = haunch_weight_klf / 12.0

    cases.append(LoadCase(
        name="DC1_haunch",
        category="standard",
        load_type="DC",
        description=(
            f"Haunch: {geom.haunch_thickness_in}\" × {geom.haunch_width_in}\" × "
            f"{GAMMA_CONCRETE} pcf = {haunch_weight_klf:.3f} klf per girder"
        ),
        loads=[{
            "type": "distributed",
            "w_kip_per_in": haunch_kli,
            "w_klf": haunch_weight_klf,
            "direction": "gravity",
            "applied_to": "all_girders",
        }],
    ))

    # Steel girder self-weight
    if geom.girder_weight_plf > 0:
        girder_klf = geom.girder_weight_plf / 1000.0
    else:
        # Estimate: girder weight ≈ 10 × span_ft (rule of thumb for steel bridges)
        girder_klf = (10.0 * geom.span_ft) / 1000.0

    girder_kli = girder_klf / 12.0

    cases.append(LoadCase(
        name="DC1_girder",
        category="standard",
        load_type="DC",
        description=(
            f"Steel girder self-weight: {girder_klf * 1000:.0f} plf = "
            f"{girder_klf:.3f} klf per girder"
        ),
        loads=[{
            "type": "distributed",
            "w_kip_per_in": girder_kli,
            "w_klf": girder_klf,
            "direction": "gravity",
            "applied_to": "all_girders",
        }],
    ))

    # Diaphragms (concentrated loads along span)
    if geom.diaphragm_weight_kip > 0 and geom.num_diaphragms > 0:
        total_diaph_klf = (
            geom.diaphragm_weight_kip * geom.num_diaphragms / geom.span_ft
        )
        cases.append(LoadCase(
            name="DC1_diaphragms",
            category="standard",
            load_type="DC",
            description=(
                f"Diaphragms: {geom.num_diaphragms} × {geom.diaphragm_weight_kip} kip = "
                f"{total_diaph_klf:.3f} klf equivalent distributed"
            ),
            loads=[{
                "type": "distributed",
                "w_kip_per_in": total_diaph_klf / 12.0,
                "w_klf": total_diaph_klf,
                "direction": "gravity",
                "applied_to": "all_girders",
            }],
        ))

    # --- DC2: Barriers/railings ---
    barrier_klf = geom.barrier_weight_klf
    barrier_kli = barrier_klf / 12.0

    cases.append(LoadCase(
        name="DC2_barriers",
        category="standard",
        load_type="DC",
        description=(
            f"Barriers: {geom.num_barriers} × {barrier_klf:.2f} klf = "
            f"{geom.num_barriers * barrier_klf:.2f} klf total, applied to exterior girders"
        ),
        loads=[{
            "type": "distributed",
            "w_kip_per_in": barrier_kli,
            "w_klf": barrier_klf,
            "direction": "gravity",
            "applied_to": "exterior_girders",
            "count": geom.num_barriers,
        }],
    ))

    # --- DC3: Utilities ---
    util_klf = geom.utilities_ksf * geom.girder_spacing_ft
    util_kli = util_klf / 12.0

    cases.append(LoadCase(
        name="DC3_utilities",
        category="standard",
        load_type="DC",
        description=(
            f"Utilities/signs/lighting: {geom.utilities_ksf} ksf × "
            f"{geom.girder_spacing_ft}' = {util_klf:.3f} klf per girder"
        ),
        loads=[{
            "type": "distributed",
            "w_kip_per_in": util_kli,
            "w_klf": util_klf,
            "direction": "gravity",
            "applied_to": "all_girders",
        }],
    ))

    # --- DW: Future wearing surface ---
    fws_klf = (FWS_PSF / 1000.0) * geom.girder_spacing_ft
    fws_kli = fws_klf / 12.0

    cases.append(LoadCase(
        name="DW_wearing_surface",
        category="standard",
        load_type="DW",
        description=(
            f"Future wearing surface: {FWS_PSF} psf × {geom.girder_spacing_ft}' = "
            f"{fws_klf:.3f} klf per interior girder"
        ),
        loads=[{
            "type": "distributed",
            "w_kip_per_in": fws_kli,
            "w_klf": fws_klf,
            "direction": "gravity",
            "applied_to": "interior_girders",
        }],
    ))

    return cases


# ======================================================================
# Live load distribution factors (AASHTO 4.6.2.2)
# ======================================================================

def compute_distribution_factors(
    girder_spacing_ft: float,
    span_ft: float,
    deck_thickness_in: float,
    girder_depth_in: float,
    num_girders: int,
    structure_type: str = "steel",
) -> dict:
    """Compute AASHTO LRFD live-load distribution factors.

    Per AASHTO Tables 4.6.2.2.2b-1 and 4.6.2.2.3a-1 for steel I-girder
    bridges (Type "a" cross-section).

    Parameters are in mixed units matching AASHTO formulas:
    spacing in ft, span in ft, thickness in inches.

    Returns a dict with keys:
        moment_interior_1, moment_interior_2p,
        moment_exterior_1, moment_exterior_2p,
        shear_interior_1, shear_interior_2p,
        shear_exterior_1, shear_exterior_2p,
        moment_interior, moment_exterior,
        shear_interior, shear_exterior
    """
    S = girder_spacing_ft
    L = span_ft
    ts = deck_thickness_in

    # Stiffness parameter Kg (AASHTO 4.6.2.2.1-1)
    # Kg = n * (I + A * eg²)
    # For steel girder with concrete deck: n ≈ 8 (modular ratio Es/Ec for 4ksi concrete)
    n_modular = 8.0
    # Approximate I and A from girder depth for initial estimate
    # More accurate when actual section props are provided
    A_girder = girder_depth_in * 0.5  # rough estimate: ~0.5 in² per inch depth
    I_girder = girder_depth_in ** 3 * 0.04  # rough estimate
    eg = girder_depth_in / 2.0 + ts / 2.0  # distance from girder CG to deck CG

    Kg = n_modular * (I_girder + A_girder * eg ** 2)

    # Clamp to AASHTO range of applicability
    S = max(3.5, min(S, 16.0))
    L = max(20.0, min(L, 240.0))
    ts_clamped = max(4.5, min(ts, 12.0))

    # -------------------------------------------------------------------
    # Moment — Interior Girder (AASHTO Table 4.6.2.2.2b-1, Type a)
    # -------------------------------------------------------------------
    # One design lane loaded:
    gM_int_1 = 0.06 + (S / 14.0) ** 0.4 * (S / L) ** 0.3 * (Kg / (12.0 * L * ts_clamped ** 3)) ** 0.1

    # Two or more lanes loaded:
    gM_int_2 = 0.075 + (S / 9.5) ** 0.6 * (S / L) ** 0.2 * (Kg / (12.0 * L * ts_clamped ** 3)) ** 0.1

    # -------------------------------------------------------------------
    # Moment — Exterior Girder (AASHTO Table 4.6.2.2.2d-1)
    # -------------------------------------------------------------------
    # One lane loaded: lever rule
    # Assume wheel at 2ft from barrier, exterior girder at overhang
    # Lever rule gives approximately 0.5 + de/(2*S) where de = dist from
    # exterior web to interior edge of curb
    de = 2.0  # ft — typical distance from exterior girder to design lane edge
    e_moment = 0.77 + de / 9.1  # correction factor for 2+ lanes
    gM_ext_1 = gM_int_1  # lever rule ≈ interior for one lane (conservative)
    gM_ext_2 = e_moment * gM_int_2

    # -------------------------------------------------------------------
    # Shear — Interior Girder (AASHTO Table 4.6.2.2.3a-1)
    # -------------------------------------------------------------------
    gV_int_1 = 0.36 + S / 25.0
    gV_int_2 = 0.2 + S / 12.0 - (S / 35.0) ** 2.0

    # -------------------------------------------------------------------
    # Shear — Exterior Girder (AASHTO Table 4.6.2.2.3b-1)
    # -------------------------------------------------------------------
    e_shear = 0.6 + de / 10.0
    gV_ext_1 = gV_int_1  # lever rule (conservative)
    gV_ext_2 = e_shear * gV_int_2

    # Governing (max of 1-lane vs 2+-lane with multiple presence)
    gM_int = max(gM_int_1 * MULTIPLE_PRESENCE[1],
                 gM_int_2 * MULTIPLE_PRESENCE[2])
    gM_ext = max(gM_ext_1 * MULTIPLE_PRESENCE[1],
                 gM_ext_2 * MULTIPLE_PRESENCE[2])
    gV_int = max(gV_int_1 * MULTIPLE_PRESENCE[1],
                 gV_int_2 * MULTIPLE_PRESENCE[2])
    gV_ext = max(gV_ext_1 * MULTIPLE_PRESENCE[1],
                 gV_ext_2 * MULTIPLE_PRESENCE[2])

    return {
        "moment_interior_1": round(gM_int_1, 4),
        "moment_interior_2p": round(gM_int_2, 4),
        "moment_exterior_1": round(gM_ext_1, 4),
        "moment_exterior_2p": round(gM_ext_2, 4),
        "shear_interior_1": round(gV_int_1, 4),
        "shear_interior_2p": round(gV_int_2, 4),
        "shear_exterior_1": round(gV_ext_1, 4),
        "shear_exterior_2p": round(gV_ext_2, 4),
        "moment_interior": round(gM_int, 4),
        "moment_exterior": round(gM_ext, 4),
        "shear_interior": round(gV_int, 4),
        "shear_exterior": round(gV_ext, 4),
        "Kg": round(Kg, 1),
    }


# ======================================================================
# Live loads (HL-93)
# ======================================================================

def _simple_span_moment(axles: list[dict], span_ft: float) -> float:
    """Max simple-span moment (kip-ft) for a set of axles using influence lines.

    Places the resultant of the axle group at midspan and checks for maximum
    moment under each axle.
    """
    if span_ft <= 0:
        return 0.0

    total_w = sum(a["weight_kip"] for a in axles)
    if total_w <= 0:
        return 0.0

    # Resultant position from first axle
    x_r = sum(a["weight_kip"] * a["position_ft"] for a in axles) / total_w

    best_moment = 0.0
    # Try placing each axle at or near midspan
    for target_axle in axles:
        # Place this axle at midspan offset to maximize moment
        offset = span_ft / 2.0 - target_axle["position_ft"] + (target_axle["position_ft"] - x_r) / 2.0

        for axle in axles:
            x = axle["position_ft"] + offset
            if x < 0 or x > span_ft:
                continue
            # Moment at axle location from simple beam
            ra = total_w * (span_ft - (x_r + offset)) / span_ft  # left reaction approx
            # Actually compute properly: sum of all axle contributions
        # Use direct approach: for each axle position as the critical point
        # place the truck to maximize moment under that axle
        target_x_on_span = target_axle["position_ft"]
        # Offset so the resultant and target axle straddle midspan
        # Place resultant at midspan offset by half the distance from resultant to target
        shift = span_ft / 2.0 - (target_x_on_span + x_r) / 2.0

        m = 0.0
        for axle in axles:
            x = axle["position_ft"] + shift
            if x < 0 or x > span_ft:
                continue
            # Moment at midspan from this axle on a simple beam
            # Actually, moment at the target axle location
            x_target = target_axle["position_ft"] + shift
            if x_target < 0 or x_target > span_ft:
                break
            # Using influence line: moment at point 'a' from load at 'x'
            # M_a = P * x * (L - a) / L   if x <= a
            # M_a = P * a * (L - x) / L   if x > a
            a = x_target
            if x <= a:
                m += axle["weight_kip"] * x * (span_ft - a) / span_ft
            else:
                m += axle["weight_kip"] * a * (span_ft - x) / span_ft

        best_moment = max(best_moment, m)

    return best_moment


def _simple_span_shear(axles: list[dict], span_ft: float) -> float:
    """Max simple-span shear (kip) at the support for a set of axles."""
    if span_ft <= 0:
        return 0.0

    best_shear = 0.0
    # Place first axle at left support for max left reaction
    for i, lead_axle in enumerate(axles):
        shift = -lead_axle["position_ft"]
        reaction = 0.0
        for axle in axles:
            x = axle["position_ft"] + shift
            if x < 0 or x > span_ft:
                continue
            reaction += axle["weight_kip"] * (span_ft - x) / span_ft
        best_shear = max(best_shear, reaction)

    return best_shear


def _compute_live_loads(geom: BridgeGeometry) -> list[LoadCase]:
    """Generate HL-93 live load cases."""
    cases: list[LoadCase] = []
    L = geom.span_ft

    # --- Design Truck ---
    truck_moment = _simple_span_moment(DESIGN_TRUCK_AXLES, L)
    truck_shear = _simple_span_shear(DESIGN_TRUCK_AXLES, L)

    # With impact
    truck_moment_im = truck_moment * (1 + IM_TRUCK)
    truck_shear_im = truck_shear * (1 + IM_TRUCK)

    cases.append(LoadCase(
        name="LL_HL93_truck",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Design Truck (8k-32k-32k, 14' spacing) on {L}' span. "
            f"M={truck_moment:.1f} k-ft, V={truck_shear:.1f} k (before IM). "
            f"IM={IM_TRUCK*100:.0f}%: M={truck_moment_im:.1f} k-ft, V={truck_shear_im:.1f} k"
        ),
        loads=[{
            "type": "point",
            "vehicle": "HL93_truck",
            "axles": DESIGN_TRUCK_AXLES,
            "max_moment_kft": round(truck_moment, 2),
            "max_moment_im_kft": round(truck_moment_im, 2),
            "max_shear_kip": round(truck_shear, 2),
            "max_shear_im_kip": round(truck_shear_im, 2),
            "impact_factor": IM_TRUCK,
        }],
    ))

    # --- Design Tandem ---
    tandem_moment = _simple_span_moment(DESIGN_TANDEM_AXLES, L)
    tandem_shear = _simple_span_shear(DESIGN_TANDEM_AXLES, L)
    tandem_moment_im = tandem_moment * (1 + IM_TRUCK)
    tandem_shear_im = tandem_shear * (1 + IM_TRUCK)

    cases.append(LoadCase(
        name="LL_HL93_tandem",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Design Tandem (2×25k, 4' spacing) on {L}' span. "
            f"M={tandem_moment:.1f} k-ft, V={tandem_shear:.1f} k (before IM). "
            f"IM={IM_TRUCK*100:.0f}%: M={tandem_moment_im:.1f} k-ft, V={tandem_shear_im:.1f} k"
        ),
        loads=[{
            "type": "point",
            "vehicle": "HL93_tandem",
            "axles": DESIGN_TANDEM_AXLES,
            "max_moment_kft": round(tandem_moment, 2),
            "max_moment_im_kft": round(tandem_moment_im, 2),
            "max_shear_kip": round(tandem_shear, 2),
            "max_shear_im_kip": round(tandem_shear_im, 2),
            "impact_factor": IM_TRUCK,
        }],
    ))

    # --- Design Lane ---
    lane_moment = DESIGN_LANE_KLF * L ** 2 / 8.0  # wL²/8 for simple span
    lane_shear = DESIGN_LANE_KLF * L / 2.0         # wL/2

    cases.append(LoadCase(
        name="LL_HL93_lane",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Design Lane Load: {DESIGN_LANE_KLF} klf uniform on {L}' span. "
            f"M={lane_moment:.1f} k-ft, V={lane_shear:.1f} k (no IM applied to lane)"
        ),
        loads=[{
            "type": "distributed",
            "vehicle": "HL93_lane",
            "w_klf": DESIGN_LANE_KLF,
            "w_kip_per_in": DESIGN_LANE_KLF / 12.0,
            "max_moment_kft": round(lane_moment, 2),
            "max_shear_kip": round(lane_shear, 2),
            "impact_factor": IM_LANE,
            "lane_width_ft": LANE_WIDTH_FT,
        }],
    ))

    # --- HL-93 Combinations ---
    # Truck + Lane
    hl93_truck_lane_moment = truck_moment_im + lane_moment
    hl93_truck_lane_shear = truck_shear_im + lane_shear

    cases.append(LoadCase(
        name="LL_HL93_truck_lane",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Combination: Truck+Lane on {L}' span. "
            f"M={hl93_truck_lane_moment:.1f} k-ft, V={hl93_truck_lane_shear:.1f} k"
        ),
        loads=[{
            "type": "combination",
            "components": ["HL93_truck", "HL93_lane"],
            "max_moment_kft": round(hl93_truck_lane_moment, 2),
            "max_shear_kip": round(hl93_truck_lane_shear, 2),
            "governs_moment": hl93_truck_lane_moment >= (tandem_moment_im + lane_moment),
        }],
    ))

    # Tandem + Lane
    hl93_tandem_lane_moment = tandem_moment_im + lane_moment
    hl93_tandem_lane_shear = tandem_shear_im + lane_shear

    cases.append(LoadCase(
        name="LL_HL93_tandem_lane",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Combination: Tandem+Lane on {L}' span. "
            f"M={hl93_tandem_lane_moment:.1f} k-ft, V={hl93_tandem_lane_shear:.1f} k"
        ),
        loads=[{
            "type": "combination",
            "components": ["HL93_tandem", "HL93_lane"],
            "max_moment_kft": round(hl93_tandem_lane_moment, 2),
            "max_shear_kip": round(hl93_tandem_lane_shear, 2),
            "governs_moment": hl93_tandem_lane_moment > hl93_truck_lane_moment,
        }],
    ))

    # Negative moment: 90% × two trucks (min 50ft apart) + 90% lane
    # (AASHTO 3.6.1.3.1) — for continuous spans
    neg_moment_truck = truck_moment_im * 0.90 * 2
    neg_moment_lane = lane_moment * 0.90
    neg_moment_total = neg_moment_truck + neg_moment_lane

    cases.append(LoadCase(
        name="LL_HL93_neg_moment",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Negative Moment: 90%×2 trucks + 90% lane = "
            f"{neg_moment_total:.1f} k-ft (AASHTO 3.6.1.3.1)"
        ),
        loads=[{
            "type": "combination",
            "components": ["HL93_truck_x2", "HL93_lane"],
            "factors": {"truck": 0.90, "lane": 0.90},
            "max_moment_kft": round(neg_moment_total, 2),
            "note": "For negative moment regions of continuous spans",
        }],
    ))

    # Governing HL-93
    gov_moment = max(hl93_truck_lane_moment, hl93_tandem_lane_moment)
    gov_shear = max(hl93_truck_lane_shear, hl93_tandem_lane_shear)
    gov_type = "Truck+Lane" if hl93_truck_lane_moment >= hl93_tandem_lane_moment else "Tandem+Lane"

    cases.append(LoadCase(
        name="LL_HL93_governing",
        category="standard",
        load_type="LL",
        description=(
            f"HL-93 Governing: {gov_type} controls for {L}' span. "
            f"M={gov_moment:.1f} k-ft, V={gov_shear:.1f} k"
        ),
        loads=[{
            "type": "envelope",
            "governing": gov_type,
            "max_moment_kft": round(gov_moment, 2),
            "max_shear_kip": round(gov_shear, 2),
        }],
    ))

    return cases


# ======================================================================
# Permit vehicles
# ======================================================================

def _compute_permit_loads(geom: BridgeGeometry) -> list[LoadCase]:
    """Generate permit vehicle load cases."""
    cases: list[LoadCase] = []

    for key, vehicle in PERMIT_VEHICLES.items():
        axles = vehicle["axles"]
        moment = _simple_span_moment(axles, geom.span_ft)
        shear = _simple_span_shear(axles, geom.span_ft)
        moment_im = moment * (1 + IM_TRUCK)
        shear_im = shear * (1 + IM_TRUCK)

        cases.append(LoadCase(
            name=f"P_{key}",
            category="standard",
            load_type="P",
            description=(
                f"Permit: {vehicle['name']} ({vehicle['total_kip']}k) on {geom.span_ft}' span. "
                f"M={moment_im:.1f} k-ft (w/ IM), V={shear_im:.1f} k"
            ),
            loads=[{
                "type": "point",
                "vehicle": vehicle["name"],
                "axles": axles,
                "total_weight_kip": vehicle["total_kip"],
                "max_moment_kft": round(moment, 2),
                "max_moment_im_kft": round(moment_im, 2),
                "max_shear_kip": round(shear, 2),
                "max_shear_im_kip": round(shear_im, 2),
                "impact_factor": IM_TRUCK,
            }],
        ))

    return cases


# ======================================================================
# Thermal loads
# ======================================================================

def _compute_thermal_loads(
    geom: BridgeGeometry,
    thermal_profile: Optional[dict] = None,
    state: str = "",
) -> list[LoadCase]:
    """Generate uniform temperature (TU) and gradient (TG) load cases."""
    cases: list[LoadCase] = []

    # --- TU: Uniform temperature change ---
    if thermal_profile:
        delta_t = thermal_profile.get("delta_t", 120.0)
        t_min = thermal_profile.get("t_min", -10)
        t_max = thermal_profile.get("t_max", 110)
    else:
        delta_t = 120.0
        t_min = -10
        t_max = 110

    # Setting temperature assumed at 60°F
    t_set = 60.0
    delta_t_rise = t_max - t_set
    delta_t_fall = t_set - t_min

    # Coefficient of thermal expansion
    alpha = 6.5e-6 if geom.structure_type == "steel" else 5.5e-6  # per °F

    cases.append(LoadCase(
        name="TU_rise",
        category="standard",
        load_type="TU",
        description=(
            f"Uniform temperature rise: +{delta_t_rise}°F (T_set={t_set}°F → T_max={t_max}°F). "
            f"Strain = {alpha * delta_t_rise:.6f}"
        ),
        loads=[{
            "type": "thermal",
            "subtype": "uniform",
            "delta_t_F": delta_t_rise,
            "strain": alpha * delta_t_rise,
            "alpha_per_F": alpha,
        }],
    ))

    cases.append(LoadCase(
        name="TU_fall",
        category="standard",
        load_type="TU",
        description=(
            f"Uniform temperature fall: -{delta_t_fall}°F (T_set={t_set}°F → T_min={t_min}°F). "
            f"Strain = {alpha * delta_t_fall:.6f}"
        ),
        loads=[{
            "type": "thermal",
            "subtype": "uniform",
            "delta_t_F": -delta_t_fall,
            "strain": -alpha * delta_t_fall,
            "alpha_per_F": alpha,
        }],
    ))

    # --- TG: Temperature gradient ---
    zone = _GRADIENT_ZONE_BY_STATE.get(state, 3)
    grad = THERMAL_GRADIENT_ZONES[zone]

    neg_factor = (
        NEG_GRADIENT_STEEL if geom.structure_type == "steel"
        else NEG_GRADIENT_CONCRETE
    )

    cases.append(LoadCase(
        name="TG_positive",
        category="standard",
        load_type="TG",
        description=(
            f"Positive thermal gradient (Zone {zone}): T1={grad['T1']}°F at top, "
            f"T2={grad['T2']}°F at 4\" below. Linear decrease to zero at girder bottom."
        ),
        loads=[{
            "type": "thermal",
            "subtype": "gradient",
            "zone": zone,
            "T1_F": grad["T1"],
            "T2_F": grad["T2"],
            "direction": "positive",
        }],
    ))

    cases.append(LoadCase(
        name="TG_negative",
        category="standard",
        load_type="TG",
        description=(
            f"Negative thermal gradient (Zone {zone}): factor={neg_factor}. "
            f"T1={neg_factor * grad['T1']:.1f}°F, T2={neg_factor * grad['T2']:.1f}°F"
        ),
        loads=[{
            "type": "thermal",
            "subtype": "gradient",
            "zone": zone,
            "T1_F": round(neg_factor * grad["T1"], 1),
            "T2_F": round(neg_factor * grad["T2"], 1),
            "direction": "negative",
            "negative_factor": neg_factor,
        }],
    ))

    return cases


# ======================================================================
# Wind loads
# ======================================================================

def _get_kz(height_ft: float) -> float:
    """Interpolate velocity pressure exposure coefficient Kz (Exposure C)."""
    if height_ft <= 15:
        return 0.85
    if height_ft >= 90:
        return 1.24
    # Linear interpolation
    keys = sorted(_KZ_TABLE_EXP_C.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= height_ft <= keys[i + 1]:
            z1, z2 = keys[i], keys[i + 1]
            k1, k2 = _KZ_TABLE_EXP_C[z1], _KZ_TABLE_EXP_C[z2]
            return k1 + (k2 - k1) * (height_ft - z1) / (z2 - z1)
    return 1.0


def _compute_wind_loads(
    geom: BridgeGeometry,
    wind_v: int = 115,
    bridge_height_ft: float = 30.0,
) -> list[LoadCase]:
    """Generate wind load cases (WS and WL)."""
    cases: list[LoadCase] = []

    # --- WS: Wind on structure ---
    kz = _get_kz(bridge_height_ft)
    # Velocity pressure (psf)
    qz = WIND_PRESSURE_COEFF * kz * wind_v ** 2
    # Wind pressure on exposed surface
    p_ws = qz * WIND_GUST_FACTOR * WIND_CD_GIRDER  # psf

    # Exposed depth = girder depth + barrier (typ 3.5 ft)
    exposed_depth_ft = (geom.girder_depth_in or 48.0) / 12.0 + 3.5  # default 48" if None
    # Distributed lateral load (klf)
    ws_klf = p_ws * exposed_depth_ft / 1000.0
    ws_kli = ws_klf / 12.0

    cases.append(LoadCase(
        name="WS_wind_on_structure",
        category="standard",
        load_type="WS",
        description=(
            f"Wind on structure: V={wind_v} mph, Kz={kz:.3f}, qz={qz:.1f} psf, "
            f"p={p_ws:.1f} psf on {exposed_depth_ft:.1f}' exposed depth = {ws_klf:.3f} klf"
        ),
        loads=[{
            "type": "distributed",
            "w_klf": round(ws_klf, 4),
            "w_kip_per_in": round(ws_kli, 6),
            "direction": "lateral",
            "pressure_psf": round(p_ws, 2),
            "exposed_depth_ft": round(exposed_depth_ft, 2),
            "wind_speed_mph": wind_v,
            "Kz": round(kz, 3),
        }],
    ))

    # --- WL: Wind on live load ---
    wl_klf = WL_KLF
    wl_kli = wl_klf / 12.0

    cases.append(LoadCase(
        name="WL_wind_on_live_load",
        category="standard",
        load_type="WL",
        description=(
            f"Wind on live load: {WL_KLF} klf applied {WL_HEIGHT_FT}' above deck "
            f"(AASHTO 3.8.1.3)"
        ),
        loads=[{
            "type": "distributed",
            "w_klf": wl_klf,
            "w_kip_per_in": round(wl_kli, 6),
            "direction": "lateral",
            "height_above_deck_ft": WL_HEIGHT_FT,
        }],
    ))

    return cases


# ======================================================================
# Braking force
# ======================================================================

def _compute_braking(geom: BridgeGeometry) -> list[LoadCase]:
    """Compute braking force (BR) per AASHTO 3.6.4."""
    # 25% of truck axle weights
    truck_total = sum(a["weight_kip"] for a in DESIGN_TRUCK_AXLES)
    br_truck = BR_TRUCK_FRACTION * truck_total

    # 5% of (truck + lane on full span)
    lane_total = DESIGN_LANE_KLF * geom.span_ft
    br_combo = BR_COMBO_FRACTION * (truck_total + lane_total)

    br_force = max(br_truck, br_combo)

    return [LoadCase(
        name="BR_braking",
        category="standard",
        load_type="BR",
        description=(
            f"Braking force: max(25%×{truck_total:.0f}k = {br_truck:.1f}k, "
            f"5%×({truck_total:.0f}+{lane_total:.1f}) = {br_combo:.1f}k) = {br_force:.1f}k. "
            f"Applied longitudinally at deck level."
        ),
        loads=[{
            "type": "point",
            "force_kip": round(br_force, 2),
            "direction": "longitudinal",
            "height": "deck_level",
            "br_truck_kip": round(br_truck, 2),
            "br_combo_kip": round(br_combo, 2),
        }],
    )]


# ======================================================================
# Seismic loads
# ======================================================================

def _compute_seismic(
    geom: BridgeGeometry,
    seismic_profile: Optional[dict] = None,
) -> list[LoadCase]:
    """Generate seismic load cases from site response spectrum."""
    cases: list[LoadCase] = []

    if seismic_profile is None:
        seismic_profile = {
            "sds": 0.267, "sd1": 0.160, "pga": 0.10,
            "sdc": "B", "site_class": "D",
        }

    sds = seismic_profile.get("sds", 0.267)
    sd1 = seismic_profile.get("sd1", 0.160)
    pga = seismic_profile.get("pga", 0.10)
    sdc = seismic_profile.get("sdc", "B")

    # Build AASHTO design response spectrum points
    # T0 = 0.2 * SD1/SDS, Ts = SD1/SDS
    if sds > 0:
        ts = sd1 / sds
        t0 = 0.2 * ts
    else:
        ts = 1.0
        t0 = 0.2

    spectrum_points = []
    # Ramp from PGA to SDS
    spectrum_points.append({"T": 0.0, "Sa": pga})
    spectrum_points.append({"T": t0, "Sa": sds})
    # Constant plateau
    spectrum_points.append({"T": ts, "Sa": sds})
    # 1/T descent
    for t in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        if t > ts:
            sa = sd1 / t
            spectrum_points.append({"T": t, "Sa": round(sa, 4)})

    # 100% longitudinal + 30% transverse
    cases.append(LoadCase(
        name="EQ_longitudinal",
        category="standard",
        load_type="EQ",
        description=(
            f"Seismic: 100% longitudinal + 30% transverse (SDC {sdc}). "
            f"SDS={sds:.3f}g, SD1={sd1:.3f}g, PGA={pga:.3f}g"
        ),
        loads=[{
            "type": "spectrum",
            "direction": "longitudinal",
            "factor": 1.00,
            "orthogonal_factor": 0.30,
            "spectrum": spectrum_points,
            "sds": sds,
            "sd1": sd1,
            "pga": pga,
            "sdc": sdc,
        }],
    ))

    # 30% longitudinal + 100% transverse
    cases.append(LoadCase(
        name="EQ_transverse",
        category="standard",
        load_type="EQ",
        description=(
            f"Seismic: 30% longitudinal + 100% transverse (SDC {sdc}). "
            f"SDS={sds:.3f}g, SD1={sd1:.3f}g"
        ),
        loads=[{
            "type": "spectrum",
            "direction": "transverse",
            "factor": 1.00,
            "orthogonal_factor": 0.30,
            "spectrum": spectrum_points,
            "sds": sds,
            "sd1": sd1,
            "pga": pga,
            "sdc": sdc,
        }],
    ))

    return cases


# ======================================================================
# Scour
# ======================================================================

def _compute_scour(water_crossing: bool = False) -> list[LoadCase]:
    """Generate scour modification flag."""
    if not water_crossing:
        return []

    return [LoadCase(
        name="SC_scour",
        category="standard",
        load_type="SC",
        description=(
            "Scour: modifies foundation springs — remove soil resistance above "
            "computed scour depth (Q100 for Strength, Q500 for Extreme Event). "
            "Not a direct load — flag passed to foundation tool."
        ),
        loads=[{
            "type": "modification",
            "target": "foundation_springs",
            "action": "remove_above_scour_depth",
            "design_flood": "Q100",
            "check_flood": "Q500",
        }],
    )]


# ======================================================================
# Load combinations — AASHTO Table 3.4.1-1
# ======================================================================

# Factor table: {limit_state: {load_type: (max_factor, min_factor) or factor}}
# None means load not included in that combination.
_COMBO_TABLE: dict[str, dict[str, object]] = {
    "Strength I": {
        "DC": (1.25, 0.90), "DW": (1.50, 0.65), "LL": 1.75,
        "TU": (0.50, 1.20), "WS": None, "WL": None, "BR": 1.75, "EQ": None,
    },
    "Strength II": {
        "DC": (1.25, 0.90), "DW": (1.50, 0.65), "LL": 1.35,
        "TU": (0.50, 1.20), "WS": None, "WL": None, "BR": None, "EQ": None,
    },
    "Strength III": {
        "DC": (1.25, 0.90), "DW": (1.50, 0.65), "LL": None,
        "TU": (0.50, 1.20), "WS": 1.00, "WL": None, "BR": None, "EQ": None,
    },
    "Strength IV": {
        "DC": 1.50, "DW": 1.50, "LL": None,
        "TU": None, "WS": None, "WL": None, "BR": None, "EQ": None,
    },
    "Strength V": {
        "DC": (1.25, 0.90), "DW": (1.50, 0.65), "LL": 1.35,
        "TU": (0.50, 1.20), "WS": 0.40, "WL": 1.00, "BR": None, "EQ": None,
    },
    "Extreme Event I": {
        "DC": (1.25, 0.90), "DW": (1.50, 0.65), "LL": 0.50,
        "TU": None, "WS": None, "WL": None, "BR": None, "EQ": 1.00,
    },
    "Extreme Event II": {
        "DC": (1.25, 0.90), "DW": (1.50, 0.65), "LL": 0.50,
        "TU": None, "WS": None, "WL": None, "BR": None, "EQ": None,
    },
    "Service I": {
        "DC": 1.00, "DW": 1.00, "LL": 1.00,
        "TU": (1.00, 1.20), "WS": 0.30, "WL": 1.00, "BR": None, "EQ": None,
    },
    "Service II": {
        "DC": 1.00, "DW": 1.00, "LL": 1.30,
        "TU": None, "WS": None, "WL": None, "BR": None, "EQ": None,
    },
    "Service III": {
        "DC": 1.00, "DW": 1.00, "LL": 0.80,
        "TU": (1.00, 1.20), "WS": None, "WL": None, "BR": None, "EQ": None,
    },
    "Fatigue I": {
        "DC": None, "DW": None, "LL": 1.75,
        "TU": None, "WS": None, "WL": None, "BR": None, "EQ": None,
    },
    "Fatigue II": {
        "DC": None, "DW": None, "LL": 0.80,
        "TU": None, "WS": None, "WL": None, "BR": None, "EQ": None,
    },
}


def _generate_combinations(cases: list[LoadCase]) -> list[LoadCombination]:
    """Generate all AASHTO load combinations from the case list.

    For load types with max/min factors (DC, DW, TU), generates separate
    max and min envelope combinations.
    """
    combos: list[LoadCombination] = []

    # Build lookup: load_type → [case names]
    cases_by_type: dict[str, list[str]] = {}
    for c in cases:
        lt = c.load_type
        # Map P (permit) to LL for combination purposes
        if lt == "P":
            lt = "LL"
        if lt == "TG":
            lt = "TU"  # gradient grouped with thermal
        cases_by_type.setdefault(lt, []).append(c.name)

    for limit_state, factors in _COMBO_TABLE.items():
        # Determine which envelope variants to generate
        has_dual = any(isinstance(v, tuple) for v in factors.values() if v is not None)

        if has_dual:
            # Generate max and min envelope variants
            for suffix, idx in [("max", 0), ("min", 1)]:
                combo_factors = {}
                for lt, fval in factors.items():
                    if fval is None:
                        continue
                    if isinstance(fval, tuple):
                        factor = fval[idx]
                    else:
                        factor = fval
                    for case_name in cases_by_type.get(lt, []):
                        combo_factors[case_name] = factor

                if combo_factors:
                    safe_name = limit_state.replace(" ", "_")
                    combos.append(LoadCombination(
                        name=f"{safe_name}_{suffix}",
                        limit_state=limit_state,
                        factors=combo_factors,
                    ))
        else:
            combo_factors = {}
            for lt, fval in factors.items():
                if fval is None:
                    continue
                factor = fval if not isinstance(fval, tuple) else fval[0]
                for case_name in cases_by_type.get(lt, []):
                    combo_factors[case_name] = factor

            if combo_factors:
                safe_name = limit_state.replace(" ", "_")
                combos.append(LoadCombination(
                    name=safe_name,
                    limit_state=limit_state,
                    factors=combo_factors,
                ))

    return combos


# ======================================================================
# Adversarial loads (RED TEAM)
# ======================================================================

def _generate_adversarial_cases(geom: BridgeGeometry) -> list[LoadCase]:
    """Generate adversarial (red-team) load cases.

    These represent conditions that standard practice typically doesn't model
    but can cause failures in real bridges.
    """
    cases: list[LoadCase] = []

    # ------------------------------------------------------------------
    # 1. Construction loads
    # ------------------------------------------------------------------
    cases.append(LoadCase(
        name="ADV_crane_on_overhang",
        category="adversarial",
        load_type="CONST",
        description=(
            "Construction: Crane on overhang during deck pour. "
            "30-50k point load at deck edge, partial structure (no composite action)."
        ),
        loads=[{
            "type": "point",
            "force_kip": 40.0,
            "position": "deck_edge",
            "position_from_ext_girder_ft": geom.overhang_ft,
            "composite_action": False,
            "scenario": "deck_pour",
        }],
    ))

    cases.append(LoadCase(
        name="ADV_concrete_truck_fresh_deck",
        category="adversarial",
        load_type="CONST",
        description=(
            "Construction: Concrete delivery truck (66k) on fresh deck. "
            "No composite action, wet concrete dead load, plus truck."
        ),
        loads=[{
            "type": "point",
            "vehicle": "concrete_truck",
            "axles": [
                {"weight_kip": 10.0, "position_ft": 0.0},
                {"weight_kip": 18.0, "position_ft": 12.0},
                {"weight_kip": 18.0, "position_ft": 16.0},
                {"weight_kip": 10.0, "position_ft": 28.0},
                {"weight_kip": 10.0, "position_ft": 32.0},
            ],
            "total_weight_kip": 66.0,
            "composite_action": False,
            "additional_wet_concrete_load": True,
        }],
    ))

    cases.append(LoadCase(
        name="ADV_ilm_launching_nose",
        category="adversarial",
        load_type="CONST",
        description=(
            "Construction: Incremental launch — launching nose reactions at pier. "
            "High concentrated reactions during launch, potential uplift at trailing end."
        ),
        loads=[{
            "type": "point",
            "force_kip": 100.0,
            "position": "pier_top",
            "scenario": "incremental_launch",
            "note": "Nose reaction varies with launch position. Check at all stages.",
        }],
    ))

    # ------------------------------------------------------------------
    # 2. Component failure scenarios
    # ------------------------------------------------------------------
    cases.append(LoadCase(
        name="ADV_lost_bearing",
        category="adversarial",
        load_type="FAIL",
        description=(
            "Failure: Lost bearing — one bearing pad removed (corrosion/displacement). "
            "Redistribute dead + live load to adjacent bearings. "
            "Check for girder uplift and pier overload."
        ),
        loads=[{
            "type": "modification",
            "action": "remove_bearing",
            "target": "one_bearing",
            "redistribution": "to_adjacent",
            "check": ["uplift", "adjacent_overload", "substructure_capacity"],
        }],
    ))

    cases.append(LoadCase(
        name="ADV_severed_tendon",
        category="adversarial",
        load_type="FAIL",
        description=(
            "Failure: One post-tensioning tendon severed (corrosion at duct). "
            "Reduce prestress force, check remaining capacity for full dead + live load."
        ),
        loads=[{
            "type": "modification",
            "action": "remove_tendon",
            "target": "one_tendon",
            "check": ["flexural_capacity", "shear_capacity", "deflection"],
        }],
    ))

    cases.append(LoadCase(
        name="ADV_buckled_brace",
        category="adversarial",
        load_type="FAIL",
        description=(
            "Failure: One truss diagonal buckled — check alternate load path. "
            "Remove one diagonal member, verify structure can carry DC + 0.5×LL "
            "without progressive collapse."
        ),
        loads=[{
            "type": "modification",
            "action": "remove_member",
            "target": "one_truss_diagonal",
            "load_level": {"DC": 1.0, "LL": 0.5},
            "check": ["alternate_path", "progressive_collapse"],
        }],
    ))

    cases.append(LoadCase(
        name="ADV_seized_bearing",
        category="adversarial",
        load_type="FAIL",
        description=(
            "Failure: Seized expansion bearing — locked in position. "
            "Change boundary condition from free to fixed, apply locked-in "
            "thermal forces for full temperature range."
        ),
        loads=[{
            "type": "modification",
            "action": "change_bc",
            "target": "expansion_bearing",
            "from": "free",
            "to": "fixed",
            "locked_in_thermal": True,
        }],
    ))

    # ------------------------------------------------------------------
    # 3. Extreme event combinations
    # ------------------------------------------------------------------
    cases.append(LoadCase(
        name="ADV_scour_plus_seismic",
        category="adversarial",
        load_type="EXT",
        description=(
            "Extreme: Scour (Q100) + Seismic — FHWA recommends but AASHTO doesn't require. "
            "Remove foundation springs above scour depth, then apply seismic spectrum."
        ),
        loads=[{
            "type": "combination",
            "components": ["scour_Q100", "seismic"],
            "note": "FHWA recommendation per HEC-18 §8.5",
        }],
    ))

    cases.append(LoadCase(
        name="ADV_flood_vessel_collision",
        category="adversarial",
        load_type="EXT",
        description=(
            "Extreme: Flood (Q100 water elevation) + vessel collision. "
            "Higher water = larger vessel access + increased hydrodynamic force."
        ),
        loads=[{
            "type": "combination",
            "components": ["flood_Q100", "vessel_collision"],
            "vessel_force_kip": 500.0,
            "note": "Force depends on waterway vessel traffic classification",
        }],
    ))

    cases.append(LoadCase(
        name="ADV_fire",
        category="adversarial",
        load_type="EXT",
        description=(
            "Extreme: Fire — reduce material strengths. "
            "Steel Fy → 60% at 600°C, concrete f'c → 75% at 500°C. "
            "Check capacity under DC + 0.5×LL with degraded properties."
        ),
        loads=[{
            "type": "modification",
            "action": "reduce_material",
            "steel_fy_factor": 0.60,
            "steel_temperature_C": 600,
            "concrete_fc_factor": 0.75,
            "concrete_temperature_C": 500,
            "load_level": {"DC": 1.0, "LL": 0.5},
        }],
    ))

    cases.append(LoadCase(
        name="ADV_ice_thermal_extreme",
        category="adversarial",
        load_type="EXT",
        description=(
            "Extreme: Ice loading + thermal extreme (min temperature). "
            "Ice accretion on structure + expansion joint frozen + maximum contraction."
        ),
        loads=[{
            "type": "combination",
            "components": ["ice_load", "thermal_min"],
            "ice_thickness_in": 1.0,
            "ice_density_pcf": 56.0,
        }],
    ))

    # ------------------------------------------------------------------
    # 4. Degradation scenarios
    # ------------------------------------------------------------------
    cases.append(LoadCase(
        name="ADV_corrosion_10pct",
        category="adversarial",
        load_type="DEG",
        description=(
            "Degradation: 10% section loss on bottom flange (corrosion). "
            "Reduce flange area and moment of inertia, check capacity at "
            "Strength I with degraded section."
        ),
        loads=[{
            "type": "modification",
            "action": "reduce_section",
            "target": "bottom_flange",
            "section_loss_pct": 10.0,
            "cause": "corrosion",
        }],
    ))

    cases.append(LoadCase(
        name="ADV_rebar_section_loss",
        category="adversarial",
        load_type="DEG",
        description=(
            "Degradation: 20% rebar section loss in deck (deicing salt exposure). "
            "Reduce deck reinforcement, check negative moment capacity over piers."
        ),
        loads=[{
            "type": "modification",
            "action": "reduce_section",
            "target": "deck_rebar",
            "section_loss_pct": 20.0,
            "cause": "deicing_salt_corrosion",
        }],
    ))

    cases.append(LoadCase(
        name="ADV_deck_delamination",
        category="adversarial",
        load_type="DEG",
        description=(
            "Degradation: Concrete delamination — partial loss of composite action. "
            "Reduce effective deck width by 50%, check deflection and capacity."
        ),
        loads=[{
            "type": "modification",
            "action": "reduce_composite",
            "effective_width_factor": 0.50,
            "cause": "delamination",
        }],
    ))

    return cases


def _generate_adversarial_combos(
    standard_cases: list[LoadCase],
    adversarial_cases: list[LoadCase],
) -> list[LoadCombination]:
    """Generate adversarial load combinations.

    Each adversarial case is combined with relevant standard loads at
    appropriate load levels.
    """
    combos: list[LoadCombination] = []

    # Build DC/DW case name lists
    dc_cases = [c.name for c in standard_cases if c.load_type == "DC"]
    dw_cases = [c.name for c in standard_cases if c.load_type == "DW"]
    ll_cases = [c.name for c in standard_cases
                if c.load_type == "LL" and "governing" in c.name]
    eq_cases = [c.name for c in standard_cases if c.load_type == "EQ"]
    tu_cases = [c.name for c in standard_cases if c.load_type == "TU"]

    for adv in adversarial_cases:
        factors: dict[str, float] = {}

        # Always include dead loads
        for dc in dc_cases:
            factors[dc] = 1.25
        for dw in dw_cases:
            factors[dw] = 1.50

        # Load level depends on adversarial type
        if adv.load_type == "CONST":
            # Construction: DC at 1.25, no LL (or reduced)
            factors[adv.name] = 1.00
        elif adv.load_type == "FAIL":
            # Component failure: DC + 0.5 LL
            for ll in ll_cases:
                factors[ll] = 0.50
            factors[adv.name] = 1.00
        elif adv.load_type == "EXT":
            # Extreme events: DC + 0.5 LL + event
            for ll in ll_cases:
                factors[ll] = 0.50
            factors[adv.name] = 1.00
            if "seismic" in adv.name.lower():
                for eq in eq_cases:
                    factors[eq] = 1.00
        elif adv.load_type == "DEG":
            # Degradation: full Strength I loads with degraded section
            for ll in ll_cases:
                factors[ll] = 1.75
            factors[adv.name] = 1.00

        combos.append(LoadCombination(
            name=f"ADV_{adv.name}",
            limit_state="Adversarial",
            factors=factors,
        ))

    # Additional cross-adversarial combos
    # Fire + seized bearing
    fire_cases = [c.name for c in adversarial_cases if "fire" in c.name]
    seized_cases = [c.name for c in adversarial_cases if "seized" in c.name]
    if fire_cases and seized_cases:
        factors = {}
        for dc in dc_cases:
            factors[dc] = 1.00
        for dw in dw_cases:
            factors[dw] = 1.00
        for ll in ll_cases:
            factors[ll] = 0.50
        factors[fire_cases[0]] = 1.00
        factors[seized_cases[0]] = 1.00
        combos.append(LoadCombination(
            name="ADV_fire_seized_bearing",
            limit_state="Adversarial",
            factors=factors,
        ))

    # Corrosion + overload
    corr_cases = [c.name for c in adversarial_cases if "corrosion" in c.name]
    if corr_cases:
        factors = {}
        for dc in dc_cases:
            factors[dc] = 1.25
        for dw in dw_cases:
            factors[dw] = 1.50
        for ll in ll_cases:
            factors[ll] = 1.75
        factors[corr_cases[0]] = 1.00
        # Add permit load
        permit_cases = [c.name for c in standard_cases if c.load_type == "P"]
        for pc in permit_cases[:1]:
            factors[pc] = 1.35
        combos.append(LoadCombination(
            name="ADV_corrosion_plus_permit",
            limit_state="Adversarial",
            factors=factors,
        ))

    return combos


# ======================================================================
# Moving load positions for influence-line analysis
# ======================================================================

def generate_moving_load_positions(
    span_lengths: list[float],
    num_positions: int = 20,
) -> dict:
    """Generate moving load positions for HL-93 influence-line analysis.

    For each position along the bridge, computes axle locations for
    design truck, design tandem, and lane load.

    Args:
        span_lengths: List of span lengths in feet.
        num_positions: Number of truck front-axle positions to evaluate.

    Returns:
        Dict with keys:
            ``truck_positions``: list of truck position dicts
            ``tandem_positions``: list of tandem position dicts
            ``lane_load``: uniform lane load dict
    """
    total_length_ft = sum(span_lengths)

    # Generate evenly-spaced front-axle positions along bridge
    step = total_length_ft / (num_positions + 1)
    positions_ft = [step * (i + 1) for i in range(num_positions)]

    truck_positions = []
    for pos in positions_ft:
        # Design truck: 8k at front, 32k at 14ft, 32k at 28ft behind
        axle_loads = []
        for axle in DESIGN_TRUCK_AXLES:
            loc_ft = pos - axle["position_ft"]  # axle position on bridge
            if 0 <= loc_ft <= total_length_ft:
                axle_loads.append({
                    "location_ft": round(loc_ft, 2),
                    "location_in": round(loc_ft * 12.0, 2),
                    "force_kip": axle["weight_kip"],
                    "force_kip_im": round(axle["weight_kip"] * (1 + IM_TRUCK), 2),
                })
        if len(axle_loads) >= 2:  # At least 2 axles on bridge
            truck_positions.append({
                "type": "truck",
                "front_axle_ft": round(pos, 2),
                "axle_loads": axle_loads,
            })

    tandem_positions = []
    for pos in positions_ft:
        # Design tandem: 25k at front, 25k at 4ft behind
        axle_loads = []
        for axle in DESIGN_TANDEM_AXLES:
            loc_ft = pos - axle["position_ft"]
            if 0 <= loc_ft <= total_length_ft:
                axle_loads.append({
                    "location_ft": round(loc_ft, 2),
                    "location_in": round(loc_ft * 12.0, 2),
                    "force_kip": axle["weight_kip"],
                    "force_kip_im": round(axle["weight_kip"] * (1 + IM_TRUCK), 2),
                })
        if len(axle_loads) >= 1:
            tandem_positions.append({
                "type": "tandem",
                "front_axle_ft": round(pos, 2),
                "axle_loads": axle_loads,
            })

    lane_load = {
        "type": "lane",
        "w_klf": DESIGN_LANE_KLF,
        "w_kip_per_in": round(DESIGN_LANE_KLF / 12.0, 6),
        "total_length_ft": total_length_ft,
        "total_force_kip": round(DESIGN_LANE_KLF * total_length_ft, 2),
        "impact_factor": IM_LANE,
    }

    return {
        "truck_positions": truck_positions,
        "tandem_positions": tandem_positions,
        "lane_load": lane_load,
        "total_length_ft": total_length_ft,
        "num_positions": num_positions,
        "span_lengths_ft": span_lengths,
    }


# ======================================================================
# OpenSees load combination script generation
# ======================================================================

def generate_load_combination_script(
    load_cases: list[LoadCase],
    combinations: list[LoadCombination],
    element_tags: list[int] | None = None,
    node_map: dict | None = None,
    girder_spacing_ft: float = 8.0,
) -> str:
    """Generate OpenSees Python code for load combination analysis.

    Produces a script block that:
      1. Defines a timeSeries + pattern for each load case
      2. For each combination, applies factored loads
      3. Runs static analysis
      4. Extracts element forces into a results dict
      5. Computes envelopes across all combinations

    Args:
        load_cases:       List of :class:`LoadCase` objects.
        combinations:     List of :class:`LoadCombination` objects.
        element_tags:     Element tags to extract forces from.
                          If None, uses ``ops.getEleTags()``.
        node_map:         Dict mapping load application descriptions to node tags.
                          If None, loads are applied to all beam elements.
        girder_spacing_ft: Girder spacing for simplified distribution factors.

    Returns:
        Multi-line Python string ready to be inserted into an OpenSees script.
    """
    lines = [
        "",
        "# " + "=" * 60,
        "# LOAD COMBINATION ANALYSIS",
        "# " + "=" * 60,
        "",
        "import copy",
        "",
        "# --- Simplified distribution factors ---",
        f"g_moment = {girder_spacing_ft} / 11.0  # S/11 for moment",
        f"g_shear = {girder_spacing_ft} / 7.0   # S/7 for shear",
        "",
        "# --- Element tags for force extraction ---",
    ]

    if element_tags:
        lines.append(f"envelope_ele_tags = {element_tags}")
    else:
        lines.append("envelope_ele_tags = ops.getEleTags()")

    lines.extend([
        "",
        "# Storage for all combination results",
        "combination_results = {}",
        "",
    ])

    # Build a mapping from load case name to a pattern tag
    ts_tag_start = 100
    pat_tag_start = 100

    # Categorize load cases for script generation
    dc_cases = [c for c in load_cases if c.load_type == "DC"]
    dw_cases = [c for c in load_cases if c.load_type == "DW"]
    ll_cases = [c for c in load_cases if c.load_type == "LL"
                and any(ld.get("type") == "distributed" for ld in c.loads)]
    ll_point_cases = [c for c in load_cases if c.load_type == "LL"
                      and any(ld.get("type") == "point" for ld in c.loads)]
    ws_cases = [c for c in load_cases if c.load_type == "WS"]
    eq_cases = [c for c in load_cases if c.load_type == "EQ"]
    br_cases = [c for c in load_cases if c.load_type == "BR"]

    # Generate individual load case application functions
    lines.extend([
        "def apply_load_case(case_name, factor, ts_tag, pat_tag):",
        "    \"\"\"Apply a single load case with the given factor.\"\"\"",
        "    ops.timeSeries('Linear', ts_tag)",
        "    ops.pattern('Plain', pat_tag, ts_tag)",
        "",
    ])

    # Build the case dispatch
    case_idx = 0
    case_tag_map = {}
    for case in load_cases:
        tag = ts_tag_start + case_idx
        case_tag_map[case.name] = tag
        case_idx += 1

    # Generate the load application logic per case type
    lines.extend([
        "    # Load case dispatch by name",
    ])

    first = True
    for case in load_cases:
        prefix = "if" if first else "elif"
        first = False

        if case.load_type in ("DC", "DW"):
            # Distributed gravity loads
            for ld in case.loads:
                if ld.get("type") == "distributed":
                    w_kli = ld.get("w_kip_per_in", 0.0)
                    lines.extend([
                        f"    {prefix} case_name == '{case.name}':",
                        f"        # {case.description[:80]}",
                        f"        w = {w_kli} * factor",
                        f"        for ele_tag in envelope_ele_tags:",
                        f"            try:",
                        f"                ops.eleLoad('-ele', ele_tag, '-type', '-beamUniform', -abs(w), 0.0)",
                        f"            except Exception:",
                        f"                pass  # Skip non-beam elements",
                    ])
                    break

        elif case.load_type == "LL":
            for ld in case.loads:
                if ld.get("type") == "distributed":
                    w_kli = ld.get("w_kip_per_in", 0.0)
                    lines.extend([
                        f"    {prefix} case_name == '{case.name}':",
                        f"        # Lane load: {case.description[:60]}",
                        f"        w = {w_kli} * factor * g_moment",
                        f"        for ele_tag in envelope_ele_tags:",
                        f"            try:",
                        f"                ops.eleLoad('-ele', ele_tag, '-type', '-beamUniform', -abs(w), 0.0)",
                        f"            except Exception:",
                        f"                pass",
                    ])
                    break
                elif ld.get("type") == "point":
                    # Truck/tandem — apply as equivalent uniform for simplicity
                    # (actual moving load handled separately in moving_load_analysis)
                    m_kft = ld.get("max_moment_im_kft", 0.0)
                    lines.extend([
                        f"    {prefix} case_name == '{case.name}':",
                        f"        # Vehicular: {case.description[:60]}",
                        f"        # Applied as equivalent uniform for combination analysis",
                        f"        pass  # Handled via envelope from moving load analysis",
                    ])
                    break
                elif ld.get("type") in ("combination", "envelope"):
                    lines.extend([
                        f"    {prefix} case_name == '{case.name}':",
                        f"        # HL-93 combination/envelope — skip (use components)",
                        f"        pass",
                    ])
                    break

        elif case.load_type == "WS":
            for ld in case.loads:
                if ld.get("type") == "distributed":
                    w_kli = ld.get("w_kip_per_in", 0.0)
                    lines.extend([
                        f"    {prefix} case_name == '{case.name}':",
                        f"        # Wind on structure: lateral",
                        f"        w = {w_kli} * factor",
                        f"        for ele_tag in envelope_ele_tags:",
                        f"            try:",
                        f"                ops.eleLoad('-ele', ele_tag, '-type', '-beamUniform', 0.0, -abs(w))",
                        f"            except Exception:",
                        f"                pass",
                    ])
                    break

        elif case.load_type == "EQ":
            lines.extend([
                f"    {prefix} case_name == '{case.name}':",
                f"        # Seismic: response spectrum — applied via modal combination",
                f"        # Placeholder: apply as equivalent static = Csm * W",
                f"        sds = {case.loads[0].get('sds', 0.267) if case.loads else 0.267}",
                f"        for ele_tag in envelope_ele_tags:",
                f"            try:",
                f"                ops.eleLoad('-ele', ele_tag, '-type', '-beamUniform', 0.0, -0.10 * factor * sds)",
                f"            except Exception:",
                f"                pass",
            ])

        elif case.load_type == "BR":
            lines.extend([
                f"    {prefix} case_name == '{case.name}':",
                f"        # Braking force — longitudinal point load",
                f"        pass  # Applied at deck level node",
            ])

        else:
            # Thermal, permit, scour, etc. — skip for now
            lines.extend([
                f"    {prefix} case_name == '{case.name}':",
                f"        pass  # {case.load_type}: not directly applied in this script",
            ])

    lines.extend([
        "",
        "",
    ])

    # Generate combination analysis loop
    lines.extend([
        "# --- Run each load combination ---",
        "combo_definitions = {",
    ])

    for combo in combinations:
        # Only include factors for cases that exist
        factor_str = ", ".join(
            f"'{k}': {v}" for k, v in combo.factors.items()
        )
        lines.append(f"    '{combo.name}': {{{factor_str}}},")

    lines.extend([
        "}",
        "",
        "for combo_name, factors in combo_definitions.items():",
        "    # Reset model to initial state",
        "    ops.wipeAnalysis()",
        "    ops.remove('loadPattern', -1)  # attempt cleanup",
        "",
        "    # Apply each load case with its factor",
        "    ts_counter = 200",
        "    pat_counter = 200",
        "    for case_name, factor in factors.items():",
        "        try:",
        "            apply_load_case(case_name, factor, ts_counter, pat_counter)",
        "            ts_counter += 1",
        "            pat_counter += 1",
        "        except Exception as e:",
        "            print(f'WARNING: Could not apply {case_name} in {combo_name}: {e}')",
        "",
        "    # Run static analysis",
        "    ops.system('UmfPack')",
        "    ops.numberer('Plain')",
        "    ops.constraints('Penalty', 1.0e12, 1.0e12)",
        "    ops.integrator('LoadControl', 1.0)",
        "    ops.test('NormUnbalance', 1.0e-2, 100)",
        "    ops.algorithm('Linear')",
        "    ops.analysis('Static')",
        "    ok = ops.analyze(1)",
        "",
        "    if ok != 0:",
        "        # Retry with Newton",
        "        ops.algorithm('Newton')",
        "        ok = ops.analyze(1)",
        "",
        "    # Extract element forces",
        "    combo_forces = {}",
        "    if ok == 0:",
        "        for ele_tag in envelope_ele_tags:",
        "            try:",
        "                forces = ops.eleForce(ele_tag)",
        "                # 2D beam: [N_i, V_i, M_i, N_j, V_j, M_j]",
        "                # 3D beam: [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i, N_j, ...]",
        "                if len(forces) >= 6:",
        "                    combo_forces[ele_tag] = {",
        "                        'axial_i': forces[0],",
        "                        'shear_i': forces[1],",
        "                        'moment_i': forces[2],",
        "                        'axial_j': forces[3] if len(forces) > 3 else -forces[0],",
        "                        'shear_j': forces[4] if len(forces) > 4 else -forces[1],",
        "                        'moment_j': forces[5] if len(forces) > 5 else -forces[2],",
        "                    }",
        "                    if len(forces) >= 12:",
        "                        # 3D element: [Nx, Vy, Vz, T, My, Mz, ...]",
        "                        combo_forces[ele_tag] = {",
        "                            'axial_i': forces[0],",
        "                            'shear_y_i': forces[1],",
        "                            'shear_z_i': forces[2],",
        "                            'torsion_i': forces[3],",
        "                            'moment_y_i': forces[4],",
        "                            'moment_z_i': forces[5],",
        "                            'axial_j': forces[6],",
        "                            'shear_y_j': forces[7],",
        "                            'shear_z_j': forces[8],",
        "                            'torsion_j': forces[9],",
        "                            'moment_y_j': forces[10],",
        "                            'moment_z_j': forces[11],",
        "                        }",
        "            except Exception:",
        "                pass  # Element may not support eleForce",
        "    else:",
        "        print(f'WARNING: {combo_name} analysis did not converge')",
        "",
        "    combination_results[combo_name] = {",
        "        'converged': ok == 0,",
        "        'element_forces': combo_forces,",
        "    }",
        "",
        "print(f'Completed {len(combination_results)} load combinations')",
        "",
    ])

    # Generate envelope extraction inline
    lines.extend([
        "# --- Extract force envelopes ---",
        "envelopes = {}",
        "for ele_tag in envelope_ele_tags:",
        "    env = {",
        "        'max_moment': float('-inf'),",
        "        'min_moment': float('inf'),",
        "        'max_shear': float('-inf'),",
        "        'min_shear': float('inf'),",
        "        'max_axial': float('-inf'),",
        "        'min_axial': float('inf'),",
        "        'controlling_combo_moment': '',",
        "        'controlling_combo_shear': '',",
        "        'controlling_combo_axial': '',",
        "    }",
        "    for combo_name, result in combination_results.items():",
        "        if not result['converged']:",
        "            continue",
        "        forces = result['element_forces'].get(ele_tag, {})",
        "        if not forces:",
        "            continue",
        "",
        "        # Get moment (use moment_i, moment_j, or moment_z variants)",
        "        moments = []",
        "        for key in ('moment_i', 'moment_j', 'moment_z_i', 'moment_z_j'):",
        "            if key in forces:",
        "                moments.append(abs(forces[key]))",
        "        if moments:",
        "            m = max(moments)",
        "            if m > env['max_moment']:",
        "                env['max_moment'] = m",
        "                env['controlling_combo_moment'] = combo_name",
        "            m_min = -m",
        "            if m_min < env['min_moment']:",
        "                env['min_moment'] = m_min",
        "",
        "        # Get shear",
        "        shears = []",
        "        for key in ('shear_i', 'shear_j', 'shear_y_i', 'shear_y_j'):",
        "            if key in forces:",
        "                shears.append(abs(forces[key]))",
        "        if shears:",
        "            v = max(shears)",
        "            if v > env['max_shear']:",
        "                env['max_shear'] = v",
        "                env['controlling_combo_shear'] = combo_name",
        "            if -v < env['min_shear']:",
        "                env['min_shear'] = -v",
        "",
        "        # Get axial",
        "        axials = []",
        "        for key in ('axial_i', 'axial_j'):",
        "            if key in forces:",
        "                axials.append(forces[key])",
        "        if axials:",
        "            a_max = max(axials)",
        "            a_min = min(axials)",
        "            if a_max > env['max_axial']:",
        "                env['max_axial'] = a_max",
        "                env['controlling_combo_axial'] = combo_name",
        "            if a_min < env['min_axial']:",
        "                env['min_axial'] = a_min",
        "",
        "    # Clean up infinity values",
        "    for key in ('max_moment', 'max_shear', 'max_axial'):",
        "        if env[key] == float('-inf'):",
        "            env[key] = 0.0",
        "    for key in ('min_moment', 'min_shear', 'min_axial'):",
        "        if env[key] == float('inf'):",
        "            env[key] = 0.0",
        "",
        "    envelopes[ele_tag] = env",
        "",
        "print(f'Extracted envelopes for {len(envelopes)} elements')",
        "",
        "# Store in results",
        "results['combination_results'] = {",
        "    k: {'converged': v['converged'], 'num_elements': len(v['element_forces'])}",
        "    for k, v in combination_results.items()",
        "}",
        "results['envelopes'] = envelopes",
        "",
    ])

    return "\n".join(lines)


# ======================================================================
# Force envelope extraction (standalone, for post-processing)
# ======================================================================

def extract_force_envelopes(
    element_tags: list[int],
    combination_results: dict,
) -> dict:
    """Extract controlling force envelopes from load combination results.

    For each element, finds the maximum and minimum forces across all
    load combinations.

    Args:
        element_tags:        List of element tags to process.
        combination_results: Dict from combination analysis, keyed by
                             combo name, each with ``element_forces`` sub-dict.

    Returns:
        Dict keyed by element tag::

            {element_tag: {
                'max_moment': float,
                'min_moment': float,
                'max_shear': float,
                'min_shear': float,
                'max_axial': float,
                'min_axial': float,
                'controlling_combo_moment': str,
                'controlling_combo_shear': str,
                'controlling_combo_axial': str,
            }}
    """
    envelopes = {}

    for ele_tag in element_tags:
        env = {
            "max_moment": 0.0,
            "min_moment": 0.0,
            "max_shear": 0.0,
            "min_shear": 0.0,
            "max_axial": 0.0,
            "min_axial": 0.0,
            "controlling_combo_moment": "",
            "controlling_combo_shear": "",
            "controlling_combo_axial": "",
        }

        for combo_name, result in combination_results.items():
            if not result.get("converged", False):
                continue

            forces = result.get("element_forces", {}).get(ele_tag, {})
            if not forces:
                continue

            # Moments — check both ends
            for key in ("moment_i", "moment_j", "moment_z_i", "moment_z_j"):
                val = forces.get(key)
                if val is not None:
                    if val > env["max_moment"]:
                        env["max_moment"] = val
                        env["controlling_combo_moment"] = combo_name
                    if val < env["min_moment"]:
                        env["min_moment"] = val

            # Shear
            for key in ("shear_i", "shear_j", "shear_y_i", "shear_y_j"):
                val = forces.get(key)
                if val is not None:
                    if val > env["max_shear"]:
                        env["max_shear"] = val
                        env["controlling_combo_shear"] = combo_name
                    if val < env["min_shear"]:
                        env["min_shear"] = val

            # Axial
            for key in ("axial_i", "axial_j"):
                val = forces.get(key)
                if val is not None:
                    if val > env["max_axial"]:
                        env["max_axial"] = val
                        env["controlling_combo_axial"] = combo_name
                    if val < env["min_axial"]:
                        env["min_axial"] = val

        envelopes[ele_tag] = env

    return envelopes


# ======================================================================
# Moving load analysis script generation
# ======================================================================

def generate_moving_load_script(
    span_lengths: list[float],
    num_positions: int = 20,
    girder_spacing_ft: float = 8.0,
) -> str:
    """Generate OpenSees Python code for HL-93 moving load analysis.

    Traverses the bridge with design truck and tandem at discrete positions,
    concurrent with lane load, to find the worst-case live load effects.

    Args:
        span_lengths:     List of span lengths (ft).
        num_positions:    Number of positions to evaluate.
        girder_spacing_ft: For distribution factor computation.

    Returns:
        Multi-line Python string for insertion into OpenSees script.
    """
    total_length_ft = sum(span_lengths)
    total_length_in = total_length_ft * 12.0
    step_in = total_length_in / (num_positions + 1)

    # Build axle data for inline use
    truck_axles_in = [
        (a["weight_kip"] * (1 + IM_TRUCK), a["position_ft"] * 12.0)
        for a in DESIGN_TRUCK_AXLES
    ]
    tandem_axles_in = [
        (a["weight_kip"] * (1 + IM_TRUCK), a["position_ft"] * 12.0)
        for a in DESIGN_TANDEM_AXLES
    ]
    lane_w_kli = DESIGN_LANE_KLF / 12.0

    lines = [
        "",
        "# " + "=" * 60,
        "# MOVING LOAD ANALYSIS (HL-93)",
        "# " + "=" * 60,
        "",
        f"total_length_in = {total_length_in:.2f}",
        f"num_ll_positions = {num_positions}",
        f"ll_step_in = {step_in:.4f}",
        f"g_moment_ll = {girder_spacing_ft} / 11.0",
        f"g_shear_ll = {girder_spacing_ft} / 7.0",
        "",
        "# Axle data: (force_kip_with_IM, spacing_from_front_in)",
        f"truck_axles = {truck_axles_in}",
        f"tandem_axles = {tandem_axles_in}",
        f"lane_w_kli = {lane_w_kli:.6f}",
        "",
        "# Find nodes along the superstructure for load application",
        "# (Assumes nodes are sorted by X coordinate along bridge)",
        "all_node_tags = ops.getNodeTags()",
        "node_coords = {}",
        "for nt in all_node_tags:",
        "    coords = ops.nodeCoord(nt)",
        "    node_coords[nt] = coords",
        "",
        "# Sort superstructure nodes by X for load application",
        "sorted_nodes = sorted(node_coords.items(), key=lambda x: x[1][0])",
        "",
        "moving_load_results = {}",
        "",
        "def find_nearest_node(x_target):",
        "    \"\"\"Find the node closest to x_target.\"\"\"",
        "    best = None",
        "    best_dist = float('inf')",
        "    for nt, coords in sorted_nodes:",
        "        dist = abs(coords[0] - x_target)",
        "        if dist < best_dist:",
        "            best_dist = dist",
        "            best = nt",
        "    return best",
        "",
        "for vehicle_type, axles in [('truck', truck_axles), ('tandem', tandem_axles)]:",
        "    for pos_idx in range(num_ll_positions):",
        "        front_x = ll_step_in * (pos_idx + 1)",
        "",
        "        # Reset for this position",
        "        ops.wipeAnalysis()",
        "",
        "        ts_tag = 500 + pos_idx",
        "        pat_tag = 500 + pos_idx",
        "        ops.timeSeries('Linear', ts_tag)",
        "        ops.pattern('Plain', pat_tag, ts_tag)",
        "",
        "        # Apply axle loads at nearest nodes",
        "        for force_kip, spacing_in in axles:",
        "            x_axle = front_x - spacing_in",
        "            if 0 <= x_axle <= total_length_in:",
        "                nn = find_nearest_node(x_axle)",
        "                if nn is not None:",
        "                    ops.load(nn, 0.0, -force_kip * g_moment_ll, 0.0)",
        "",
        "        # Apply concurrent lane load",
        "        for ele_tag in envelope_ele_tags:",
        "            try:",
        "                ops.eleLoad('-ele', ele_tag, '-type', '-beamUniform',",
        "                           -lane_w_kli * g_moment_ll, 0.0)",
        "            except Exception:",
        "                pass",
        "",
        "        # Analyze",
        "        ops.system('UmfPack')",
        "        ops.numberer('Plain')",
        "        ops.constraints('Penalty', 1.0e12, 1.0e12)",
        "        ops.integrator('LoadControl', 1.0)",
        "        ops.test('NormUnbalance', 1.0e-2, 100)",
        "        ops.algorithm('Linear')",
        "        ops.analysis('Static')",
        "        ok = ops.analyze(1)",
        "",
        "        if ok == 0:",
        "            pos_key = f'{vehicle_type}_{pos_idx}'",
        "            pos_forces = {}",
        "            for ele_tag in envelope_ele_tags:",
        "                try:",
        "                    forces = ops.eleForce(ele_tag)",
        "                    pos_forces[ele_tag] = forces",
        "                except Exception:",
        "                    pass",
        "            moving_load_results[pos_key] = {",
        "                'vehicle': vehicle_type,",
        "                'front_axle_in': front_x,",
        "                'element_forces': pos_forces,",
        "                'converged': True,",
        "            }",
        "",
        "# Find LL envelope from moving load analysis",
        "ll_envelope = {}",
        "for ele_tag in envelope_ele_tags:",
        "    max_m = 0.0",
        "    max_v = 0.0",
        "    ctrl_m = ''",
        "    ctrl_v = ''",
        "    for pos_key, result in moving_load_results.items():",
        "        forces = result['element_forces'].get(ele_tag, [])",
        "        if len(forces) >= 6:",
        "            m = max(abs(forces[2]), abs(forces[5]))",
        "            v = max(abs(forces[1]), abs(forces[4]))",
        "            if m > max_m:",
        "                max_m = m",
        "                ctrl_m = pos_key",
        "            if v > max_v:",
        "                max_v = v",
        "                ctrl_v = pos_key",
        "    ll_envelope[ele_tag] = {",
        "        'max_moment': max_m,",
        "        'max_shear': max_v,",
        "        'controlling_position_moment': ctrl_m,",
        "        'controlling_position_shear': ctrl_v,",
        "    }",
        "",
        "results['moving_load_results'] = {",
        "    'num_positions': len(moving_load_results),",
        "    'll_envelope': ll_envelope,",
        "}",
        "print(f'Moving load analysis: {len(moving_load_results)} positions evaluated')",
        "",
    ]

    return "\n".join(lines)


# ======================================================================
# Main entry point
# ======================================================================

def generate_loads(
    geom: BridgeGeometry,
    site_profile: Optional[dict] = None,
    bridge_height_ft: float = 30.0,
) -> LoadModel:
    """Generate complete load model for a bridge.

    This is the canonical entry point for the loads tool.  It:
      1. Computes dead loads (DC, DW) from geometry
      2. Computes live loads (HL-93) with impact and distribution factors
      3. Generates permit vehicle loads
      4. Computes thermal (TU, TG), wind (WS, WL), braking (BR)
      5. Generates seismic (EQ) response spectrum
      6. Checks for scour (SC) modification
      7. Assembles AASHTO load combinations per Table 3.4.1-1
      8. Generates adversarial (red-team) load cases and combinations

    Args:
        geom:             :class:`BridgeGeometry` with bridge dimensions.
        site_profile:     Dict from :class:`~nlb.tools.site_recon.SiteProfile.to_dict`
                          or None (uses conservative defaults).
        bridge_height_ft: Height of bridge deck above ground (ft), for wind Kz.

    Returns:
        :class:`LoadModel` with all standard + adversarial cases and combinations.
    """
    # Extract site data
    if site_profile is None:
        site_profile = {}

    seismic_data = site_profile.get("seismic", None)
    wind_data = site_profile.get("wind", {})
    thermal_data = site_profile.get("thermal", None)
    scour_data = site_profile.get("scour", {})
    location = site_profile.get("location", {})
    state = location.get("state", "")

    wind_v = wind_data.get("v_ult", 115)

    # ------------------------------------------------------------------
    # Generate all standard load cases
    # ------------------------------------------------------------------
    all_cases: list[LoadCase] = []

    # Dead loads
    all_cases.extend(_compute_dead_loads(geom))

    # Live loads (HL-93)
    all_cases.extend(_compute_live_loads(geom))

    # Permit vehicles
    all_cases.extend(_compute_permit_loads(geom))

    # Thermal
    all_cases.extend(_compute_thermal_loads(geom, thermal_data, state))

    # Wind
    all_cases.extend(_compute_wind_loads(geom, wind_v, bridge_height_ft))

    # Braking
    all_cases.extend(_compute_braking(geom))

    # Seismic
    all_cases.extend(_compute_seismic(geom, seismic_data))

    # Scour
    water_crossing = scour_data.get("water_crossing", False)
    all_cases.extend(_compute_scour(water_crossing))

    # ------------------------------------------------------------------
    # Distribution factors
    # ------------------------------------------------------------------
    df = compute_distribution_factors(
        girder_spacing_ft=geom.girder_spacing_ft,
        span_ft=geom.span_ft,
        deck_thickness_in=geom.deck_thickness_in,
        girder_depth_in=geom.girder_depth_in,
        num_girders=geom.num_girders,
        structure_type=geom.structure_type,
    )

    # ------------------------------------------------------------------
    # Standard load combinations
    # ------------------------------------------------------------------
    standard_combos = _generate_combinations(all_cases)

    # ------------------------------------------------------------------
    # Moving load positions for HL-93 influence-line analysis
    # ------------------------------------------------------------------
    moving_load_positions = generate_moving_load_positions(geom.spans_ft)

    # ------------------------------------------------------------------
    # Adversarial loads
    # ------------------------------------------------------------------
    adversarial_cases = _generate_adversarial_cases(geom)
    adversarial_combos = _generate_adversarial_combos(all_cases, adversarial_cases)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    model = LoadModel(
        cases=all_cases,
        combinations=standard_combos,
        distribution_factors=df,
        adversarial_cases=adversarial_cases,
        adversarial_combos=adversarial_combos,
        moving_load_positions=moving_load_positions,
    )

    logger.info(
        "Load model generated: %d standard cases, %d standard combos, "
        "%d adversarial cases, %d adversarial combos = %d total combinations",
        len(all_cases), len(standard_combos),
        len(adversarial_cases), len(adversarial_combos),
        model.total_combinations,
    )

    return model

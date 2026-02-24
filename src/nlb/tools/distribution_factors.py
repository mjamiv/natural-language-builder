"""AASHTO LRFD Live-Load Distribution Factor Calculator.

Computes distribution factors for steel I-girder bridges with composite
concrete deck (Type "a" cross-section) per AASHTO LRFD 9th Ed.
Section 4.6.2.2.2.

References:
    AASHTO LRFD Bridge Design Specifications, 9th Edition
    Tables 4.6.2.2.2b-1, 4.6.2.2.2d-1, 4.6.2.2.3a-1, 4.6.2.2.3c-1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Multiple presence factors (AASHTO Table 3.6.1.1.2-1)
# NOTE: Already embedded in the tabulated 2+ lane formulas.
# Must be applied explicitly only for lever rule (1-lane loaded).
# ---------------------------------------------------------------------------
_MPF = {1: 1.20, 2: 1.00, 3: 0.85, 4: 0.65}

# AASHTO range of applicability for Type (a) cross-sections
_ROA = {
    "S_min": 3.5,   "S_max": 16.0,   # ft
    "L_min": 20.0,  "L_max": 240.0,  # ft
    "ts_min": 4.5,  "ts_max": 12.0,  # in
    "Nb_min": 4,
}


@dataclass
class DistributionFactors:
    """Live-load distribution factors for a single girder line.

    Governing values (``gM_int``, etc.) are the maximum of the 1-lane and
    2+-lane cases.  The AASHTO formulas already embed the multiple-presence
    factor (m=1.20 for 1 lane, m=1.00 for 2 lanes); the governing factor is
    therefore simply ``max(g_1lane, g_2plus)``.

    The lever-rule values (exterior, 1 lane) explicitly include m=1.20.
    """

    # --- Governing (max of 1-lane and 2+-lane) ----------------------------
    gM_int: float   # Interior girder moment DF
    gM_ext: float   # Exterior girder moment DF
    gV_int: float   # Interior girder shear DF
    gV_ext: float   # Exterior girder shear DF

    # --- Individual cases -------------------------------------------------
    gM_int_1: float   # Interior moment, 1 lane loaded
    gM_int_2: float   # Interior moment, 2+ lanes loaded
    gM_ext_1: float   # Exterior moment, 1 lane (lever rule × m=1.20)
    gM_ext_2: float   # Exterior moment, 2+ lanes (e × g_int_2)
    gV_int_1: float   # Interior shear, 1 lane loaded
    gV_int_2: float   # Interior shear, 2+ lanes loaded
    gV_ext_1: float   # Exterior shear, 1 lane (lever rule × m=1.20)
    gV_ext_2: float   # Exterior shear, 2+ lanes (e × g_int_2)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_kg(n: float, I_girder: float, A_girder: float, eg: float) -> float:
    """Longitudinal stiffness parameter per AASHTO 4.6.2.2.1-1.

    Args:
        n:        Modular ratio, Es / Ec (dimensionless).
        I_girder: Moment of inertia of the steel girder alone (in⁴).
        A_girder: Cross-sectional area of the steel girder alone (in²).
        eg:       Distance from girder centroid to centroid of deck (in).

    Returns:
        Kg in in⁴.
    """
    return n * (I_girder + A_girder * eg ** 2)


# ---------------------------------------------------------------------------
# Lever rule
# ---------------------------------------------------------------------------

def _lever_rule_exterior(S: float, de: float, m: float = 1.20) -> float:
    """Compute the exterior-girder DF using the lever rule.

    AASHTO 4.6.2.2.2d-1 note: "Lever Rule" for one design lane loaded.

    Lane placement (AASHTO 3.6.1.3.1):
      - First wheel at 2 ft from the interior face of the traffic barrier.
      - Second wheel 6 ft further toward interior (standard 6-ft wheel spacing).

    Geometry (all distances measured from exterior girder, positive toward
    interior of bridge):
      - ``de`` = distance from interior face of barrier to exterior web of
        exterior girder (AASHTO sign convention: positive when girder is
        inboard of barrier).
      - Wheel 1 at  (de - 2) ft from exterior girder.
      - Wheel 2 at  (de + 4) ft from exterior girder.

    The lever rule treats the deck as a simply-supported beam spanning *S* ft
    between the exterior and first interior girder.

    Args:
        S:  Girder spacing (ft).
        de: Distance from exterior girder to interior face of barrier (ft).
        m:  Multiple presence factor (default 1.20 for 1 lane).

    Returns:
        Distribution factor g (lanes per girder).
    """
    # Wheel positions from exterior girder (toward interior = positive)
    x1 = de - 2.0   # first wheel
    x2 = de + 4.0   # second wheel (x1 + 6 ft)

    # Reaction fractions via statics (moments about first interior girder)
    #   R_ext = P * (S - x) / S
    # Cantilever case (x < 0) gives R_ext > 1.0 (correct physically).
    # Cap the second wheel contribution at 0 if it falls past the interior girder.
    r1 = (S - x1) / S
    r2 = max(0.0, (S - x2) / S)

    return m * (r1 + r2)


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def compute_distribution_factors(
    S: float,
    L: float,
    ts: float,
    Kg: float,
    de: float = 2.0,
    Nb: int = 5,
    num_lanes: int = 0,
    roadway_width: float = 0.0,
    strict_roa: bool = False,
) -> DistributionFactors:
    """Compute AASHTO LRFD live-load distribution factors.

    Valid for Type (a) cross-sections: steel I-girder with composite concrete
    deck (AASHTO Tables 4.6.2.2.2b-1, 4.6.2.2.2d-1, 4.6.2.2.3a-1, 4.6.2.2.3c-1).

    The multiple-presence factor is **embedded** in both the 1-lane and 2+-lane
    tabulated formulas (m = 1.20 and 1.00 respectively), so the governing factor
    is simply ``max(g_1lane, g_2plus)`` with no additional scaling.

    For exterior girders with one lane, the lever rule is used directly and
    m = 1.20 is applied explicitly (it is not embedded in the formula).

    Args:
        S:             Girder spacing (ft), range 3.5–16.0.
        L:             Span length (ft), range 20–240.
        ts:            Structural slab thickness (in), range 4.5–12.0.
        Kg:            Longitudinal stiffness parameter (in⁴).
        de:            Distance from exterior girder to interior edge of
                       traffic barrier (ft).  Default 2.0.
        Nb:            Number of beams/girders (≥ 4).  Default 5.
        num_lanes:     Number of design lanes (0 = auto from roadway_width).
        roadway_width: Clear roadway width (ft), used when num_lanes == 0.
        strict_roa:    If True, raise ValueError when parameters fall outside
                       AASHTO range of applicability.  Default False (clamp).

    Returns:
        :class:`DistributionFactors` dataclass.

    Raises:
        ValueError: If ``Nb < 4``, or if ``strict_roa=True`` and any parameter
                    is outside the AASHTO range of applicability.
    """
    # --- Validate Nb -------------------------------------------------------
    if Nb < _ROA["Nb_min"]:
        raise ValueError(f"Nb={Nb} is below the AASHTO minimum of {_ROA['Nb_min']}.")

    # --- Range of applicability --------------------------------------------
    for param, val, lo, hi in [
        ("S",  S,  _ROA["S_min"],  _ROA["S_max"]),
        ("L",  L,  _ROA["L_min"],  _ROA["L_max"]),
        ("ts", ts, _ROA["ts_min"], _ROA["ts_max"]),
    ]:
        if strict_roa and (val < lo or val > hi):
            raise ValueError(
                f"{param}={val} is outside the AASHTO range of applicability "
                f"[{lo}, {hi}]."
            )

    # Clamp for formula stability (silently if strict_roa is False)
    S_c  = max(_ROA["S_min"],  min(S,  _ROA["S_max"]))
    L_c  = max(_ROA["L_min"],  min(L,  _ROA["L_max"]))
    ts_c = max(_ROA["ts_min"], min(ts, _ROA["ts_max"]))

    # --- Kg correction term ------------------------------------------------
    Kg_term = Kg / (12.0 * L_c * ts_c ** 3)

    # -----------------------------------------------------------------------
    # Interior Girder — Moment (AASHTO Table 4.6.2.2.2b-1, Type a)
    # -----------------------------------------------------------------------
    # One lane loaded (m = 1.20 embedded):
    gM_int_1 = (
        0.06
        + (S_c / 14.0) ** 0.4
        * (S_c / L_c) ** 0.3
        * Kg_term ** 0.1
    )

    # Two or more lanes loaded (m = 1.00 embedded):
    gM_int_2 = (
        0.075
        + (S_c / 9.5) ** 0.6
        * (S_c / L_c) ** 0.2
        * Kg_term ** 0.1
    )

    # -----------------------------------------------------------------------
    # Interior Girder — Shear (AASHTO Table 4.6.2.2.3a-1, Type a)
    # -----------------------------------------------------------------------
    gV_int_1 = 0.36 + S_c / 25.0
    gV_int_2 = 0.2 + S_c / 12.0 - (S_c / 35.0) ** 2

    # -----------------------------------------------------------------------
    # Exterior Girder — Moment (AASHTO Table 4.6.2.2.2d-1)
    # -----------------------------------------------------------------------
    # One lane: lever rule (m = 1.20 applied inside _lever_rule_exterior)
    gM_ext_1 = _lever_rule_exterior(S_c, de, m=_MPF[1])

    # Two or more lanes: e-factor method
    e_moment = 0.77 + de / 9.1
    gM_ext_2 = e_moment * gM_int_2

    # -----------------------------------------------------------------------
    # Exterior Girder — Shear (AASHTO Table 4.6.2.2.3c-1)
    # -----------------------------------------------------------------------
    # One lane: lever rule
    gV_ext_1 = _lever_rule_exterior(S_c, de, m=_MPF[1])

    # Two or more lanes: e-factor method
    e_shear = 0.6 + de / 10.0
    gV_ext_2 = e_shear * gV_int_2

    # -----------------------------------------------------------------------
    # Governing values (AASHTO formulas already embed MPF → take simple max)
    # -----------------------------------------------------------------------
    gM_int = max(gM_int_1, gM_int_2)
    gM_ext = max(gM_ext_1, gM_ext_2)
    gV_int = max(gV_int_1, gV_int_2)
    gV_ext = max(gV_ext_1, gV_ext_2)

    return DistributionFactors(
        gM_int=round(gM_int, 4),
        gM_ext=round(gM_ext, 4),
        gV_int=round(gV_int, 4),
        gV_ext=round(gV_ext, 4),
        gM_int_1=round(gM_int_1, 4),
        gM_int_2=round(gM_int_2, 4),
        gM_ext_1=round(gM_ext_1, 4),
        gM_ext_2=round(gM_ext_2, 4),
        gV_int_1=round(gV_int_1, 4),
        gV_int_2=round(gV_int_2, 4),
        gV_ext_1=round(gV_ext_1, 4),
        gV_ext_2=round(gV_ext_2, 4),
    )

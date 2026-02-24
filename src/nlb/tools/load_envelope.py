"""AASHTO LRFD Load Combination Envelope Engine.

Post-processing engine: takes per-case unfactored element forces and produces
AASHTO LRFD factored envelopes with controlling load combination tracking.

References
----------
AASHTO LRFD Bridge Design Specifications, 9th Edition:
  §3.4   Load Factors and Combinations (Table 3.4.1-1)
  §3.6.2 Fatigue Loads

Units
-----
Forces may be in any consistent unit system (kip, kN, etc.).
This module does *not* enforce units — the caller is responsible for using
a consistent system throughout.

Limit States Implemented
------------------------
- Strength I:  1.25DC + 1.50DW + 1.75(LL+IM)  [γDC_min=0.90, γDW_min=0.65]
- Strength III: 1.25DC + 1.50DW + 1.00WS
- Strength V:  1.25DC + 1.50DW + 1.35(LL+IM) + 0.40WS + 1.00WL
- Service I:   1.0DC + 1.0DW + 1.0(LL+IM) + 0.30WS + 1.0WL
- Service II:  1.0DC + 1.0DW + 1.3(LL+IM)
- Fatigue I:   1.75(LL+IM)  — single truck only, DC/DW excluded
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ======================================================================
# AASHTO Table 3.4.1-1 — Load factor definitions
# ======================================================================
# Format per case type:
#   scalar float  → same factor for both max/min load effect
#   (max, min)    → DC/DW/TU style: two variants generated
#   None          → load type not included in this limit state
#
# Variants with (max,min) tuples produce two combo instances:
#   "<LimitState>_max"  — use max factor (unfavorable for positive effects)
#   "<LimitState>_min"  — use min factor (favorable/unfavorable for reversed effects)

LIMIT_STATE_FACTORS: dict[str, dict[str, object]] = {
    "Strength_I": {
        "DC": (1.25, 0.90),
        "DW": (1.50, 0.65),
        "LL": 1.75,
        "WS": None,
        "WL": None,
        "TU": (0.50, 1.20),
        "BR": 1.75,
        "EQ": None,
    },
    "Strength_III": {
        "DC": (1.25, 0.90),
        "DW": (1.50, 0.65),
        "LL": None,
        "WS": 1.00,
        "WL": None,
        "TU": (0.50, 1.20),
        "BR": None,
        "EQ": None,
    },
    "Strength_V": {
        "DC": (1.25, 0.90),
        "DW": (1.50, 0.65),
        "LL": 1.35,
        "WS": 0.40,
        "WL": 1.00,
        "TU": (0.50, 1.20),
        "BR": None,
        "EQ": None,
    },
    "Service_I": {
        "DC": 1.00,
        "DW": 1.00,
        "LL": 1.00,
        "WS": 0.30,
        "WL": 1.00,
        "TU": (1.00, 1.20),
        "BR": None,
        "EQ": None,
    },
    "Service_II": {
        "DC": 1.00,
        "DW": 1.00,
        "LL": 1.30,
        "WS": None,
        "WL": None,
        "TU": None,
        "BR": None,
        "EQ": None,
    },
    "Fatigue_I": {
        # Only LL (single fatigue truck) — no DC, DW, or other permanents
        "DC": None,
        "DW": None,
        "LL": 1.75,
        "WS": None,
        "WL": None,
        "TU": None,
        "BR": None,
        "EQ": None,
    },
}

# ======================================================================
# Data classes
# ======================================================================


@dataclass
class CaseForces:
    """Unfactored element forces for a single load case.

    Attributes:
        case_name:      Unique identifier, e.g. ``"DC1_deck_slab"``.
        case_type:      AASHTO load type code: ``"DC"``, ``"DW"``, ``"LL"``,
                        ``"WS"``, ``"WL"``, ``"TU"``, ``"EQ"``, ``"BR"``.
        element_forces: Mapping of element tag → force dict.
                        Each force dict may contain any subset of:
                        ``"Mz_i"`` (moment at i-end, kip-ft or kip-in),
                        ``"Vy_i"`` (shear at i-end),
                        ``"N_i"``  (axial at i-end).
                        Missing keys are treated as zero.
    """

    case_name: str
    case_type: str  # "DC","DW","LL","WS","WL","TU","EQ","BR"
    element_forces: dict = field(default_factory=dict)
    # {elem_tag: {"Mz_i": val, "Vy_i": val, "N_i": val}}


@dataclass
class ForceEnvelope:
    """Factored force envelope for a single element.

    Stores the maximum and minimum factored force effects across all
    applicable AASHTO load combinations, together with the controlling
    combination name.

    Attributes:
        element_tag:   OpenSees element tag (integer key).
        Mz_max:        Maximum (most positive) factored bending moment.
        Mz_min:        Minimum (most negative) factored bending moment.
        Vy_max:        Maximum (most positive) factored shear.
        Vy_min:        Minimum (most negative) factored shear.
        N_max:         Maximum (most tensile/positive) factored axial force.
        N_min:         Minimum (most compressive/negative) factored axial force.
        Mz_max_combo:  Name of the controlling load combination for Mz_max.
        Mz_min_combo:  Name of the controlling load combination for Mz_min.
        Vy_max_combo:  Name of the controlling load combination for Vy_max.
        Vy_min_combo:  Name of the controlling load combination for Vy_min.
        N_max_combo:   Name of the controlling load combination for N_max.
        N_min_combo:   Name of the controlling load combination for N_min.
    """

    element_tag: int
    Mz_max: float = 0.0
    Mz_min: float = 0.0
    Vy_max: float = 0.0
    Vy_min: float = 0.0
    N_max: float = 0.0
    N_min: float = 0.0
    Mz_max_combo: str = ""
    Mz_min_combo: str = ""
    Vy_max_combo: str = ""
    Vy_min_combo: str = ""
    N_max_combo: str = ""
    N_min_combo: str = ""

    def to_dict(self) -> dict:
        """Serialize to plain dict (for JSON / report output)."""
        return {
            "element_tag": self.element_tag,
            "Mz_max": self.Mz_max,
            "Mz_min": self.Mz_min,
            "Vy_max": self.Vy_max,
            "Vy_min": self.Vy_min,
            "N_max": self.N_max,
            "N_min": self.N_min,
            "Mz_max_combo": self.Mz_max_combo,
            "Mz_min_combo": self.Mz_min_combo,
            "Vy_max_combo": self.Vy_max_combo,
            "Vy_min_combo": self.Vy_min_combo,
            "N_max_combo": self.N_max_combo,
            "N_min_combo": self.N_min_combo,
        }


# ======================================================================
# Internal helpers
# ======================================================================


def _resolve_limit_states(
    limit_states: Optional[list[str]],
) -> dict[str, dict[str, object]]:
    """Return the factor table filtered to the requested limit states.

    Parameters
    ----------
    limit_states:
        List of limit state keys (e.g. ``["Strength_I", "Service_II"]``).
        If ``None``, all limit states in :data:`LIMIT_STATE_FACTORS` are used.

    Returns
    -------
    dict
        Subset of :data:`LIMIT_STATE_FACTORS` matching the request.
    """
    if limit_states is None:
        return LIMIT_STATE_FACTORS

    resolved: dict[str, dict[str, object]] = {}
    for key in limit_states:
        if key in LIMIT_STATE_FACTORS:
            resolved[key] = LIMIT_STATE_FACTORS[key]
        else:
            logger.warning("Unknown limit state '%s' — skipped.", key)
    return resolved


def _expand_combos(
    factor_table: dict[str, dict[str, object]],
) -> list[tuple[str, dict[str, float]]]:
    """Expand the factor table into concrete combo variants.

    For limit states where DC (or DW/TU) has ``(max_factor, min_factor)``
    tuples, two variants are generated:
      * ``"<LimitState>_max"`` — all tuples resolved to their [0] (max) value
      * ``"<LimitState>_min"`` — all tuples resolved to their [1] (min) value

    For limit states without any tuple entries a single variant with the
    original name is produced.

    Parameters
    ----------
    factor_table:
        Mapping of limit state name → {case_type: factor_or_tuple_or_None}.

    Returns
    -------
    list of (combo_name, {case_type: resolved_factor})
        Ready-to-use combo definitions.
    """
    combos: list[tuple[str, dict[str, float]]] = []

    for ls_name, type_factors in factor_table.items():
        has_dual = any(
            isinstance(v, tuple) for v in type_factors.values() if v is not None
        )

        if has_dual:
            for suffix, idx in [("max", 0), ("min", 1)]:
                resolved: dict[str, float] = {}
                for case_type, fval in type_factors.items():
                    if fval is None:
                        continue
                    if isinstance(fval, tuple):
                        resolved[case_type] = float(fval[idx])
                    else:
                        resolved[case_type] = float(fval)
                combos.append((f"{ls_name}_{suffix}", resolved))
        else:
            resolved = {}
            for case_type, fval in type_factors.items():
                if fval is None:
                    continue
                resolved[case_type] = float(fval)  # type: ignore[arg-type]
            combos.append((ls_name, resolved))

    return combos


def _group_cases_by_type(
    case_forces: list[CaseForces],
) -> dict[str, list[CaseForces]]:
    """Group CaseForces objects by their case_type."""
    groups: dict[str, list[CaseForces]] = {}
    for cf in case_forces:
        groups.setdefault(cf.case_type, []).append(cf)
    return groups


def _collect_element_tags(case_forces: list[CaseForces]) -> set:
    """Collect all unique element tags across all load cases."""
    tags: set = set()
    for cf in case_forces:
        tags.update(cf.element_forces.keys())
    return tags


def _compute_combo_force(
    elem_tag: int,
    type_factors: dict[str, float],
    cases_by_type: dict[str, list[CaseForces]],
) -> tuple[float, float, float]:
    """Compute factored (Mz, Vy, N) for one element under one combo variant.

    Parameters
    ----------
    elem_tag:      Element tag to evaluate.
    type_factors:  {case_type: factor} for this combo variant.
    cases_by_type: Grouped CaseForces by type.

    Returns
    -------
    (Mz, Vy, N) — summed factored forces.
    """
    Mz = 0.0
    Vy = 0.0
    N = 0.0

    for case_type, factor in type_factors.items():
        for cf in cases_by_type.get(case_type, []):
            forces = cf.element_forces.get(elem_tag, {})
            Mz += factor * forces.get("Mz_i", 0.0)
            Vy += factor * forces.get("Vy_i", 0.0)
            N += factor * forces.get("N_i", 0.0)

    return Mz, Vy, N


# ======================================================================
# Public API
# ======================================================================


def compute_factored_envelopes(
    case_forces: list[CaseForces],
    limit_states: Optional[list[str]] = None,
) -> dict[int, ForceEnvelope]:
    """Compute AASHTO LRFD factored force envelopes for all elements.

    For each element found in ``case_forces`` and for each applicable load
    combination, sums ``γᵢ × unfactored_force_i`` over all contributing
    load cases.  Tracks the algebraic maximum and minimum of each force
    component together with the controlling combination name.

    DC (and DW, TU when applicable) are evaluated with *both* their maximum
    and minimum load factors to capture reversed-effect scenarios (e.g.
    hogging vs. sagging at continuous span supports).

    The LL "envelope" is naturally obtained by summing all CaseForces objects
    whose ``case_type == "LL"`` — callers should pass the LL cases that
    represent the governing truck/tandem positions.

    Parameters
    ----------
    case_forces:
        List of :class:`CaseForces` objects, one per load case.
        May contain multiple cases of the same type (e.g. multiple LL
        positions, multiple DC sub-cases) — all are summed for each combo.
    limit_states:
        Optional list of limit state keys to evaluate, e.g.
        ``["Strength_I", "Service_II"]``.  If ``None``, all six standard
        limit states are used.

    Returns
    -------
    dict[int, ForceEnvelope]
        Mapping of element tag → :class:`ForceEnvelope` with factored
        extremes and controlling combo labels.

    Examples
    --------
    >>> dc = CaseForces("DC1", "DC", {1: {"Mz_i": 100.0, "Vy_i": 20.0, "N_i": 0.0}})
    >>> ll = CaseForces("LL1", "LL", {1: {"Mz_i": 200.0, "Vy_i": 40.0, "N_i": 0.0}})
    >>> envs = compute_factored_envelopes([dc, ll])
    >>> envs[1].Mz_max_combo
    'Strength_I_max'
    """
    if not case_forces:
        return {}

    factor_table = _resolve_limit_states(limit_states)
    combos = _expand_combos(factor_table)

    if not combos:
        return {}

    cases_by_type = _group_cases_by_type(case_forces)
    all_elements = _collect_element_tags(case_forces)

    envelopes: dict[int, ForceEnvelope] = {}

    for elem_tag in all_elements:
        Mz_max = float("-inf")
        Mz_min = float("inf")
        Vy_max = float("-inf")
        Vy_min = float("inf")
        N_max = float("-inf")
        N_min = float("inf")
        Mz_max_combo = ""
        Mz_min_combo = ""
        Vy_max_combo = ""
        Vy_min_combo = ""
        N_max_combo = ""
        N_min_combo = ""

        for combo_name, type_factors in combos:
            Mz, Vy, N = _compute_combo_force(
                elem_tag, type_factors, cases_by_type
            )

            if Mz > Mz_max:
                Mz_max = Mz
                Mz_max_combo = combo_name
            if Mz < Mz_min:
                Mz_min = Mz
                Mz_min_combo = combo_name

            if Vy > Vy_max:
                Vy_max = Vy
                Vy_max_combo = combo_name
            if Vy < Vy_min:
                Vy_min = Vy
                Vy_min_combo = combo_name

            if N > N_max:
                N_max = N
                N_max_combo = combo_name
            if N < N_min:
                N_min = N
                N_min_combo = combo_name

        # Resolve -inf / +inf sentinels (element had no forces at all)
        if Mz_max == float("-inf"):
            Mz_max = 0.0
        if Mz_min == float("inf"):
            Mz_min = 0.0
        if Vy_max == float("-inf"):
            Vy_max = 0.0
        if Vy_min == float("inf"):
            Vy_min = 0.0
        if N_max == float("-inf"):
            N_max = 0.0
        if N_min == float("inf"):
            N_min = 0.0

        envelopes[elem_tag] = ForceEnvelope(
            element_tag=elem_tag,
            Mz_max=Mz_max,
            Mz_min=Mz_min,
            Vy_max=Vy_max,
            Vy_min=Vy_min,
            N_max=N_max,
            N_min=N_min,
            Mz_max_combo=Mz_max_combo,
            Mz_min_combo=Mz_min_combo,
            Vy_max_combo=Vy_max_combo,
            Vy_min_combo=Vy_min_combo,
            N_max_combo=N_max_combo,
            N_min_combo=N_min_combo,
        )

        logger.debug(
            "Element %d: Mz=[%.3f, %.3f] (%s / %s), "
            "Vy=[%.3f, %.3f] (%s / %s)",
            elem_tag,
            Mz_max, Mz_min, Mz_max_combo, Mz_min_combo,
            Vy_max, Vy_min, Vy_max_combo, Vy_min_combo,
        )

    logger.info(
        "Factored envelopes: %d elements, %d limit states, %d combo variants",
        len(envelopes),
        len(factor_table),
        len(combos),
    )

    return envelopes


def compute_dcr_from_envelopes(
    envelopes: dict[int, ForceEnvelope],
    section_capacities: "dict | list",
) -> list[dict]:
    """Compute demand-to-capacity ratios (DCR) from factored envelopes.

    Parameters
    ----------
    envelopes:
        Output of :func:`compute_factored_envelopes`.
    section_capacities:
        Either:

        * A **dict** keyed by element tag →
          ``{"Mn": float, "Vn": float, "Pn": float}``,
          where Mn is nominal moment capacity, Vn shear, Pn axial.
          Any key may be omitted — missing capacities are simply not
          checked.

        * A **list** of dicts each containing an ``"element_tag"`` key
          plus ``"Mn"``, ``"Vn"``, ``"Pn"`` as above.

    Returns
    -------
    list[dict]
        One entry per element in ``envelopes``.  Each dict contains:

        * ``"element_tag"``
        * ``"DCR_Mz"``, ``"DCR_Mz_max"``, ``"DCR_Mz_min"``  (if Mn given)
        * ``"DCR_Vy"``, ``"DCR_Vy_max"``, ``"DCR_Vy_min"``  (if Vn given)
        * ``"DCR_N"``,  ``"DCR_N_max"``,  ``"DCR_N_min"``   (if Pn given)
        * ``"DCR_max"``  — overall maximum DCR for this element
        * ``"status"``   — ``"OVERSTRESSED"`` if DCR_max > 1.0, else ``"OK"``

    Raises
    ------
    TypeError
        If ``section_capacities`` is not a dict or list.
    """
    # Normalise section_capacities to {element_tag: cap_dict}
    if isinstance(section_capacities, dict):
        caps: dict[int, dict] = section_capacities  # type: ignore[assignment]
    elif isinstance(section_capacities, list):
        caps = {}
        for item in section_capacities:
            tag = item.get("element_tag")
            if tag is not None:
                caps[tag] = item
    else:
        raise TypeError(
            f"section_capacities must be dict or list, got {type(section_capacities)}"
        )

    results: list[dict] = []

    for elem_tag, env in envelopes.items():
        cap = caps.get(elem_tag, {})

        result: dict = {"element_tag": elem_tag}

        Mn = cap.get("Mn")
        Vn = cap.get("Vn")
        Pn = cap.get("Pn")

        if Mn is not None and Mn > 0:
            dcr_mz_max = abs(env.Mz_max) / Mn
            dcr_mz_min = abs(env.Mz_min) / Mn
            result["DCR_Mz_max"] = dcr_mz_max
            result["DCR_Mz_min"] = dcr_mz_min
            result["DCR_Mz"] = max(dcr_mz_max, dcr_mz_min)
            result["Mz_demand_max"] = env.Mz_max
            result["Mz_demand_min"] = env.Mz_min
            result["Mz_capacity"] = Mn
            result["Mz_max_combo"] = env.Mz_max_combo
            result["Mz_min_combo"] = env.Mz_min_combo

        if Vn is not None and Vn > 0:
            dcr_vy_max = abs(env.Vy_max) / Vn
            dcr_vy_min = abs(env.Vy_min) / Vn
            result["DCR_Vy_max"] = dcr_vy_max
            result["DCR_Vy_min"] = dcr_vy_min
            result["DCR_Vy"] = max(dcr_vy_max, dcr_vy_min)
            result["Vy_demand_max"] = env.Vy_max
            result["Vy_demand_min"] = env.Vy_min
            result["Vy_capacity"] = Vn
            result["Vy_max_combo"] = env.Vy_max_combo
            result["Vy_min_combo"] = env.Vy_min_combo

        if Pn is not None and Pn > 0:
            dcr_n_max = abs(env.N_max) / Pn
            dcr_n_min = abs(env.N_min) / Pn
            result["DCR_N_max"] = dcr_n_max
            result["DCR_N_min"] = dcr_n_min
            result["DCR_N"] = max(dcr_n_max, dcr_n_min)
            result["N_demand_max"] = env.N_max
            result["N_demand_min"] = env.N_min
            result["N_capacity"] = Pn
            result["N_max_combo"] = env.N_max_combo
            result["N_min_combo"] = env.N_min_combo

        # Overall maximum DCR (across all checked components)
        dcr_values = [
            v
            for k, v in result.items()
            if k.startswith("DCR_") and not k.endswith("_max") and not k.endswith("_min")
        ]
        if dcr_values:
            result["DCR_max"] = max(dcr_values)
            result["status"] = "OVERSTRESSED" if result["DCR_max"] > 1.0 else "OK"
        else:
            result["DCR_max"] = None
            result["status"] = "NO_CAPACITY"

        results.append(result)

    return results

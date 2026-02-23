"""Red Team Engine — Adversarial Analysis of Bridge Designs.

The red team engine is the core differentiator of the Natural Language Builder.
It takes analysis results and ATTACKS the design through 7 distinct attack
vectors, finding vulnerabilities that standard practice misses.

Attack Vectors
--------------
1. **DCR Scanner** — Demand/capacity ratio scan across all elements
2. **Failure Cascade** — Progressive collapse / alternate load path analysis
3. **Construction Vulnerability** — Staged construction risk assessment
4. **Sensitivity Sweep** — Parameter variation tornado analysis
5. **Extreme Event Combiner** — Adversarial load combination assessment
6. **Robustness Check** — Systematic component removal analysis
7. **History Matcher** — Comparison against real bridge failure database

Risk Rating
-----------
* **RED:** Any CRITICAL finding (DCR > 1.0) OR cascade causing collapse
* **YELLOW:** Any WARNING (DCR > 0.85) OR dominant sensitivity OR adversarial
  exceedance > 15%
* **GREEN:** All DCRs < 0.85, no cascades, no dominant sensitivities

Units
-----
All internal units: kip, inch, second (KIS) — consistent with OpenSees.

References
----------
AASHTO LRFD Bridge Design Specifications, 9th Edition:
  §1.3.2  Limit States
  §3.4    Load Factors and Combinations
  §C1.3.5 Redundancy
GSA Progressive Collapse Analysis and Design Guidelines (2013)
UFC 4-023-03: Design of Buildings to Resist Progressive Collapse (2009)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FAILURES_JSON = _DATA_DIR / "failures.json"

# ---------------------------------------------------------------------------
# Severity ordering for sorting
# ---------------------------------------------------------------------------
_SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "NOTE": 2}

# ---------------------------------------------------------------------------
# Sensitivity classification thresholds
# ---------------------------------------------------------------------------
SENSITIVITY_DOMINANT_THRESHOLD = 0.10    # >10% DCR change
SENSITIVITY_MODERATE_THRESHOLD = 0.05    # 5-10% DCR change
# Below 5% → INSENSITIVE

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single red-team finding.

    Attributes:
        severity:          ``'CRITICAL'``, ``'WARNING'``, or ``'NOTE'``.
        vector:            Which attack vector found this.
        element:           Element tag (if applicable), else None.
        location:          Human-readable location ("Span 2, 0.4L").
        description:       What was found.
        dcr:               Demand/capacity ratio (if applicable).
        controlling_combo: Load combination that governs.
        recommendation:    What to do about it.
        precedent:         Historical failure match (if any).
    """
    severity: str
    vector: str
    element: int | None
    location: str
    description: str
    dcr: float | None
    controlling_combo: str
    recommendation: str
    precedent: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CascadeChain:
    """A progressive collapse chain.

    Attributes:
        trigger_element: The element whose failure starts the chain.
        chain:           Ordered list of (element, dcr) tuples in cascade.
        causes_collapse: True if the chain leads to global instability.
        description:     Human-readable cascade narrative.
    """
    trigger_element: int
    chain: list[tuple[int, float]] = field(default_factory=list)
    causes_collapse: bool = False
    description: str = ""


@dataclass
class SensitivityResult:
    """Result of varying one parameter in the sensitivity sweep.

    Attributes:
        parameter:      Name of the varied parameter.
        base_dcr:       DCR at baseline parameter value.
        low_dcr:        DCR at parameter -20%.
        high_dcr:       DCR at parameter +20%.
        delta_dcr:      Maximum absolute change in DCR.
        classification: ``'DOMINANT'``, ``'MODERATE'``, or ``'INSENSITIVE'``.
    """
    parameter: str
    base_dcr: float
    low_dcr: float
    high_dcr: float
    delta_dcr: float
    classification: str


@dataclass
class HistoryMatch:
    """A match against the bridge failure database.

    Attributes:
        failure_name: Name of the historical failure.
        year:         Year of the failure.
        score:        Similarity score (higher = more similar).
        lesson:       Key engineering lesson from the failure.
        matching_factors: What matched (list of factor descriptions).
    """
    failure_name: str
    year: int
    score: int
    lesson: str
    matching_factors: list[str] = field(default_factory=list)


@dataclass
class RedTeamReport:
    """Complete red-team analysis report.

    Attributes:
        findings:            All findings across all attack vectors.
        risk_rating:         ``'GREEN'``, ``'YELLOW'``, or ``'RED'``.
        summary:             1-paragraph executive summary.
        attack_vectors_run:  List of attack vector names executed.
        total_load_cases:    Total number of load cases analyzed.
        total_combinations:  Total number of load combinations checked.
        analysis_time_sec:   Wall-clock time for the full red-team run.
        cascade_chains:      Progressive collapse chains (from vector 2).
        sensitivity_results: Tornado diagram data (from vector 4).
        history_matches:     Failure database matches (from vector 7).
        robustness_results:  Component removal results (from vector 6).
    """
    findings: list[Finding] = field(default_factory=list)
    risk_rating: str = "GREEN"
    summary: str = ""
    attack_vectors_run: list[str] = field(default_factory=list)
    total_load_cases: int = 0
    total_combinations: int = 0
    analysis_time_sec: float = 0.0
    cascade_chains: list[CascadeChain] = field(default_factory=list)
    sensitivity_results: list[SensitivityResult] = field(default_factory=list)
    history_matches: list[HistoryMatch] = field(default_factory=list)
    robustness_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ===================================================================
# FAILURE DATABASE
# ===================================================================

def load_failure_database() -> list[dict]:
    """Load the bridge failure database from JSON.

    Returns:
        List of failure records with keys: name, year, type, spans,
        material, failure_mode, cause, lesson, details.
    """
    try:
        with open(_FAILURES_JSON, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load failure database: %s", exc)
        return []


# ===================================================================
# VECTOR 1: DCR SCANNER
# ===================================================================

def dcr_scanner(
    element_results: list[dict],
    critical_threshold: float = 1.0,
    warning_threshold: float = 0.85,
    note_threshold: float = 0.70,
) -> list[Finding]:
    """Scan all elements for demand/capacity ratio exceedances.

    Classifies each element as CRITICAL (DCR > 1.0), WARNING (> 0.85),
    or NOTE (> 0.70). Elements below 0.70 are not flagged.

    Args:
        element_results: List of dicts, each with:
            - ``element``: int — element tag
            - ``location``: str — human-readable location
            - ``dcr``: float — demand/capacity ratio
            - ``force_type``: str — "moment", "shear", or "axial"
            - ``controlling_combo``: str — load combination name
            - ``demand``: float (optional) — demand value
            - ``capacity``: float (optional) — capacity value
        critical_threshold: DCR threshold for CRITICAL. Default 1.0.
        warning_threshold:  DCR threshold for WARNING. Default 0.85.
        note_threshold:     DCR threshold for NOTE. Default 0.70.

    Returns:
        Sorted list of Finding objects (severity descending, then DCR descending).
    """
    findings: list[Finding] = []

    for er in element_results:
        dcr = er.get("dcr", 0.0)
        if dcr is None:
            continue

        element = er.get("element")
        location = er.get("location", f"Element {element}")
        force_type = er.get("force_type", "unknown")
        combo = er.get("controlling_combo", "unknown")
        demand = er.get("demand")
        capacity = er.get("capacity")

        if dcr > critical_threshold:
            severity = "CRITICAL"
            desc = (
                f"FAILS — DCR = {dcr:.3f} for {force_type}. "
                f"Demand exceeds capacity by {(dcr - 1.0) * 100:.1f}%."
            )
            if demand is not None and capacity is not None:
                desc += f" Demand = {demand:.1f}, Capacity = {capacity:.1f}."
            recommendation = (
                f"Increase section capacity or reduce demand. "
                f"Check {force_type} design for {location}."
            )
        elif dcr > warning_threshold:
            severity = "WARNING"
            margin = (1.0 - dcr) * 100
            desc = (
                f"Close call — DCR = {dcr:.3f} for {force_type}. "
                f"Only {margin:.1f}% margin to failure."
            )
            recommendation = (
                f"Consider increasing capacity. Margin is thin — "
                f"any load increase or section loss could cause failure."
            )
        elif dcr > note_threshold:
            severity = "NOTE"
            desc = (
                f"DCR = {dcr:.3f} for {force_type}. "
                f"Within acceptable range but worth monitoring."
            )
            recommendation = "Monitor during service. No immediate action required."
        else:
            continue

        findings.append(Finding(
            severity=severity,
            vector="DCR Scanner",
            element=element,
            location=location,
            description=desc,
            dcr=dcr,
            controlling_combo=combo,
            recommendation=recommendation,
        ))

    # Sort: severity order first (CRITICAL < WARNING < NOTE by sort value),
    # then DCR descending within each severity
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), -(f.dcr or 0)))
    return findings


# ===================================================================
# VECTOR 2: FAILURE CASCADE
# ===================================================================

def failure_cascade(
    element_results: list[dict],
    analyze_fn=None,
    dcr_threshold: float = 1.0,
    progressive_collapse_combo: str = "DC + 0.5×LL",
) -> tuple[list[Finding], list[CascadeChain]]:
    """Analyze progressive collapse by removing failed elements.

    For each element with DCR above the threshold, conceptually removes it
    (sets stiffness to near-zero) and re-analyzes to find chain reactions.

    Args:
        element_results:  Element DCR results (same format as dcr_scanner).
        analyze_fn:       Callable(removed_elements: list[int]) -> list[dict].
                          Re-runs analysis with specified elements removed and
                          returns updated element_results. If None, uses a
                          simplified estimation model.
        dcr_threshold:    DCR threshold to trigger cascade analysis.
        progressive_collapse_combo: Name of the load combo used.

    Returns:
        Tuple of (findings, cascade_chains).
    """
    findings: list[Finding] = []
    chains: list[CascadeChain] = []

    # Identify elements that trigger cascade analysis
    trigger_elements = [
        er for er in element_results
        if (er.get("dcr") or 0) > dcr_threshold
    ]

    if not trigger_elements:
        return findings, chains

    for trigger in trigger_elements:
        trigger_elem = trigger.get("element")
        trigger_loc = trigger.get("location", f"Element {trigger_elem}")
        trigger_dcr = trigger.get("dcr", 0.0)

        chain = CascadeChain(trigger_element=trigger_elem)

        if analyze_fn is not None:
            # Re-analyze with element removed
            try:
                updated_results = analyze_fn([trigger_elem])
                # Check for new failures
                for ur in updated_results:
                    ur_dcr = ur.get("dcr", 0.0)
                    if ur_dcr and ur_dcr > 1.0 and ur.get("element") != trigger_elem:
                        chain.chain.append((ur["element"], ur_dcr))
            except Exception as exc:
                logger.warning("Cascade analysis failed for element %s: %s",
                               trigger_elem, exc)
                chain.causes_collapse = True
                chain.description = (
                    f"Analysis became unstable after removing {trigger_loc} "
                    f"— indicates potential global collapse."
                )
        else:
            # Simplified estimation: elements near the trigger get
            # increased demand proportional to proximity
            _estimate_cascade(trigger, element_results, chain)

        # Build description
        if chain.chain:
            chain_desc_parts = [f"{trigger_loc} fails (DCR {trigger_dcr:.2f})"]
            for elem, d in chain.chain:
                chain_desc_parts.append(f"Element {elem} overloads (DCR {d:.2f})")
            chain.description = " → ".join(chain_desc_parts)

            if len(chain.chain) >= 3 or chain.causes_collapse:
                chain.causes_collapse = True

        chains.append(chain)

        # Generate finding
        if chain.causes_collapse:
            severity = "CRITICAL"
            desc = (
                f"Progressive collapse: removal of {trigger_loc} causes "
                f"chain failure of {len(chain.chain)} additional elements. "
                f"Chain: {chain.description}"
            )
            recommendation = (
                "Add redundancy or alternate load paths. Consider ductile "
                "detailing per AASHTO §1.3.4 for improved robustness."
            )
        elif chain.chain:
            severity = "WARNING"
            desc = (
                f"Cascade risk: removal of {trigger_loc} overloads "
                f"{len(chain.chain)} additional element(s). "
                f"Chain: {chain.description}"
            )
            recommendation = (
                "Verify alternate load path capacity. Consider adding "
                "redundant members."
            )
        else:
            severity = "NOTE"
            desc = (
                f"Element {trigger_loc} fails (DCR {trigger_dcr:.2f}) but "
                f"no cascade propagation detected."
            )
            recommendation = "Address the individual element failure."

        findings.append(Finding(
            severity=severity,
            vector="Failure Cascade",
            element=trigger_elem,
            location=trigger_loc,
            description=desc,
            dcr=trigger_dcr,
            controlling_combo=progressive_collapse_combo,
            recommendation=recommendation,
        ))

    return findings, chains


def _estimate_cascade(
    trigger: dict,
    all_results: list[dict],
    chain: CascadeChain,
) -> None:
    """Simplified cascade estimation without re-analysis.

    Estimates demand redistribution based on element adjacency and
    current utilization. Elements already near capacity are most
    vulnerable to cascade.
    """
    trigger_elem = trigger.get("element", 0)
    trigger_dcr = trigger.get("dcr", 0.0)

    # Estimate redistribution: nearby elements pick up ~30% more load
    redistribution_factor = 0.30

    for er in all_results:
        elem = er.get("element")
        if elem == trigger_elem:
            continue
        current_dcr = er.get("dcr", 0.0)
        if current_dcr is None:
            continue

        # Simple adjacency: elements within ±2 tags are "adjacent"
        if elem is not None and trigger_elem is not None:
            if abs(elem - trigger_elem) <= 2:
                new_dcr = current_dcr * (1.0 + redistribution_factor)
                if new_dcr > 1.0:
                    chain.chain.append((elem, round(new_dcr, 3)))

    # If many elements cascade, flag as potential collapse
    if len(chain.chain) >= 3:
        chain.causes_collapse = True


# ===================================================================
# VECTOR 3: CONSTRUCTION VULNERABILITY
# ===================================================================

def construction_vulnerability(
    stage_results: list[dict],
    site_constraints: dict | None = None,
) -> list[Finding]:
    """Analyze vulnerability during construction stages.

    Args:
        stage_results: List of dicts with:
            - ``stage_name``: str — name of the construction stage
            - ``stage_number``: int — sequential stage number
            - ``max_dcr``: float — maximum DCR across all elements in this stage
            - ``max_dcr_element``: int — element with highest DCR
            - ``max_dcr_location``: str — location of highest DCR element
            - ``requires_temp_support``: bool — whether temp supports needed
            - ``description``: str — description of what happens in this stage
        site_constraints: Dict of site constraints, e.g.:
            - ``no_equipment_in_water``: bool
            - ``restricted_access``: list[str]
            - ``night_work_only``: bool

    Returns:
        List of Finding objects for construction vulnerabilities.
    """
    findings: list[Finding] = []

    if not stage_results:
        return findings

    if site_constraints is None:
        site_constraints = {}

    # Find the most vulnerable stage
    worst_stage = max(stage_results, key=lambda s: s.get("max_dcr", 0.0))
    worst_dcr = worst_stage.get("max_dcr", 0.0)
    worst_name = worst_stage.get("stage_name", "Unknown")
    worst_num = worst_stage.get("stage_number", 0)
    worst_elem = worst_stage.get("max_dcr_element")
    worst_loc = worst_stage.get("max_dcr_location", f"Element {worst_elem}")

    if worst_dcr > 1.0:
        severity = "CRITICAL"
        desc = (
            f"Stage {worst_num} ({worst_name}) FAILS — "
            f"Element at {worst_loc} has DCR = {worst_dcr:.3f}. "
            f"Structure cannot support construction loads in this configuration."
        )
        recommendation = (
            "Redesign construction sequence. Add temporary supports or "
            "change erection method to reduce demands during this stage."
        )
    elif worst_dcr > 0.85:
        severity = "WARNING"
        desc = (
            f"Stage {worst_num} ({worst_name}) is the most vulnerable — "
            f"Element at {worst_loc} has DCR = {worst_dcr:.3f}. "
            f"Only {(1.0 - worst_dcr) * 100:.1f}% margin during construction."
        )
        recommendation = (
            "Consider adding temporary supports or adjusting the pour sequence "
            "to reduce peak demands during this stage."
        )
    elif worst_dcr > 0.70:
        severity = "NOTE"
        desc = (
            f"Stage {worst_num} ({worst_name}) is the most vulnerable — "
            f"Element at {worst_loc} has DCR = {worst_dcr:.3f}."
        )
        recommendation = "Monitor during construction. Acceptable margin exists."
    else:
        # All stages look fine, but still report most vulnerable
        severity = "NOTE"
        desc = (
            f"All construction stages within acceptable limits. "
            f"Most vulnerable: Stage {worst_num} ({worst_name}), "
            f"max DCR = {worst_dcr:.3f}."
        )
        recommendation = "No special measures required for construction."

    findings.append(Finding(
        severity=severity,
        vector="Construction Vulnerability",
        element=worst_elem,
        location=worst_loc,
        description=desc,
        dcr=worst_dcr,
        controlling_combo=f"Construction Stage {worst_num}",
        recommendation=recommendation,
    ))

    # Check for stages requiring temporary supports
    for stage in stage_results:
        if stage.get("requires_temp_support", False):
            stage_name = stage.get("stage_name", "Unknown")
            stage_num = stage.get("stage_number", 0)

            # Cross-reference with site constraints
            constraint_conflict = ""
            if site_constraints.get("no_equipment_in_water") and "pier" in stage_name.lower():
                constraint_conflict = (
                    " CONFLICT: Site constraint prohibits equipment in water, "
                    "but temporary supports may require in-water work."
                )

            findings.append(Finding(
                severity="WARNING" if constraint_conflict else "NOTE",
                vector="Construction Vulnerability",
                element=None,
                location=f"Stage {stage_num}",
                description=(
                    f"Stage {stage_num} ({stage_name}) requires temporary supports."
                    f"{constraint_conflict}"
                ),
                dcr=stage.get("max_dcr"),
                controlling_combo=f"Construction Stage {stage_num}",
                recommendation=(
                    "Verify temporary support design. Coordinate with contractor "
                    "for support locations and capacities."
                ),
            ))

    return findings


# ===================================================================
# VECTOR 4: SENSITIVITY SWEEP
# ===================================================================

# Default parameters and their variation ranges
SENSITIVITY_PARAMETERS = [
    {"name": "Soil Stiffness", "key": "soil_stiffness", "variation": 0.20},
    {"name": "Concrete Strength (f'c)", "key": "fc", "variation": 0.20},
    {"name": "Steel Yield Strength (fy)", "key": "fy", "variation": 0.20},
    {"name": "Bearing Friction", "key": "bearing_friction", "variation": 0.20},
    {"name": "Scour Depth", "key": "scour_depth", "variation": 0.20},
    {"name": "Thermal Range", "key": "thermal_range", "variation": 0.20},
    {"name": "Live Load Magnitude", "key": "live_load", "variation": 0.20},
]


def sensitivity_sweep(
    base_dcr: float,
    analyze_fn=None,
    parameter_results: list[dict] | None = None,
) -> tuple[list[Finding], list[SensitivityResult]]:
    """Vary key parameters ±20% and classify sensitivity.

    Can operate in two modes:
    1. With ``analyze_fn``: calls function for each parameter variation
    2. With ``parameter_results``: uses pre-computed results

    Args:
        base_dcr:          DCR at baseline parameter values.
        analyze_fn:        Callable(param_key: str, factor: float) -> float.
                           Returns max DCR with the parameter scaled by factor.
                           factor < 1.0 means reduced, > 1.0 means increased.
        parameter_results: Pre-computed list of dicts with:
            - ``parameter``: str — parameter name
            - ``low_dcr``: float — DCR at -20%
            - ``high_dcr``: float — DCR at +20%

    Returns:
        Tuple of (findings, sensitivity_results).
    """
    findings: list[Finding] = []
    results: list[SensitivityResult] = []

    if parameter_results is not None:
        # Use pre-computed results
        for pr in parameter_results:
            name = pr["parameter"]
            low_dcr = pr["low_dcr"]
            high_dcr = pr["high_dcr"]
            _process_sensitivity(name, base_dcr, low_dcr, high_dcr,
                                 findings, results)
    elif analyze_fn is not None:
        for param in SENSITIVITY_PARAMETERS:
            name = param["name"]
            key = param["key"]
            variation = param["variation"]

            try:
                low_dcr = analyze_fn(key, 1.0 - variation)
                high_dcr = analyze_fn(key, 1.0 + variation)
            except Exception as exc:
                logger.warning("Sensitivity analysis failed for %s: %s", name, exc)
                continue

            _process_sensitivity(name, base_dcr, low_dcr, high_dcr,
                                 findings, results)
    else:
        logger.warning("No analysis function or pre-computed results provided "
                       "for sensitivity sweep.")

    return findings, results


def _process_sensitivity(
    name: str,
    base_dcr: float,
    low_dcr: float,
    high_dcr: float,
    findings: list[Finding],
    results: list[SensitivityResult],
) -> None:
    """Process one sensitivity parameter and generate findings."""
    if base_dcr <= 0:
        delta_dcr = max(abs(high_dcr - base_dcr), abs(low_dcr - base_dcr))
        pct_change = 1.0  # flag as dominant if base is zero
    else:
        delta_dcr = max(abs(high_dcr - base_dcr), abs(low_dcr - base_dcr))
        pct_change = delta_dcr / base_dcr

    if pct_change > SENSITIVITY_DOMINANT_THRESHOLD:
        classification = "DOMINANT"
    elif pct_change > SENSITIVITY_MODERATE_THRESHOLD:
        classification = "MODERATE"
    else:
        classification = "INSENSITIVE"

    results.append(SensitivityResult(
        parameter=name,
        base_dcr=round(base_dcr, 4),
        low_dcr=round(low_dcr, 4),
        high_dcr=round(high_dcr, 4),
        delta_dcr=round(delta_dcr, 4),
        classification=classification,
    ))

    if classification == "DOMINANT":
        findings.append(Finding(
            severity="WARNING",
            vector="Sensitivity Sweep",
            element=None,
            location="Global",
            description=(
                f"{name} is a DOMINANT parameter — ±20% variation causes "
                f"{pct_change * 100:.1f}% change in DCR "
                f"(base={base_dcr:.3f}, low={low_dcr:.3f}, high={high_dcr:.3f})."
            ),
            dcr=max(low_dcr, high_dcr),
            controlling_combo="Sensitivity Analysis",
            recommendation=(
                f"Obtain accurate values for {name}. Design is highly "
                f"sensitive to this parameter — conservative assumptions "
                f"may significantly impact economy or safety."
            ),
        ))
    elif classification == "MODERATE":
        findings.append(Finding(
            severity="NOTE",
            vector="Sensitivity Sweep",
            element=None,
            location="Global",
            description=(
                f"{name} has MODERATE sensitivity — ±20% variation causes "
                f"{pct_change * 100:.1f}% change in DCR."
            ),
            dcr=max(low_dcr, high_dcr),
            controlling_combo="Sensitivity Analysis",
            recommendation=(
                f"Verify {name} assumptions. Moderate impact on design."
            ),
        ))


# ===================================================================
# VECTOR 5: EXTREME EVENT COMBINER
# ===================================================================

def extreme_event_combiner(
    adversarial_results: list[dict],
    standard_results: list[dict],
) -> list[Finding]:
    """Compare adversarial load combinations against standard AASHTO results.

    Args:
        adversarial_results: List of dicts from adversarial combos:
            - ``combo_name``: str
            - ``max_demand``: float — peak demand from this combo
            - ``demand_type``: str — "moment", "shear", etc.
            - ``element``: int
            - ``location``: str
        standard_results: List of dicts from standard AASHTO combos:
            - ``combo_name``: str
            - ``max_demand``: float — peak demand (envelope)
            - ``demand_type``: str

    Returns:
        List of Finding objects for adversarial exceedances.
    """
    findings: list[Finding] = []

    # Build standard envelope by demand type
    standard_envelope: dict[str, float] = {}
    standard_combo_names: dict[str, str] = {}
    for sr in standard_results:
        dtype = sr.get("demand_type", "unknown")
        demand = sr.get("max_demand", 0.0)
        if demand > standard_envelope.get(dtype, 0.0):
            standard_envelope[dtype] = demand
            standard_combo_names[dtype] = sr.get("combo_name", "unknown")

    for ar in adversarial_results:
        combo_name = ar.get("combo_name", "unknown")
        adv_demand = ar.get("max_demand", 0.0)
        dtype = ar.get("demand_type", "unknown")
        element = ar.get("element")
        location = ar.get("location", f"Element {element}")

        std_demand = standard_envelope.get(dtype, 0.0)
        std_combo = standard_combo_names.get(dtype, "unknown")

        if std_demand <= 0:
            continue

        exceedance = (adv_demand - std_demand) / std_demand

        if exceedance > 0.15:
            severity = "WARNING"
            desc = (
                f"Adversarial combo '{combo_name}' produces {exceedance * 100:.1f}% "
                f"higher {dtype} than standard '{std_combo}'. "
                f"Adversarial = {adv_demand:.1f}, Standard = {std_demand:.1f}."
            )
            recommendation = (
                f"Check design for adversarial combination '{combo_name}'. "
                f"This scenario exceeds standard AASHTO by >{exceedance * 100:.0f}%."
            )
        elif exceedance > 0.05:
            severity = "NOTE"
            desc = (
                f"Adversarial combo '{combo_name}' produces {exceedance * 100:.1f}% "
                f"higher {dtype} than standard '{std_combo}'."
            )
            recommendation = (
                f"Be aware of adversarial combination '{combo_name}'. "
                f"Modest exceedance over standard AASHTO."
            )
        else:
            continue  # Within standard envelope — no finding

        findings.append(Finding(
            severity=severity,
            vector="Extreme Event Combiner",
            element=element,
            location=location,
            description=desc,
            dcr=None,
            controlling_combo=combo_name,
            recommendation=recommendation,
        ))

    # Sort by exceedance (descending)
    findings.sort(key=lambda f: f.severity == "WARNING", reverse=True)
    return findings


# ===================================================================
# VECTOR 6: ROBUSTNESS CHECK (ALTERNATE LOAD PATH)
# ===================================================================

def robustness_check(
    components: list[dict],
    analyze_fn=None,
    robustness_results: list[dict] | None = None,
) -> list[Finding]:
    """Systematically remove one major component and check survival.

    Components to remove: girder lines, bearings, columns, foundations.
    Load level: DC + 0.5×LL per GSA/AASHTO progressive collapse criteria.

    Args:
        components: List of dicts describing removable components:
            - ``name``: str — e.g. "Girder G3", "Column C2"
            - ``type``: str — "girder", "bearing", "column", "foundation"
            - ``elements``: list[int] — element tags comprising this component
        analyze_fn: Callable(removed_elements: list[int]) -> dict.
                    Returns {"max_dcr": float, "stable": bool}.
        robustness_results: Pre-computed results list of dicts:
            - ``component``: str — component name
            - ``max_dcr``: float — max DCR after removal
            - ``stable``: bool — whether structure remains stable
            - ``type``: str — component type

    Returns:
        List of Finding objects.
    """
    findings: list[Finding] = []

    results_to_process = robustness_results or []

    if analyze_fn is not None and not robustness_results:
        for comp in components:
            name = comp.get("name", "Unknown")
            elements = comp.get("elements", [])
            comp_type = comp.get("type", "unknown")

            try:
                result = analyze_fn(elements)
                results_to_process.append({
                    "component": name,
                    "max_dcr": result.get("max_dcr", 0.0),
                    "stable": result.get("stable", True),
                    "type": comp_type,
                })
            except Exception as exc:
                logger.warning("Robustness analysis failed for %s: %s", name, exc)
                results_to_process.append({
                    "component": name,
                    "max_dcr": 999.0,
                    "stable": False,
                    "type": comp_type,
                })

    for rr in results_to_process:
        name = rr.get("component", "Unknown")
        max_dcr = rr.get("max_dcr", 0.0)
        stable = rr.get("stable", True)
        comp_type = rr.get("type", "unknown")

        if not stable:
            findings.append(Finding(
                severity="CRITICAL",
                vector="Robustness Check",
                element=None,
                location=name,
                description=(
                    f"Bridge COLLAPSES if {name} is lost — "
                    f"structure becomes globally unstable."
                ),
                dcr=max_dcr,
                controlling_combo="DC + 0.5×LL (Progressive Collapse)",
                recommendation=(
                    f"Provide alternate load path around {name}. "
                    f"Consider adding redundancy or protective measures "
                    f"(barriers, fenders) to prevent loss of this component."
                ),
            ))
        elif max_dcr > 1.0:
            findings.append(Finding(
                severity="WARNING",
                vector="Robustness Check",
                element=None,
                location=name,
                description=(
                    f"Bridge survives loss of {name} structurally but "
                    f"remaining elements are overstressed (max DCR = {max_dcr:.2f})."
                ),
                dcr=max_dcr,
                controlling_combo="DC + 0.5×LL (Progressive Collapse)",
                recommendation=(
                    f"Strengthen remaining members to provide full alternate "
                    f"load path capacity after loss of {name}."
                ),
            ))
        else:
            findings.append(Finding(
                severity="NOTE",
                vector="Robustness Check",
                element=None,
                location=name,
                description=(
                    f"Bridge survives loss of {name} "
                    f"(max DCR = {max_dcr:.2f} under DC + 0.5×LL)."
                ),
                dcr=max_dcr,
                controlling_combo="DC + 0.5×LL (Progressive Collapse)",
                recommendation=f"Adequate redundancy for loss of {name}.",
            ))

    return findings


# ===================================================================
# VECTOR 7: HISTORY MATCHER
# ===================================================================

def history_matcher(
    bridge_info: dict,
    failure_db: list[dict] | None = None,
    score_threshold: int = 5,
) -> tuple[list[Finding], list[HistoryMatch]]:
    """Match bridge characteristics against historical failure database.

    Scoring algorithm:
    - Bridge type match: +3
    - Material match: +2
    - Span range within ±20%: +2
    - Similar structural detail: +3 per matching detail

    Args:
        bridge_info: Dict describing the bridge:
            - ``type``: str — "steel_girder", "steel_truss", "concrete", etc.
            - ``material``: str — "steel", "concrete"
            - ``max_span_ft``: float — longest span
            - ``details``: list[str] — structural details like
              "fracture_critical", "pin_hanger", etc.
        failure_db: Failure database records. If None, loads from file.
        score_threshold: Minimum score to report a match. Default 5.

    Returns:
        Tuple of (findings, history_matches).
    """
    findings: list[Finding] = []
    matches: list[HistoryMatch] = []

    if failure_db is None:
        failure_db = load_failure_database()

    if not failure_db:
        return findings, matches

    bridge_type = bridge_info.get("type", "").lower()
    bridge_material = bridge_info.get("material", "").lower()
    bridge_span = bridge_info.get("max_span_ft", 0.0)
    bridge_details = [d.lower() for d in bridge_info.get("details", [])]

    for failure in failure_db:
        score = 0
        matching_factors: list[str] = []

        # Type match (+3)
        failure_type = failure.get("type", "").lower()
        if failure_type and bridge_type:
            # Partial matching: "steel_girder" matches "steel_girder"
            # Also: "steel" in type matches any steel type
            if failure_type == bridge_type:
                score += 3
                matching_factors.append(f"Type match: {failure_type}")
            elif bridge_type.split("_")[0] in failure_type or failure_type.split("_")[0] in bridge_type:
                score += 1
                matching_factors.append(f"Partial type match: {bridge_type} ~ {failure_type}")

        # Material match (+2)
        failure_material = failure.get("material", "").lower()
        if failure_material and bridge_material and failure_material == bridge_material:
            score += 2
            matching_factors.append(f"Material match: {failure_material}")

        # Span range ±20% (+2)
        failure_spans = failure.get("spans", [])
        if failure_spans and bridge_span > 0:
            max_failure_span = max(failure_spans)
            if abs(max_failure_span - bridge_span) / max(bridge_span, 1) <= 0.20:
                score += 2
                matching_factors.append(
                    f"Span range match: {bridge_span:.0f}' ~ {max_failure_span:.0f}'"
                )

        # Detail match (+3 per matching detail)
        failure_details = [d.lower() for d in failure.get("details", [])]
        for detail in bridge_details:
            if detail in failure_details:
                score += 3
                matching_factors.append(f"Detail match: {detail}")

        if score >= score_threshold:
            match = HistoryMatch(
                failure_name=failure.get("name", "Unknown"),
                year=failure.get("year", 0),
                score=score,
                lesson=failure.get("lesson", ""),
                matching_factors=matching_factors,
            )
            matches.append(match)

    # Sort by score descending
    matches.sort(key=lambda m: m.score, reverse=True)

    # Generate findings for top matches
    for match in matches:
        if match.score >= score_threshold + 3:
            severity = "WARNING"
        else:
            severity = "NOTE"

        findings.append(Finding(
            severity=severity,
            vector="History Matcher",
            element=None,
            location="Global",
            description=(
                f"Bridge shares characteristics with {match.failure_name} "
                f"({match.year}) — similarity score {match.score}. "
                f"Matching factors: {', '.join(match.matching_factors)}."
            ),
            dcr=None,
            controlling_combo="Historical Precedent",
            recommendation=match.lesson,
            precedent=match.failure_name,
        ))

    return findings, matches


# ===================================================================
# RISK RATING LOGIC
# ===================================================================

def compute_risk_rating(
    findings: list[Finding],
    cascade_chains: list[CascadeChain] | None = None,
    sensitivity_results: list[SensitivityResult] | None = None,
    adversarial_exceedance_max: float = 0.0,
) -> str:
    """Compute overall risk rating from findings and analysis results.

    Rating logic:
    - **RED:** Any CRITICAL finding OR any cascade causing collapse
    - **YELLOW:** Any WARNING finding OR dominant sensitivity OR
      adversarial exceedance > 15%
    - **GREEN:** All findings are NOTE or less

    Args:
        findings:                  All findings from all vectors.
        cascade_chains:            Cascade chains from vector 2.
        sensitivity_results:       Sensitivity results from vector 4.
        adversarial_exceedance_max: Max exceedance ratio from vector 5.

    Returns:
        ``'RED'``, ``'YELLOW'``, or ``'GREEN'``.
    """
    # Check for RED conditions
    has_critical = any(f.severity == "CRITICAL" for f in findings)
    has_collapse = False
    if cascade_chains:
        has_collapse = any(c.causes_collapse for c in cascade_chains)

    if has_critical or has_collapse:
        return "RED"

    # Check for YELLOW conditions
    has_warning = any(f.severity == "WARNING" for f in findings)
    has_dominant = False
    if sensitivity_results:
        has_dominant = any(s.classification == "DOMINANT" for s in sensitivity_results)

    if has_warning or has_dominant or adversarial_exceedance_max > 0.15:
        return "YELLOW"

    return "GREEN"


# ===================================================================
# EXECUTIVE SUMMARY GENERATOR
# ===================================================================

def generate_summary(
    findings: list[Finding],
    risk_rating: str,
    attack_vectors_run: list[str],
    total_combinations: int,
) -> str:
    """Generate a 1-paragraph executive summary of red-team results.

    Args:
        findings:           All findings.
        risk_rating:        Overall risk rating.
        attack_vectors_run: Which vectors were executed.
        total_combinations: Total load combinations checked.

    Returns:
        Executive summary paragraph.
    """
    n_critical = sum(1 for f in findings if f.severity == "CRITICAL")
    n_warning = sum(1 for f in findings if f.severity == "WARNING")
    n_note = sum(1 for f in findings if f.severity == "NOTE")

    if risk_rating == "RED":
        tone = (
            f"The red-team analysis identified {n_critical} CRITICAL "
            f"finding(s) that require immediate attention. "
        )
    elif risk_rating == "YELLOW":
        tone = (
            f"The red-team analysis identified {n_warning} WARNING "
            f"finding(s) that should be addressed before final design. "
        )
    else:
        tone = (
            "The red-team analysis found no critical or warning-level "
            "issues. The design appears robust. "
        )

    summary = (
        f"{tone}"
        f"A total of {len(findings)} findings were generated across "
        f"{len(attack_vectors_run)} attack vectors, evaluating "
        f"{total_combinations} load combinations. "
        f"Findings breakdown: {n_critical} critical, {n_warning} warnings, "
        f"{n_note} notes."
    )

    # Add top finding if any
    if findings:
        top = findings[0]
        summary += f" Top finding: {top.description}"

    return summary


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def run_red_team(
    element_results: list[dict],
    bridge_info: dict | None = None,
    stage_results: list[dict] | None = None,
    site_constraints: dict | None = None,
    sensitivity_base_dcr: float | None = None,
    sensitivity_parameter_results: list[dict] | None = None,
    adversarial_results: list[dict] | None = None,
    standard_results: list[dict] | None = None,
    components: list[dict] | None = None,
    robustness_results: list[dict] | None = None,
    analyze_fn=None,
    cascade_analyze_fn=None,
    robustness_analyze_fn=None,
    sensitivity_analyze_fn=None,
    load_model: dict | None = None,
    vectors: list[str] | None = None,
) -> RedTeamReport:
    """Run the complete red-team analysis.

    This is the canonical entry point. It runs all 7 attack vectors
    (or a subset if ``vectors`` is specified) and assembles the
    RedTeamReport.

    Args:
        element_results:               DCR results for all elements.
        bridge_info:                   Bridge characteristics for history matching.
        stage_results:                 Construction stage results.
        site_constraints:              Site constraint dict.
        sensitivity_base_dcr:          Baseline DCR for sensitivity sweep.
        sensitivity_parameter_results: Pre-computed sensitivity results.
        adversarial_results:           Results from adversarial load combos.
        standard_results:              Results from standard AASHTO combos.
        components:                    Components for robustness check.
        robustness_results:            Pre-computed robustness results.
        analyze_fn:                    General re-analysis function.
        cascade_analyze_fn:            Cascade-specific analysis function.
        robustness_analyze_fn:         Robustness-specific analysis function.
        sensitivity_analyze_fn:        Sensitivity-specific analysis function.
        load_model:                    Load model dict for statistics.
        vectors:                       List of vectors to run, or None for all.
                                       Valid: ["dcr", "cascade", "construction",
                                       "sensitivity", "extreme", "robustness",
                                       "history"]

    Returns:
        Complete :class:`RedTeamReport`.
    """
    start_time = time.time()

    all_vectors = ["dcr", "cascade", "construction", "sensitivity",
                   "extreme", "robustness", "history"]
    if vectors is None:
        vectors = all_vectors

    report = RedTeamReport()
    all_findings: list[Finding] = []
    cascade_chains: list[CascadeChain] = []
    sensitivity_results_list: list[SensitivityResult] = []
    history_matches_list: list[HistoryMatch] = []
    adversarial_exceedance_max = 0.0

    # --- Vector 1: DCR Scanner ---
    if "dcr" in vectors and element_results:
        dcr_findings = dcr_scanner(element_results)
        all_findings.extend(dcr_findings)
        report.attack_vectors_run.append("DCR Scanner")

    # --- Vector 2: Failure Cascade ---
    if "cascade" in vectors and element_results:
        cascade_findings, cascade_chains = failure_cascade(
            element_results,
            analyze_fn=cascade_analyze_fn or analyze_fn,
        )
        all_findings.extend(cascade_findings)
        report.cascade_chains = cascade_chains
        report.attack_vectors_run.append("Failure Cascade")

    # --- Vector 3: Construction Vulnerability ---
    if "construction" in vectors and stage_results:
        construction_findings = construction_vulnerability(
            stage_results, site_constraints
        )
        all_findings.extend(construction_findings)
        report.attack_vectors_run.append("Construction Vulnerability")

    # --- Vector 4: Sensitivity Sweep ---
    if "sensitivity" in vectors and sensitivity_base_dcr is not None:
        sens_findings, sensitivity_results_list = sensitivity_sweep(
            sensitivity_base_dcr,
            analyze_fn=sensitivity_analyze_fn,
            parameter_results=sensitivity_parameter_results,
        )
        all_findings.extend(sens_findings)
        report.sensitivity_results = sensitivity_results_list
        report.attack_vectors_run.append("Sensitivity Sweep")

    # --- Vector 5: Extreme Event Combiner ---
    if "extreme" in vectors and adversarial_results and standard_results:
        extreme_findings = extreme_event_combiner(
            adversarial_results, standard_results
        )
        all_findings.extend(extreme_findings)
        report.attack_vectors_run.append("Extreme Event Combiner")

        # Compute max exceedance for risk rating
        for ar in adversarial_results:
            dtype = ar.get("demand_type", "")
            adv_demand = ar.get("max_demand", 0.0)
            # Find standard envelope for this demand type
            for sr in standard_results:
                if sr.get("demand_type") == dtype:
                    std_demand = sr.get("max_demand", 0.0)
                    if std_demand > 0:
                        exc = (adv_demand - std_demand) / std_demand
                        adversarial_exceedance_max = max(adversarial_exceedance_max, exc)

    # --- Vector 6: Robustness Check ---
    if "robustness" in vectors and (components or robustness_results):
        robust_findings = robustness_check(
            components or [],
            analyze_fn=robustness_analyze_fn or analyze_fn,
            robustness_results=robustness_results,
        )
        all_findings.extend(robust_findings)
        report.robustness_results = robustness_results or []
        report.attack_vectors_run.append("Robustness Check")

    # --- Vector 7: History Matcher ---
    if "history" in vectors and bridge_info:
        history_findings, history_matches_list = history_matcher(bridge_info)
        all_findings.extend(history_findings)
        report.history_matches = history_matches_list
        report.attack_vectors_run.append("History Matcher")

    # --- Sort all findings ---
    all_findings.sort(
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), -(f.dcr or 0))
    )
    report.findings = all_findings

    # --- Risk rating ---
    report.risk_rating = compute_risk_rating(
        all_findings,
        cascade_chains=cascade_chains,
        sensitivity_results=sensitivity_results_list,
        adversarial_exceedance_max=adversarial_exceedance_max,
    )

    # --- Statistics ---
    if load_model:
        report.total_load_cases = len(load_model.get("cases", []))
        report.total_combinations = load_model.get("total_combinations", 0)
    else:
        # Estimate from inputs
        report.total_load_cases = len(element_results) if element_results else 0
        report.total_combinations = (
            len(standard_results or []) + len(adversarial_results or [])
        )

    # --- Summary ---
    report.summary = generate_summary(
        all_findings,
        report.risk_rating,
        report.attack_vectors_run,
        report.total_combinations,
    )

    report.analysis_time_sec = round(time.time() - start_time, 3)

    logger.info(
        "Red team complete: %s rating, %d findings (%d critical, %d warning, %d note), "
        "%d vectors run in %.2fs",
        report.risk_rating,
        len(all_findings),
        sum(1 for f in all_findings if f.severity == "CRITICAL"),
        sum(1 for f in all_findings if f.severity == "WARNING"),
        sum(1 for f in all_findings if f.severity == "NOTE"),
        len(report.attack_vectors_run),
        report.analysis_time_sec,
    )

    return report

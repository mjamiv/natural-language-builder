"""Report Generator — Three-Tier Bridge Red Team Reports.

Generates executive, technical, and raw data exports from red-team results.
Includes SVG visualization helpers for DCR heat maps, tornado diagrams,
cascade diagrams, and risk dashboards.

Report Tiers
------------
1. **Executive Summary** — 1-page Go/No-Go with top 3 findings
2. **Technical Report** — 11-section full narrative with SVGs
3. **Raw Data Export** — JSON for import into other tools

Units
-----
All values in kip, inch, second (KIS) unless otherwise noted in output.

References
----------
AASHTO LRFD Bridge Design Specifications, 9th Edition
FHWA Bridge Inspector's Reference Manual (2012)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from typing import Optional

from nlb.tools.red_team import (
    Finding,
    RedTeamReport,
    SensitivityResult,
    CascadeChain,
    HistoryMatch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rating emoji/text lookup
# ---------------------------------------------------------------------------
_RATING_DISPLAY = {
    "GREEN": "🟢 GREEN",
    "YELLOW": "🟡 YELLOW",
    "RED": "🔴 RED",
}

_SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "WARNING": "🟡",
    "NOTE": "🔵",
}

_RECOMMENDATION_MAP = {
    "GREEN": "**Go** — Design is robust. Proceed to final design.",
    "YELLOW": "**Conditional** — Address warnings before final design approval.",
    "RED": "**No-Go** — Critical issues must be resolved before proceeding.",
}


# ===================================================================
# EXECUTIVE REPORT
# ===================================================================

def _executive_report(
    red_team: RedTeamReport,
    model_info: dict,
    site: dict,
) -> str:
    """Generate 1-page executive summary report (markdown).

    Targets < 500 words. Go/No-Go recommendation with top 3 findings.
    """
    bridge_name = model_info.get("name", "Unnamed Bridge")
    rating_display = _RATING_DISPLAY.get(red_team.risk_rating, red_team.risk_rating)
    recommendation = _RECOMMENDATION_MAP.get(red_team.risk_rating, "Review required.")

    lines = []
    lines.append(f"# BRIDGE RED TEAM REPORT — {bridge_name}")
    lines.append("")
    lines.append(f"**Risk Rating: {rating_display}**")
    lines.append("")
    lines.append(red_team.summary)
    lines.append("")

    # Top 3 findings
    top_findings = red_team.findings[:3]
    if top_findings:
        lines.append("## Top Findings")
        lines.append("")
        for i, f in enumerate(top_findings, 1):
            emoji = _SEVERITY_EMOJI.get(f.severity, "")
            lines.append(
                f"{i}. {emoji} **{f.severity}** — {f.description}"
            )
        lines.append("")

    # Key metrics
    lines.append("## Analysis Summary")
    lines.append("")
    lines.append(f"- **Attack Vectors Run:** {len(red_team.attack_vectors_run)}")
    lines.append(f"- **Total Load Combinations:** {red_team.total_combinations}")
    lines.append(
        f"- **Findings:** "
        f"{sum(1 for f in red_team.findings if f.severity == 'CRITICAL')} critical, "
        f"{sum(1 for f in red_team.findings if f.severity == 'WARNING')} warnings, "
        f"{sum(1 for f in red_team.findings if f.severity == 'NOTE')} notes"
    )
    lines.append(f"- **Analysis Time:** {red_team.analysis_time_sec:.1f} seconds")
    lines.append("")

    # Recommendation
    lines.append(f"## Recommendation")
    lines.append("")
    lines.append(recommendation)
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# TECHNICAL REPORT
# ===================================================================

def _technical_report(
    red_team: RedTeamReport,
    model_info: dict,
    site: dict,
) -> str:
    """Generate full 11-section technical report (markdown)."""
    bridge_name = model_info.get("name", "Unnamed Bridge")
    rating_display = _RATING_DISPLAY.get(red_team.risk_rating, red_team.risk_rating)

    sections = []

    # Title
    sections.append(f"# Bridge Red Team Report — {bridge_name}")
    sections.append(f"\n**Risk Rating: {rating_display}**\n")

    # Section 1: Bridge Description
    sections.append("## 1. Bridge Description\n")
    sections.append(_section_bridge_description(model_info))

    # Section 2: Site Conditions
    sections.append("## 2. Site Conditions\n")
    sections.append(_section_site_conditions(site))

    # Section 3: Model Description
    sections.append("## 3. Model Description\n")
    sections.append(_section_model_description(model_info))

    # Section 4: Loading
    sections.append("## 4. Loading\n")
    sections.append(_section_loading(model_info, red_team))

    # Section 5: Analysis Results
    sections.append("## 5. Analysis Results\n")
    sections.append(_section_analysis_results(red_team))

    # Section 6: Red Team Findings
    sections.append("## 6. Red Team Findings\n")
    sections.append(_section_findings(red_team))

    # Section 7: Sensitivity Analysis
    sections.append("## 7. Sensitivity Analysis\n")
    sections.append(_section_sensitivity(red_team))

    # Section 8: Failure Cascade Analysis
    sections.append("## 8. Failure Cascade Analysis\n")
    sections.append(_section_cascade(red_team))

    # Section 9: Historical Precedent Matches
    sections.append("## 9. Historical Precedent Matches\n")
    sections.append(_section_history(red_team))

    # Section 10: Robustness Assessment
    sections.append("## 10. Robustness Assessment\n")
    sections.append(_section_robustness(red_team))

    # Section 11: Recommendations
    sections.append("## 11. Recommendations\n")
    sections.append(_section_recommendations(red_team))

    return "\n".join(sections)


def _section_bridge_description(model_info: dict) -> str:
    lines = []
    for key in ["type", "material", "spans", "girder_spacing", "deck_width",
                "num_girders", "num_spans", "max_span_ft"]:
        val = model_info.get(key)
        if val is not None:
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {val}")

    if not lines:
        lines.append("*Bridge description data not provided.*")
    return "\n".join(lines) + "\n"


def _section_site_conditions(site: dict) -> str:
    lines = []
    location = site.get("location", {})
    if location:
        parts = [location.get("city", ""), location.get("county", ""),
                 location.get("state", "")]
        loc_str = ", ".join(p for p in parts if p)
        if loc_str:
            lines.append(f"- **Location:** {loc_str}")

    coords = site.get("coordinates", {})
    if coords:
        lines.append(f"- **Coordinates:** {coords.get('lat', '')}, {coords.get('lon', '')}")

    seismic = site.get("seismic", {})
    if seismic:
        sdc = seismic.get("sdc", "N/A")
        sds = seismic.get("sds", "N/A")
        sd1 = seismic.get("sd1", "N/A")
        lines.append(f"- **Seismic Design Category:** {sdc} (SDS={sds}, SD1={sd1})")

    wind = site.get("wind", {})
    if wind:
        lines.append(f"- **Wind Speed:** {wind.get('v_ult', 'N/A')} mph, Exposure {wind.get('exposure', 'C')}")

    thermal = site.get("thermal", {})
    if thermal:
        lines.append(
            f"- **Thermal Range:** {thermal.get('t_min', 'N/A')}°F to {thermal.get('t_max', 'N/A')}°F "
            f"(ΔT = {thermal.get('delta_t', 'N/A')}°F)"
        )

    scour = site.get("scour", {})
    if scour and scour.get("water_crossing"):
        lines.append(f"- **Scour:** Water crossing — design flood Q100, check flood Q500")

    if not lines:
        lines.append("*Site condition data not provided.*")
    return "\n".join(lines) + "\n"


def _section_model_description(model_info: dict) -> str:
    lines = []
    for key in ["num_nodes", "num_elements", "num_dofs", "element_types"]:
        val = model_info.get(key)
        if val is not None:
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {val}")

    if not lines:
        lines.append("*Model description data not provided.*")
    return "\n".join(lines) + "\n"


def _section_loading(model_info: dict, red_team: RedTeamReport) -> str:
    lines = []
    lines.append(
        f"- **Total Load Combinations:** {red_team.total_combinations} "
        f"(standard + adversarial)"
    )
    lines.append(f"- **Total Load Cases:** {red_team.total_load_cases}")

    load_info = model_info.get("loading", {})
    if load_info:
        for key, val in load_info.items():
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {val}")

    return "\n".join(lines) + "\n"


def _section_analysis_results(red_team: RedTeamReport) -> str:
    lines = []
    if red_team.findings:
        # Find max DCR across all findings
        dcr_findings = [f for f in red_team.findings if f.dcr is not None]
        if dcr_findings:
            max_dcr_finding = max(dcr_findings, key=lambda f: f.dcr)
            lines.append(
                f"- **Maximum DCR:** {max_dcr_finding.dcr:.3f} at "
                f"{max_dcr_finding.location} ({max_dcr_finding.controlling_combo})"
            )

        n_over_1 = sum(1 for f in dcr_findings if (f.dcr or 0) > 1.0)
        n_over_085 = sum(1 for f in dcr_findings if 0.85 < (f.dcr or 0) <= 1.0)
        lines.append(f"- **Elements exceeding capacity (DCR > 1.0):** {n_over_1}")
        lines.append(f"- **Elements near capacity (DCR > 0.85):** {n_over_085}")
    else:
        lines.append("*No analysis results available.*")

    return "\n".join(lines) + "\n"


def _section_findings(red_team: RedTeamReport) -> str:
    if not red_team.findings:
        return "*No findings generated.*\n"

    # Group by vector
    by_vector: dict[str, list[Finding]] = {}
    for f in red_team.findings:
        by_vector.setdefault(f.vector, []).append(f)

    lines = []
    for vector, findings in by_vector.items():
        lines.append(f"### {vector}")
        lines.append("")
        for f in findings:
            emoji = _SEVERITY_EMOJI.get(f.severity, "")
            lines.append(f"- {emoji} **{f.severity}**")
            if f.element is not None:
                lines.append(f"  - Element: {f.element} ({f.location})")
            elif f.location:
                lines.append(f"  - Location: {f.location}")
            lines.append(f"  - {f.description}")
            if f.dcr is not None:
                lines.append(f"  - DCR: {f.dcr:.3f}")
            lines.append(f"  - Controlling: {f.controlling_combo}")
            lines.append(f"  - Recommendation: {f.recommendation}")
            if f.precedent:
                lines.append(f"  - Historical precedent: {f.precedent}")
            lines.append("")

    return "\n".join(lines)


def _section_sensitivity(red_team: RedTeamReport) -> str:
    if not red_team.sensitivity_results:
        return "*Sensitivity analysis not performed or no results available.*\n"

    lines = []

    # Summary table
    lines.append("| Parameter | Base DCR | Low (-20%) | High (+20%) | ΔDCR | Class |")
    lines.append("|-----------|----------|------------|-------------|------|-------|")
    for sr in red_team.sensitivity_results:
        lines.append(
            f"| {sr.parameter} | {sr.base_dcr:.3f} | {sr.low_dcr:.3f} | "
            f"{sr.high_dcr:.3f} | {sr.delta_dcr:.3f} | {sr.classification} |"
        )

    lines.append("")

    # Tornado diagram SVG
    if red_team.sensitivity_results:
        lines.append("### Tornado Diagram")
        lines.append("")
        svg = generate_tornado_svg(red_team.sensitivity_results)
        lines.append(svg)
        lines.append("")

    return "\n".join(lines)


def _section_cascade(red_team: RedTeamReport) -> str:
    if not red_team.cascade_chains:
        return "*No failure cascade chains detected.*\n"

    lines = []
    for i, chain in enumerate(red_team.cascade_chains, 1):
        collapse_flag = " ⚠️ **COLLAPSE**" if chain.causes_collapse else ""
        lines.append(f"### Chain {i}: Element {chain.trigger_element}{collapse_flag}")
        lines.append("")
        if chain.description:
            lines.append(f"{chain.description}")
        if chain.chain:
            lines.append("")
            lines.append("Cascade sequence:")
            for elem, dcr in chain.chain:
                lines.append(f"  → Element {elem} (DCR = {dcr:.2f})")
        lines.append("")

    # Cascade diagram SVG
    lines.append("### Cascade Diagram")
    lines.append("")
    svg = generate_cascade_svg(red_team.cascade_chains)
    lines.append(svg)

    return "\n".join(lines)


def _section_history(red_team: RedTeamReport) -> str:
    if not red_team.history_matches:
        return "*No historical precedent matches found.*\n"

    lines = []
    for match in red_team.history_matches:
        lines.append(f"### {match.failure_name} ({match.year})")
        lines.append(f"- **Similarity Score:** {match.score}")
        lines.append(f"- **Matching Factors:** {', '.join(match.matching_factors)}")
        lines.append(f"- **Key Lesson:** {match.lesson}")
        lines.append("")

    return "\n".join(lines)


def _section_robustness(red_team: RedTeamReport) -> str:
    robustness_findings = [
        f for f in red_team.findings if f.vector == "Robustness Check"
    ]
    if not robustness_findings:
        return "*Robustness assessment not performed.*\n"

    lines = []
    for f in robustness_findings:
        emoji = _SEVERITY_EMOJI.get(f.severity, "")
        lines.append(f"- {emoji} **{f.location}:** {f.description}")
    lines.append("")

    return "\n".join(lines)


def _section_recommendations(red_team: RedTeamReport) -> str:
    lines = []

    # Collect unique recommendations
    seen = set()
    critical_recs = []
    warning_recs = []
    note_recs = []

    for f in red_team.findings:
        if f.recommendation not in seen:
            seen.add(f.recommendation)
            if f.severity == "CRITICAL":
                critical_recs.append(f.recommendation)
            elif f.severity == "WARNING":
                warning_recs.append(f.recommendation)
            else:
                note_recs.append(f.recommendation)

    if critical_recs:
        lines.append("### Critical (Immediate Action Required)")
        for i, rec in enumerate(critical_recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    if warning_recs:
        lines.append("### Warnings (Address Before Final Design)")
        for i, rec in enumerate(warning_recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    if note_recs:
        lines.append("### Notes (For Awareness)")
        for i, rec in enumerate(note_recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    if not (critical_recs or warning_recs or note_recs):
        lines.append("No specific recommendations. Design appears robust.")

    recommendation = _RECOMMENDATION_MAP.get(red_team.risk_rating, "Review required.")
    lines.append(f"\n**Overall Recommendation:** {recommendation}")

    return "\n".join(lines) + "\n"


# ===================================================================
# RAW DATA EXPORT
# ===================================================================

def _raw_export(
    red_team: RedTeamReport,
    model_info: dict,
    site: dict,
) -> str:
    """Export all data as JSON."""
    data = {
        "report_type": "raw",
        "risk_rating": red_team.risk_rating,
        "summary": red_team.summary,
        "model_info": model_info,
        "site": site,
        "findings": [asdict(f) for f in red_team.findings],
        "cascade_chains": [asdict(c) for c in red_team.cascade_chains],
        "sensitivity_results": [asdict(s) for s in red_team.sensitivity_results],
        "history_matches": [asdict(m) for m in red_team.history_matches],
        "robustness_results": red_team.robustness_results,
        "attack_vectors_run": red_team.attack_vectors_run,
        "total_load_cases": red_team.total_load_cases,
        "total_combinations": red_team.total_combinations,
        "analysis_time_sec": red_team.analysis_time_sec,
    }
    return json.dumps(data, indent=2)


# ===================================================================
# SVG VISUALIZATION HELPERS
# ===================================================================

def generate_tornado_svg(
    sensitivity_results: list[SensitivityResult],
    width: int = 600,
    bar_height: int = 28,
    margin: int = 40,
) -> str:
    """Generate SVG tornado diagram from sensitivity results.

    Bars extend left (low) and right (high) from the base DCR centerline.

    Args:
        sensitivity_results: List of SensitivityResult objects.
        width:               SVG width in pixels.
        bar_height:          Height of each bar.
        margin:              Margin around the diagram.

    Returns:
        SVG string.
    """
    if not sensitivity_results:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="30"><text x="10" y="20">No data</text></svg>'

    n = len(sensitivity_results)
    # Sort by delta_dcr descending for tornado effect
    sorted_results = sorted(sensitivity_results, key=lambda s: s.delta_dcr, reverse=True)

    label_width = 180
    chart_width = width - label_width - margin * 2
    height = margin * 2 + n * (bar_height + 6) + 30

    # Find DCR range
    all_dcrs = []
    for sr in sorted_results:
        all_dcrs.extend([sr.low_dcr, sr.high_dcr, sr.base_dcr])
    min_dcr = min(all_dcrs) if all_dcrs else 0
    max_dcr = max(all_dcrs) if all_dcrs else 1
    dcr_range = max_dcr - min_dcr
    if dcr_range < 0.001:
        dcr_range = 0.1

    def dcr_to_x(dcr: float) -> float:
        return label_width + margin + (dcr - min_dcr) / dcr_range * chart_width

    # Color map
    color_map = {
        "DOMINANT": "#e74c3c",
        "MODERATE": "#f39c12",
        "INSENSITIVE": "#27ae60",
    }

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'class="tornado-diagram">'
    )
    lines.append(
        '<style>'
        '.label { font: 12px sans-serif; fill: #333; }'
        '.axis-label { font: 10px sans-serif; fill: #666; }'
        '.bar { opacity: 0.85; }'
        '.base-line { stroke: #333; stroke-width: 1.5; stroke-dasharray: 4,4; }'
        '</style>'
    )

    # Base DCR line
    if sorted_results:
        base_x = dcr_to_x(sorted_results[0].base_dcr)
        lines.append(
            f'<line x1="{base_x:.1f}" y1="{margin}" '
            f'x2="{base_x:.1f}" y2="{height - margin}" class="base-line"/>'
        )
        lines.append(
            f'<text x="{base_x:.1f}" y="{margin - 5}" '
            f'text-anchor="middle" class="axis-label">'
            f'Base DCR = {sorted_results[0].base_dcr:.3f}</text>'
        )

    # Bars
    for i, sr in enumerate(sorted_results):
        y = margin + 20 + i * (bar_height + 6)
        color = color_map.get(sr.classification, "#95a5a6")

        x_low = dcr_to_x(sr.low_dcr)
        x_high = dcr_to_x(sr.high_dcr)
        x_base = dcr_to_x(sr.base_dcr)

        # Bar from low to high
        bar_x = min(x_low, x_high)
        bar_w = abs(x_high - x_low)

        lines.append(
            f'<rect x="{bar_x:.1f}" y="{y:.1f}" '
            f'width="{max(bar_w, 1):.1f}" height="{bar_height}" '
            f'fill="{color}" class="bar" rx="3"/>'
        )

        # Label
        lines.append(
            f'<text x="{label_width + margin - 8}" y="{y + bar_height / 2 + 4:.1f}" '
            f'text-anchor="end" class="label">'
            f'{sr.parameter} ({sr.classification[0]})</text>'
        )

        # DCR values at ends
        lines.append(
            f'<text x="{x_low - 3:.1f}" y="{y + bar_height / 2 + 3:.1f}" '
            f'text-anchor="end" class="axis-label">{sr.low_dcr:.3f}</text>'
        )
        lines.append(
            f'<text x="{x_high + 3:.1f}" y="{y + bar_height / 2 + 3:.1f}" '
            f'text-anchor="start" class="axis-label">{sr.high_dcr:.3f}</text>'
        )

    lines.append('</svg>')
    return "\n".join(lines)


def generate_dcr_heatmap_svg(
    elements: list[dict],
    width: int = 700,
    height: int = 200,
) -> str:
    """Generate SVG DCR heat map — bridge elevation colored by DCR.

    Args:
        elements: List of dicts with:
            - ``element``: int
            - ``x_start``: float — start position along bridge
            - ``x_end``: float — end position
            - ``dcr``: float
            - ``location``: str
        width:  SVG width.
        height: SVG height.

    Returns:
        SVG string.
    """
    if not elements:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="30"><text x="10" y="20">No data</text></svg>'

    margin = 40
    bridge_y = height / 2
    elem_height = 30

    # Find span range
    all_x = []
    for e in elements:
        all_x.extend([e.get("x_start", 0), e.get("x_end", 0)])
    x_min = min(all_x) if all_x else 0
    x_max = max(all_x) if all_x else 100
    x_range = x_max - x_min
    if x_range < 0.001:
        x_range = 100

    def x_to_svg(x: float) -> float:
        return margin + (x - x_min) / x_range * (width - 2 * margin)

    def dcr_to_color(dcr: float) -> str:
        """Map DCR to green→yellow→red color."""
        if dcr <= 0.70:
            return "#27ae60"  # Green
        elif dcr <= 0.85:
            # Green to yellow
            t = (dcr - 0.70) / 0.15
            r = int(39 + t * (241 - 39))
            g = int(174 + t * (196 - 174))
            b = int(96 + t * (15 - 96))
            return f"#{r:02x}{g:02x}{b:02x}"
        elif dcr <= 1.0:
            # Yellow to red
            t = (dcr - 0.85) / 0.15
            r = int(241 + t * (231 - 241))
            g = int(196 - t * 196)
            b = int(15 - t * 15)
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            return "#e74c3c"  # Red

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" class="dcr-heatmap">'
    )
    lines.append(
        '<style>'
        '.elem-label { font: 9px sans-serif; fill: #333; }'
        '.legend-label { font: 10px sans-serif; fill: #666; }'
        '</style>'
    )

    for e in elements:
        x1 = x_to_svg(e.get("x_start", 0))
        x2 = x_to_svg(e.get("x_end", 0))
        dcr = e.get("dcr", 0.0)
        color = dcr_to_color(dcr)

        lines.append(
            f'<rect x="{x1:.1f}" y="{bridge_y - elem_height / 2:.1f}" '
            f'width="{max(x2 - x1, 2):.1f}" height="{elem_height}" '
            f'fill="{color}" stroke="#555" stroke-width="0.5"/>'
        )
        # DCR label
        mid_x = (x1 + x2) / 2
        lines.append(
            f'<text x="{mid_x:.1f}" y="{bridge_y + 4:.1f}" '
            f'text-anchor="middle" class="elem-label">{dcr:.2f}</text>'
        )

    # Legend
    legend_y = height - 20
    legend_items = [("≤0.70", "#27ae60"), ("0.85", "#f1c40f"),
                    ("1.00", "#e74c3c"), (">1.00", "#c0392b")]
    lx = margin
    for label, color in legend_items:
        lines.append(
            f'<rect x="{lx}" y="{legend_y}" width="12" height="12" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{lx + 16}" y="{legend_y + 10}" class="legend-label">{label}</text>'
        )
        lx += 70

    lines.append('</svg>')
    return "\n".join(lines)


def generate_cascade_svg(
    chains: list[CascadeChain],
    width: int = 600,
) -> str:
    """Generate SVG cascade/flowchart diagram.

    Shows trigger elements and their cascade chains as connected boxes.

    Args:
        chains: List of CascadeChain objects.
        width:  SVG width.

    Returns:
        SVG string.
    """
    if not chains:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="30"><text x="10" y="20">No cascades</text></svg>'

    box_w = 120
    box_h = 40
    gap_x = 40
    gap_y = 60
    margin = 20

    # Calculate height
    max_chain_len = max(len(c.chain) for c in chains) if chains else 0
    n_rows = len(chains)
    height = margin * 2 + n_rows * (box_h + gap_y)

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" class="cascade-diagram">'
    )
    lines.append(
        '<style>'
        '.trigger { fill: #e74c3c; stroke: #c0392b; stroke-width: 2; rx: 6; }'
        '.cascade { fill: #f39c12; stroke: #e67e22; stroke-width: 1.5; rx: 6; }'
        '.collapse { fill: #c0392b; stroke: #922b21; stroke-width: 2; rx: 6; }'
        '.box-text { font: bold 11px sans-serif; fill: white; }'
        '.arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }'
        '</style>'
    )
    # Arrow marker
    lines.append(
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" '
        'refX="8" refY="3" orient="auto">'
        '<polygon points="0 0, 8 3, 0 6" fill="#333"/>'
        '</marker></defs>'
    )

    for row, chain in enumerate(chains):
        y = margin + row * (box_h + gap_y)
        x = margin

        # Trigger box
        css_class = "collapse" if chain.causes_collapse else "trigger"
        lines.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" class="{css_class}"/>'
        )
        lines.append(
            f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 4}" '
            f'text-anchor="middle" class="box-text">E{chain.trigger_element}</text>'
        )

        # Chain elements
        prev_x = x + box_w
        for elem, dcr in chain.chain:
            x = prev_x + gap_x
            if x + box_w > width - margin:
                break

            # Arrow
            lines.append(
                f'<line x1="{prev_x}" y1="{y + box_h / 2}" '
                f'x2="{x}" y2="{y + box_h / 2}" class="arrow"/>'
            )

            lines.append(
                f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" class="cascade"/>'
            )
            lines.append(
                f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 4}" '
                f'text-anchor="middle" class="box-text">'
                f'E{elem} ({dcr:.2f})</text>'
            )
            prev_x = x + box_w

    lines.append('</svg>')
    return "\n".join(lines)


def generate_risk_dashboard_svg(
    red_team: RedTeamReport,
    width: int = 500,
    height: int = 250,
) -> str:
    """Generate SVG risk dashboard summary card.

    Shows rating, finding counts, and key metrics in a compact card.

    Args:
        red_team: RedTeamReport object.
        width:    SVG width.
        height:   SVG height.

    Returns:
        SVG string.
    """
    rating_colors = {
        "GREEN": "#27ae60",
        "YELLOW": "#f1c40f",
        "RED": "#e74c3c",
    }
    bg_color = rating_colors.get(red_team.risk_rating, "#95a5a6")

    n_critical = sum(1 for f in red_team.findings if f.severity == "CRITICAL")
    n_warning = sum(1 for f in red_team.findings if f.severity == "WARNING")
    n_note = sum(1 for f in red_team.findings if f.severity == "NOTE")

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" class="risk-dashboard">'
    )
    lines.append(
        '<style>'
        '.card-bg { rx: 12; }'
        '.rating { font: bold 36px sans-serif; fill: white; }'
        '.metric-label { font: 12px sans-serif; fill: rgba(255,255,255,0.8); }'
        '.metric-value { font: bold 20px sans-serif; fill: white; }'
        '.title { font: bold 14px sans-serif; fill: white; }'
        '</style>'
    )

    # Background
    lines.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="{bg_color}" class="card-bg"/>'
    )

    # Title
    lines.append(
        f'<text x="20" y="30" class="title">RED TEAM RISK DASHBOARD</text>'
    )

    # Rating
    lines.append(
        f'<text x="20" y="80" class="rating">{red_team.risk_rating}</text>'
    )

    # Metrics
    metrics = [
        ("Critical", str(n_critical), 20),
        ("Warnings", str(n_warning), 130),
        ("Notes", str(n_note), 240),
        ("Vectors", str(len(red_team.attack_vectors_run)), 350),
    ]
    for label, value, x in metrics:
        lines.append(f'<text x="{x}" y="130" class="metric-value">{value}</text>')
        lines.append(f'<text x="{x}" y="148" class="metric-label">{label}</text>')

    # Bottom line: combos and time
    lines.append(
        f'<text x="20" y="190" class="metric-label">'
        f'{red_team.total_combinations} load combinations analyzed '
        f'in {red_team.analysis_time_sec:.1f}s</text>'
    )

    # Vectors run
    vectors_str = ", ".join(red_team.attack_vectors_run[:4])
    if len(red_team.attack_vectors_run) > 4:
        vectors_str += f" +{len(red_team.attack_vectors_run) - 4} more"
    lines.append(
        f'<text x="20" y="210" class="metric-label">'
        f'Vectors: {vectors_str}</text>'
    )

    lines.append('</svg>')
    return "\n".join(lines)


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def generate_report(
    red_team: RedTeamReport,
    model_info: dict,
    site: dict,
    tier: str = "technical",
) -> str:
    """Generate a bridge red team report at the specified tier.

    Args:
        red_team:   :class:`RedTeamReport` from :func:`~nlb.tools.red_team.run_red_team`.
        model_info: Dict describing the bridge model:
            - ``name``: str — bridge name
            - ``type``: str — bridge type
            - ``material``: str
            - ``spans``: list
            - ``num_nodes``, ``num_elements``, ``num_dofs``: int
            - ``loading``: dict (optional)
        site:       Site profile dict from :meth:`SiteProfile.to_dict`.
        tier:       ``'executive'``, ``'technical'``, or ``'raw'``.

    Returns:
        Markdown string (executive/technical) or JSON string (raw).

    Raises:
        ValueError: If tier is not recognized.
    """
    tier = tier.lower().strip()

    if tier == "executive":
        return _executive_report(red_team, model_info, site)
    elif tier == "technical":
        return _technical_report(red_team, model_info, site)
    elif tier == "raw":
        return _raw_export(red_team, model_info, site)
    else:
        raise ValueError(
            f"Unknown report tier '{tier}'. "
            f"Valid options: 'executive', 'technical', 'raw'."
        )

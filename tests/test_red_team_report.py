"""Tests for Red Team Engine and Report Generator.

Tests cover:
- DCR scanner: CRITICAL/WARNING/NOTE classification
- Findings sorting: severity then DCR descending
- Risk rating logic: RED/YELLOW/GREEN thresholds
- History matcher: known failure similarity scoring
- Sensitivity classification: dominant/moderate/insensitive
- Report generation: valid markdown for each tier
- Executive summary: under 500 words
- Tornado diagram SVG: correct structure
- Failure database: loads and queries correctly
- Empty results: produce GREEN rating
"""

import json
import math
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Ensure project is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nlb.tools.red_team import (
    Finding,
    RedTeamReport,
    CascadeChain,
    SensitivityResult,
    HistoryMatch,
    dcr_scanner,
    failure_cascade,
    construction_vulnerability,
    sensitivity_sweep,
    extreme_event_combiner,
    robustness_check,
    history_matcher,
    compute_risk_rating,
    generate_summary,
    run_red_team,
    load_failure_database,
    _SEVERITY_ORDER,
    SENSITIVITY_DOMINANT_THRESHOLD,
    SENSITIVITY_MODERATE_THRESHOLD,
)
from nlb.tools.report import (
    generate_report,
    generate_tornado_svg,
    generate_dcr_heatmap_svg,
    generate_cascade_svg,
    generate_risk_dashboard_svg,
)


# ==================================================================
# FIXTURES
# ==================================================================

@pytest.fixture
def sample_element_results():
    """Element results spanning all severity levels."""
    return [
        {
            "element": 1,
            "location": "Span 1, 0.5L",
            "dcr": 1.15,
            "force_type": "moment",
            "controlling_combo": "Strength_I_max",
            "demand": 2300.0,
            "capacity": 2000.0,
        },
        {
            "element": 2,
            "location": "Span 1, 0.3L",
            "dcr": 0.92,
            "force_type": "shear",
            "controlling_combo": "Strength_I_max",
            "demand": 460.0,
            "capacity": 500.0,
        },
        {
            "element": 3,
            "location": "Span 2, 0.4L",
            "dcr": 0.75,
            "force_type": "moment",
            "controlling_combo": "Strength_V_max",
            "demand": 1500.0,
            "capacity": 2000.0,
        },
        {
            "element": 4,
            "location": "Pier 1",
            "dcr": 0.55,
            "force_type": "axial",
            "controlling_combo": "Strength_I_max",
        },
        {
            "element": 5,
            "location": "Span 2, 0.6L",
            "dcr": 1.05,
            "force_type": "moment",
            "controlling_combo": "Extreme_Event_I_max",
        },
    ]


@pytest.fixture
def sample_bridge_info():
    """Bridge info for history matching."""
    return {
        "type": "steel_girder",
        "material": "steel",
        "max_span_ft": 100.0,
        "details": ["fracture_critical", "pin_hanger"],
    }


@pytest.fixture
def sample_model_info():
    return {
        "name": "Test Bridge #1",
        "type": "steel_girder",
        "material": "steel",
        "spans": [80, 100, 80],
        "num_nodes": 120,
        "num_elements": 95,
        "num_dofs": 360,
        "max_span_ft": 100.0,
    }


@pytest.fixture
def sample_site():
    return {
        "coordinates": {"lat": 42.28, "lon": -89.09},
        "location": {"state": "IL", "county": "DeKalb", "city": "DeKalb"},
        "seismic": {"sds": 0.267, "sd1": 0.160, "sdc": "B"},
        "wind": {"v_ult": 115, "exposure": "C"},
        "thermal": {"t_min": -10, "t_max": 110, "delta_t": 120},
        "scour": {"water_crossing": True},
    }


@pytest.fixture
def sensitivity_results():
    return [
        SensitivityResult("Soil Stiffness", 0.80, 0.68, 0.94, 0.14, "DOMINANT"),
        SensitivityResult("Concrete Strength", 0.80, 0.76, 0.84, 0.04, "INSENSITIVE"),
        SensitivityResult("Live Load", 0.80, 0.73, 0.88, 0.08, "MODERATE"),
        SensitivityResult("Scour Depth", 0.80, 0.72, 0.95, 0.15, "DOMINANT"),
        SensitivityResult("Bearing Friction", 0.80, 0.78, 0.82, 0.02, "INSENSITIVE"),
    ]


@pytest.fixture
def cascade_chains():
    return [
        CascadeChain(
            trigger_element=1,
            chain=[(2, 1.35), (3, 1.12)],
            causes_collapse=False,
            description="Element 1 fails → Element 2 overloads (1.35) → Element 3 overloads (1.12)",
        ),
        CascadeChain(
            trigger_element=10,
            chain=[(11, 1.50), (12, 1.80), (13, 2.10)],
            causes_collapse=True,
            description="Element 10 fails → chain collapse",
        ),
    ]


# ==================================================================
# VECTOR 1: DCR SCANNER
# ==================================================================

class TestDCRScanner:

    def test_finds_critical(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        critical = [f for f in findings if f.severity == "CRITICAL"]
        assert len(critical) == 2  # elements 1 and 5

    def test_finds_warning(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        warnings = [f for f in findings if f.severity == "WARNING"]
        assert len(warnings) == 1  # element 2 (DCR 0.92)

    def test_finds_note(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        notes = [f for f in findings if f.severity == "NOTE"]
        assert len(notes) == 1  # element 3 (DCR 0.75)

    def test_ignores_below_threshold(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        # Element 4 (DCR 0.55) should not appear
        elem_ids = [f.element for f in findings]
        assert 4 not in elem_ids

    def test_sorted_by_severity_then_dcr(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        # First should be CRITICAL with highest DCR
        assert findings[0].severity == "CRITICAL"
        assert findings[0].dcr == 1.15

        # All CRITICALs before WARNINGs before NOTEs
        severities = [f.severity for f in findings]
        expected_order = sorted(severities, key=lambda s: _SEVERITY_ORDER[s])
        assert severities == expected_order

    def test_includes_element_and_location(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        for f in findings:
            assert f.element is not None
            assert f.location
            assert f.controlling_combo

    def test_empty_results(self):
        findings = dcr_scanner([])
        assert findings == []

    def test_all_below_threshold(self):
        results = [
            {"element": 1, "dcr": 0.50, "force_type": "moment",
             "controlling_combo": "test", "location": "test"},
        ]
        findings = dcr_scanner(results)
        assert findings == []

    def test_none_dcr_skipped(self):
        results = [
            {"element": 1, "dcr": None, "force_type": "moment",
             "controlling_combo": "test", "location": "test"},
        ]
        findings = dcr_scanner(results)
        assert findings == []

    def test_demand_capacity_in_description(self, sample_element_results):
        findings = dcr_scanner(sample_element_results)
        critical = [f for f in findings if f.element == 1][0]
        assert "2300" in critical.description
        assert "2000" in critical.description


# ==================================================================
# VECTOR 2: FAILURE CASCADE
# ==================================================================

class TestFailureCascade:

    def test_finds_cascade_without_analyze_fn(self, sample_element_results):
        findings, chains = failure_cascade(sample_element_results)
        assert len(findings) > 0
        assert len(chains) > 0

    def test_no_cascade_when_no_failures(self):
        results = [
            {"element": 1, "dcr": 0.50, "location": "Span 1",
             "force_type": "moment", "controlling_combo": "test"},
        ]
        findings, chains = failure_cascade(results)
        assert findings == []
        assert chains == []

    def test_cascade_with_analyze_fn(self):
        results = [
            {"element": 1, "dcr": 1.2, "location": "Span 1",
             "force_type": "moment", "controlling_combo": "test"},
        ]

        def mock_analyze(removed):
            return [
                {"element": 2, "dcr": 1.4, "location": "Span 2",
                 "force_type": "moment", "controlling_combo": "test"},
            ]

        findings, chains = failure_cascade(results, analyze_fn=mock_analyze)
        assert len(chains) == 1
        assert len(chains[0].chain) == 1
        assert chains[0].chain[0] == (2, 1.4)

    def test_cascade_instability(self):
        results = [
            {"element": 1, "dcr": 1.5, "location": "Pier 1",
             "force_type": "axial", "controlling_combo": "test"},
        ]

        def mock_analyze_fail(removed):
            raise RuntimeError("Analysis became unstable")

        findings, chains = failure_cascade(results, analyze_fn=mock_analyze_fail)
        assert chains[0].causes_collapse is True


# ==================================================================
# VECTOR 3: CONSTRUCTION VULNERABILITY
# ==================================================================

class TestConstructionVulnerability:

    def test_finds_critical_stage(self):
        stages = [
            {"stage_name": "Erect Girders", "stage_number": 1,
             "max_dcr": 0.60, "max_dcr_element": 10,
             "max_dcr_location": "Girder G1"},
            {"stage_name": "Pour Deck Span 2", "stage_number": 4,
             "max_dcr": 1.05, "max_dcr_element": 42,
             "max_dcr_location": "Span 2, 0.5L"},
        ]
        findings = construction_vulnerability(stages)
        assert any(f.severity == "CRITICAL" for f in findings)
        assert any("Stage 4" in f.description for f in findings)

    def test_flags_temp_support(self):
        stages = [
            {"stage_name": "Pour Deck", "stage_number": 2,
             "max_dcr": 0.70, "max_dcr_element": 5,
             "max_dcr_location": "Span 1",
             "requires_temp_support": True},
        ]
        findings = construction_vulnerability(stages)
        assert any("temporary supports" in f.description for f in findings)

    def test_empty_stages(self):
        findings = construction_vulnerability([])
        assert findings == []

    def test_site_constraint_conflict(self):
        stages = [
            {"stage_name": "Pour pier cap", "stage_number": 1,
             "max_dcr": 0.50, "max_dcr_element": 1,
             "max_dcr_location": "Pier 1",
             "requires_temp_support": True},
        ]
        constraints = {"no_equipment_in_water": True}
        findings = construction_vulnerability(stages, constraints)
        assert any("CONFLICT" in f.description for f in findings)


# ==================================================================
# VECTOR 4: SENSITIVITY SWEEP
# ==================================================================

class TestSensitivitySweep:

    def test_classifies_dominant(self):
        params = [
            {"parameter": "Soil Stiffness", "low_dcr": 0.65, "high_dcr": 0.95},
        ]
        findings, results = sensitivity_sweep(0.80, parameter_results=params)
        assert results[0].classification == "DOMINANT"
        assert any(f.severity == "WARNING" for f in findings)

    def test_classifies_moderate(self):
        params = [
            {"parameter": "Concrete Strength", "low_dcr": 0.75, "high_dcr": 0.86},
        ]
        findings, results = sensitivity_sweep(0.80, parameter_results=params)
        assert results[0].classification == "MODERATE"

    def test_classifies_insensitive(self):
        params = [
            {"parameter": "Bearing Friction", "low_dcr": 0.79, "high_dcr": 0.82},
        ]
        findings, results = sensitivity_sweep(0.80, parameter_results=params)
        assert results[0].classification == "INSENSITIVE"

    def test_dominant_threshold(self):
        # 10% change → DOMINANT
        base = 1.0
        params = [
            {"parameter": "Test", "low_dcr": 0.89, "high_dcr": 1.11},
        ]
        _, results = sensitivity_sweep(base, parameter_results=params)
        assert results[0].classification == "DOMINANT"

    def test_moderate_threshold(self):
        # 5-10% → MODERATE
        base = 1.0
        params = [
            {"parameter": "Test", "low_dcr": 0.93, "high_dcr": 1.07},
        ]
        _, results = sensitivity_sweep(base, parameter_results=params)
        assert results[0].classification == "MODERATE"

    def test_with_analyze_fn(self):
        def mock_analyze(key, factor):
            return 0.80 * factor  # Linear sensitivity

        findings, results = sensitivity_sweep(0.80, analyze_fn=mock_analyze)
        assert len(results) > 0
        # Should have at least some results for each default parameter

    def test_empty_without_inputs(self):
        findings, results = sensitivity_sweep(0.80)
        assert findings == []
        assert results == []


# ==================================================================
# VECTOR 5: EXTREME EVENT COMBINER
# ==================================================================

class TestExtremeEventCombiner:

    def test_flags_exceedance(self):
        adversarial = [
            {"combo_name": "Scour+Seismic", "max_demand": 1230,
             "demand_type": "moment", "element": 5, "location": "Pier 1"},
        ]
        standard = [
            {"combo_name": "Extreme_Event_I", "max_demand": 1000,
             "demand_type": "moment"},
        ]
        findings = extreme_event_combiner(adversarial, standard)
        assert len(findings) == 1
        assert findings[0].severity == "WARNING"
        assert "23.0%" in findings[0].description

    def test_ignores_small_exceedance(self):
        adversarial = [
            {"combo_name": "Test", "max_demand": 1030,
             "demand_type": "shear", "element": 1, "location": "test"},
        ]
        standard = [
            {"combo_name": "Strength_I", "max_demand": 1000,
             "demand_type": "shear"},
        ]
        findings = extreme_event_combiner(adversarial, standard)
        assert findings == []  # 3% < 5% threshold

    def test_note_for_moderate_exceedance(self):
        adversarial = [
            {"combo_name": "Test", "max_demand": 1100,
             "demand_type": "moment", "element": 1, "location": "test"},
        ]
        standard = [
            {"combo_name": "Strength_I", "max_demand": 1000,
             "demand_type": "moment"},
        ]
        findings = extreme_event_combiner(adversarial, standard)
        assert len(findings) == 1
        assert findings[0].severity == "NOTE"


# ==================================================================
# VECTOR 6: ROBUSTNESS CHECK
# ==================================================================

class TestRobustnessCheck:

    def test_flags_collapse(self):
        results = [
            {"component": "Column C2", "max_dcr": 999.0,
             "stable": False, "type": "column"},
        ]
        findings = robustness_check([], robustness_results=results)
        assert findings[0].severity == "CRITICAL"
        assert "COLLAPSES" in findings[0].description

    def test_flags_overstress(self):
        results = [
            {"component": "Girder G3", "max_dcr": 1.15,
             "stable": True, "type": "girder"},
        ]
        findings = robustness_check([], robustness_results=results)
        assert findings[0].severity == "WARNING"

    def test_passes_healthy(self):
        results = [
            {"component": "Bearing B1", "max_dcr": 0.87,
             "stable": True, "type": "bearing"},
        ]
        findings = robustness_check([], robustness_results=results)
        assert findings[0].severity == "NOTE"
        assert "survives" in findings[0].description

    def test_with_analyze_fn(self):
        components = [
            {"name": "Girder G1", "type": "girder", "elements": [1, 2, 3]},
        ]

        def mock_analyze(removed):
            return {"max_dcr": 0.75, "stable": True}

        findings = robustness_check(components, analyze_fn=mock_analyze)
        assert len(findings) == 1
        assert findings[0].severity == "NOTE"


# ==================================================================
# VECTOR 7: HISTORY MATCHER
# ==================================================================

class TestHistoryMatcher:

    def test_matches_mianus_river(self, sample_bridge_info):
        findings, matches = history_matcher(sample_bridge_info)
        # Should match Mianus River (steel_girder + steel + pin_hanger + fracture_critical)
        match_names = [m.failure_name for m in matches]
        assert "Mianus River Bridge" in match_names

    def test_score_threshold(self, sample_bridge_info):
        findings, matches = history_matcher(sample_bridge_info, score_threshold=5)
        for m in matches:
            assert m.score >= 5

    def test_high_threshold_filters(self, sample_bridge_info):
        _, matches = history_matcher(sample_bridge_info, score_threshold=100)
        assert matches == []

    def test_matching_factors_populated(self, sample_bridge_info):
        _, matches = history_matcher(sample_bridge_info)
        for m in matches:
            assert len(m.matching_factors) > 0

    def test_lesson_populated(self, sample_bridge_info):
        _, matches = history_matcher(sample_bridge_info)
        for m in matches:
            assert m.lesson  # Non-empty lesson

    def test_no_match_for_unrelated(self):
        info = {"type": "cable_stayed", "material": "frp",
                "max_span_ft": 5000, "details": []}
        _, matches = history_matcher(info)
        # Very few or no matches expected
        for m in matches:
            assert m.score >= 5  # Threshold still applies

    def test_empty_bridge_info(self):
        findings, matches = history_matcher({})
        # Should not crash, may have some low scores
        assert isinstance(findings, list)


# ==================================================================
# FAILURE DATABASE
# ==================================================================

class TestFailureDatabase:

    def test_loads_successfully(self):
        db = load_failure_database()
        assert isinstance(db, list)
        assert len(db) > 0

    def test_has_required_fields(self):
        db = load_failure_database()
        required = {"name", "year", "type", "material", "failure_mode",
                    "cause", "lesson", "details"}
        for record in db:
            for field in required:
                assert field in record, f"Missing field '{field}' in {record.get('name', '?')}"

    def test_known_failures_present(self):
        db = load_failure_database()
        names = [r["name"] for r in db]
        expected = [
            "I-35W Mississippi River Bridge",
            "Morandi Bridge",
            "FIU Pedestrian Bridge",
            "Mianus River Bridge",
            "Silver Bridge",
        ]
        for exp in expected:
            # Partial match
            assert any(exp in n for n in names), f"Missing: {exp}"

    def test_spans_are_lists(self):
        db = load_failure_database()
        for record in db:
            assert isinstance(record.get("spans", []), list)

    def test_details_are_lists(self):
        db = load_failure_database()
        for record in db:
            assert isinstance(record.get("details", []), list)


# ==================================================================
# RISK RATING
# ==================================================================

class TestRiskRating:

    def test_red_on_critical(self):
        findings = [Finding("CRITICAL", "test", 1, "loc", "desc",
                            1.2, "combo", "rec")]
        assert compute_risk_rating(findings) == "RED"

    def test_red_on_collapse(self):
        findings = [Finding("NOTE", "test", 1, "loc", "desc",
                            0.5, "combo", "rec")]
        chains = [CascadeChain(1, [], True)]
        assert compute_risk_rating(findings, cascade_chains=chains) == "RED"

    def test_yellow_on_warning(self):
        findings = [Finding("WARNING", "test", 1, "loc", "desc",
                            0.90, "combo", "rec")]
        assert compute_risk_rating(findings) == "YELLOW"

    def test_yellow_on_dominant_sensitivity(self):
        findings = [Finding("NOTE", "test", 1, "loc", "desc",
                            0.5, "combo", "rec")]
        sens = [SensitivityResult("param", 0.80, 0.60, 1.0, 0.20, "DOMINANT")]
        assert compute_risk_rating(findings, sensitivity_results=sens) == "YELLOW"

    def test_yellow_on_adversarial_exceedance(self):
        findings = [Finding("NOTE", "test", 1, "loc", "desc",
                            0.5, "combo", "rec")]
        assert compute_risk_rating(findings, adversarial_exceedance_max=0.20) == "YELLOW"

    def test_green_all_clear(self):
        findings = [Finding("NOTE", "test", 1, "loc", "desc",
                            0.5, "combo", "rec")]
        assert compute_risk_rating(findings) == "GREEN"

    def test_green_empty(self):
        assert compute_risk_rating([]) == "GREEN"

    def test_red_trumps_yellow(self):
        findings = [
            Finding("CRITICAL", "test", 1, "loc", "desc", 1.2, "combo", "rec"),
            Finding("WARNING", "test", 2, "loc", "desc", 0.90, "combo", "rec"),
        ]
        assert compute_risk_rating(findings) == "RED"


# ==================================================================
# RUN RED TEAM (INTEGRATION)
# ==================================================================

class TestRunRedTeam:

    def test_runs_all_vectors(self, sample_element_results, sample_bridge_info):
        report = run_red_team(
            element_results=sample_element_results,
            bridge_info=sample_bridge_info,
            sensitivity_base_dcr=0.80,
            sensitivity_parameter_results=[
                {"parameter": "Soil Stiffness", "low_dcr": 0.65, "high_dcr": 0.95},
            ],
            adversarial_results=[
                {"combo_name": "Test", "max_demand": 1200,
                 "demand_type": "moment", "element": 1, "location": "test"},
            ],
            standard_results=[
                {"combo_name": "Strength_I", "max_demand": 1000,
                 "demand_type": "moment"},
            ],
            robustness_results=[
                {"component": "Girder G1", "max_dcr": 0.75,
                 "stable": True, "type": "girder"},
            ],
        )
        assert isinstance(report, RedTeamReport)
        assert report.risk_rating in ("GREEN", "YELLOW", "RED")
        assert len(report.findings) > 0
        assert len(report.attack_vectors_run) >= 5
        assert report.analysis_time_sec >= 0

    def test_subset_vectors(self, sample_element_results):
        report = run_red_team(
            element_results=sample_element_results,
            vectors=["dcr"],
        )
        assert report.attack_vectors_run == ["DCR Scanner"]
        assert len(report.findings) > 0

    def test_empty_input(self):
        report = run_red_team(element_results=[])
        assert report.risk_rating == "GREEN"
        assert report.findings == []

    def test_summary_generated(self, sample_element_results):
        report = run_red_team(element_results=sample_element_results)
        assert report.summary
        assert "finding" in report.summary.lower()

    def test_to_dict(self, sample_element_results):
        report = run_red_team(element_results=sample_element_results)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "risk_rating" in d
        assert "findings" in d


# ==================================================================
# REPORT GENERATION
# ==================================================================

class TestReportGeneration:

    def _make_report(self, sample_element_results):
        return run_red_team(
            element_results=sample_element_results,
            sensitivity_base_dcr=0.80,
            sensitivity_parameter_results=[
                {"parameter": "Soil Stiffness", "low_dcr": 0.65, "high_dcr": 0.95},
                {"parameter": "Concrete Strength", "low_dcr": 0.78, "high_dcr": 0.82},
            ],
        )

    def test_executive_is_markdown(self, sample_element_results,
                                    sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "executive")
        assert text.startswith("#")
        assert "Risk Rating" in text

    def test_executive_under_500_words(self, sample_element_results,
                                       sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "executive")
        word_count = len(text.split())
        assert word_count < 500, f"Executive summary is {word_count} words (limit: 500)"

    def test_executive_has_top_findings(self, sample_element_results,
                                        sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "executive")
        assert "Top Findings" in text

    def test_executive_has_recommendation(self, sample_element_results,
                                          sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "executive")
        assert "Recommendation" in text

    def test_technical_has_all_sections(self, sample_element_results,
                                        sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "technical")
        for section_num in range(1, 12):
            assert f"## {section_num}." in text, f"Missing section {section_num}"

    def test_technical_is_markdown(self, sample_element_results,
                                   sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "technical")
        assert text.startswith("#")
        assert "## 1." in text

    def test_raw_is_valid_json(self, sample_element_results,
                                sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "raw")
        data = json.loads(text)
        assert "risk_rating" in data
        assert "findings" in data
        assert isinstance(data["findings"], list)

    def test_raw_has_all_fields(self, sample_element_results,
                                 sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        text = generate_report(report, sample_model_info, sample_site, "raw")
        data = json.loads(text)
        expected_keys = [
            "report_type", "risk_rating", "summary", "model_info", "site",
            "findings", "cascade_chains", "sensitivity_results",
            "history_matches", "attack_vectors_run",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_invalid_tier_raises(self, sample_element_results,
                                  sample_model_info, sample_site):
        report = self._make_report(sample_element_results)
        with pytest.raises(ValueError, match="Unknown report tier"):
            generate_report(report, sample_model_info, sample_site, "invalid")

    def test_empty_report_green(self, sample_model_info, sample_site):
        report = RedTeamReport()
        report.risk_rating = "GREEN"
        text = generate_report(report, sample_model_info, sample_site, "executive")
        assert "GREEN" in text

    def test_green_report_produces_markdown(self, sample_model_info, sample_site):
        report = RedTeamReport()
        report.risk_rating = "GREEN"
        report.summary = "No issues found."
        text = generate_report(report, sample_model_info, sample_site, "technical")
        assert "## 1." in text


# ==================================================================
# SVG VISUALIZATIONS
# ==================================================================

class TestSVGVisualizations:

    def test_tornado_has_svg_tags(self, sensitivity_results):
        svg = generate_tornado_svg(sensitivity_results)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'class="tornado-diagram"' in svg

    def test_tornado_has_bars(self, sensitivity_results):
        svg = generate_tornado_svg(sensitivity_results)
        assert "<rect" in svg
        # Should have one bar per result
        assert svg.count("class=\"bar\"") == len(sensitivity_results)

    def test_tornado_has_labels(self, sensitivity_results):
        svg = generate_tornado_svg(sensitivity_results)
        for sr in sensitivity_results:
            assert sr.parameter in svg

    def test_tornado_has_base_line(self, sensitivity_results):
        svg = generate_tornado_svg(sensitivity_results)
        assert "base-line" in svg

    def test_tornado_empty(self):
        svg = generate_tornado_svg([])
        assert "<svg" in svg
        assert "No data" in svg

    def test_dcr_heatmap_structure(self):
        elements = [
            {"element": 1, "x_start": 0, "x_end": 100, "dcr": 0.50},
            {"element": 2, "x_start": 100, "x_end": 200, "dcr": 0.85},
            {"element": 3, "x_start": 200, "x_end": 300, "dcr": 1.10},
        ]
        svg = generate_dcr_heatmap_svg(elements)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'class="dcr-heatmap"' in svg

    def test_dcr_heatmap_colors(self):
        elements = [
            {"element": 1, "x_start": 0, "x_end": 100, "dcr": 0.50},
            {"element": 2, "x_start": 100, "x_end": 200, "dcr": 1.10},
        ]
        svg = generate_dcr_heatmap_svg(elements)
        assert "#27ae60" in svg  # Green for low DCR
        assert "#e74c3c" in svg  # Red for high DCR

    def test_dcr_heatmap_empty(self):
        svg = generate_dcr_heatmap_svg([])
        assert "No data" in svg

    def test_cascade_svg_structure(self, cascade_chains):
        svg = generate_cascade_svg(cascade_chains)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'class="cascade-diagram"' in svg

    def test_cascade_svg_has_boxes(self, cascade_chains):
        svg = generate_cascade_svg(cascade_chains)
        assert "<rect" in svg
        assert "arrowhead" in svg

    def test_cascade_svg_empty(self):
        svg = generate_cascade_svg([])
        assert "No cascades" in svg

    def test_risk_dashboard_structure(self, sample_element_results):
        report = run_red_team(element_results=sample_element_results)
        svg = generate_risk_dashboard_svg(report)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'class="risk-dashboard"' in svg
        assert report.risk_rating in svg

    def test_risk_dashboard_green(self):
        report = RedTeamReport(risk_rating="GREEN")
        svg = generate_risk_dashboard_svg(report)
        assert "#27ae60" in svg  # Green color

    def test_risk_dashboard_red(self):
        report = RedTeamReport(risk_rating="RED")
        svg = generate_risk_dashboard_svg(report)
        assert "#e74c3c" in svg  # Red color


# ==================================================================
# EDGE CASES
# ==================================================================

class TestEdgeCases:

    def test_finding_to_dict(self):
        f = Finding("CRITICAL", "test", 1, "loc", "desc", 1.2, "combo", "rec", "Silver Bridge")
        d = f.to_dict()
        assert d["severity"] == "CRITICAL"
        assert d["precedent"] == "Silver Bridge"

    def test_sensitivity_with_zero_base(self):
        params = [
            {"parameter": "Test", "low_dcr": 0.0, "high_dcr": 0.1},
        ]
        findings, results = sensitivity_sweep(0.0, parameter_results=params)
        assert results[0].classification == "DOMINANT"

    def test_very_high_dcr(self):
        results = [
            {"element": 1, "dcr": 5.0, "force_type": "moment",
             "controlling_combo": "test", "location": "test"},
        ]
        findings = dcr_scanner(results)
        assert findings[0].severity == "CRITICAL"
        assert findings[0].dcr == 5.0

    def test_report_with_all_vectors(self, sample_element_results,
                                      sample_bridge_info, sample_model_info,
                                      sample_site):
        stage_results = [
            {"stage_name": "Pour Deck", "stage_number": 1,
             "max_dcr": 0.80, "max_dcr_element": 5,
             "max_dcr_location": "Span 1"},
        ]
        report = run_red_team(
            element_results=sample_element_results,
            bridge_info=sample_bridge_info,
            stage_results=stage_results,
            sensitivity_base_dcr=0.80,
            sensitivity_parameter_results=[
                {"parameter": "Test", "low_dcr": 0.65, "high_dcr": 0.95},
            ],
            adversarial_results=[
                {"combo_name": "Scour+EQ", "max_demand": 1250,
                 "demand_type": "moment", "element": 1, "location": "Pier 1"},
            ],
            standard_results=[
                {"combo_name": "EE_I", "max_demand": 1000,
                 "demand_type": "moment"},
            ],
            robustness_results=[
                {"component": "Girder G1", "max_dcr": 0.75,
                 "stable": True, "type": "girder"},
            ],
        )
        # Generate all report tiers
        exec_text = generate_report(report, sample_model_info, sample_site, "executive")
        tech_text = generate_report(report, sample_model_info, sample_site, "technical")
        raw_text = generate_report(report, sample_model_info, sample_site, "raw")

        assert "RED" in exec_text or "YELLOW" in exec_text or "GREEN" in exec_text
        assert "## 11." in tech_text
        data = json.loads(raw_text)
        assert len(data["findings"]) > 0

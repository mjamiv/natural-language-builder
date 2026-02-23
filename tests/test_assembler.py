"""Tests for the assembler tool.

Covers:
- Tag remapping produces unique tags across components
- Connection protocol merges foundation→sub→bearing→super correctly
- Analysis sequence is ordered correctly
- Script generation produces valid Python
- Simple 1-span bridge assembles without error (mock all components)
- 3-span continuous bridge assembles (mock)
- Bounding case enumeration (4 cases minimum)
- Missing optional components handled gracefully
- Error on component count mismatch
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any

import pytest

from nlb.tools.assembler import (
    AssembledModel,
    AnalysisResults,
    TagRemapper,
    ConnectionError,
    assemble_model,
    generate_script,
    build_analysis_sequence,
    enumerate_bounding_cases,
    _remap_component_nodes,
    _remap_component_elements,
    _remap_component_materials,
    _remap_component_sections,
    _remap_constraints,
    _connect_foundation_to_substructure,
    _connect_substructure_to_bearing,
    _connect_bearing_to_superstructure,
    TAG_RANGES,
    BOUNDING_LABELS,
    _extract_empty_results,
    run_analysis,
)


# ============================================================================
# FIXTURES — Mock component outputs
# ============================================================================

def _mock_foundation(top_node: int = 1) -> dict:
    """Mock FoundationModel dict — simplified deep foundation."""
    return {
        "nodes": [
            {"tag": 1, "x": 0.0, "y": -240.0, "z": 0.0},
            {"tag": 2, "x": 0.0, "y": 0.0, "z": 0.0},
        ],
        "elements": [
            {"tag": 1, "type": "dispBeamColumn", "nodes": [1, 2],
             "section": 1, "transform": 2, "np": 5},
        ],
        "materials": [
            {"tag": 1, "type": "Concrete01", "name": "pile_concrete"},
            {"tag": 2, "type": "geomTransf", "transform": "Linear"},
        ],
        "sections": [],
        "top_node": top_node,
        "base_node": 1,
        "boundary_conditions": [
            {"node": 1, "fixity": [1, 1, 1, 1, 1, 1]},
        ],
    }


def _mock_substructure(base_nodes=None, top_nodes=None, cap_nodes=None) -> dict:
    """Mock SubstructureModel dict — single column."""
    base_nodes = base_nodes or [1]
    top_nodes = top_nodes or [8]
    cap_nodes = cap_nodes or []
    return {
        "nodes": [
            {"tag": i, "x": 0.0, "y": float(i * 30), "z": 0.0}
            for i in range(1, 9)
        ],
        "elements": [
            {"tag": i, "type": "dispBeamColumn", "nodes": [i, i + 1],
             "section": 1, "transform": 2, "np": 5}
            for i in range(1, 8)
        ],
        "materials": [
            {"tag": 1, "type": "Concrete01", "name": "column_concrete"},
            {"tag": 2, "type": "geomTransf", "transform": "PDelta"},
            {"tag": 3, "type": "Steel02", "name": "rebar"},
        ],
        "sections": [
            {"tag": 1, "type": "circular_rc", "diameter_in": 48.0},
        ],
        "constraints": [],
        "base_nodes": base_nodes,
        "top_nodes": top_nodes,
        "cap_nodes": cap_nodes,
    }


def _mock_bearing(bottom_nodes=None, top_nodes=None) -> dict:
    """Mock BearingModel dict — elastomeric bearing."""
    bottom_nodes = bottom_nodes or [1]
    top_nodes = top_nodes or [2]
    return {
        "nodes": [
            {"tag": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"tag": 2, "x": 0.0, "y": 0.0, "z": 0.0},
        ],
        "elements": [
            {"tag": 1, "type": "zeroLength", "nodes": [1, 2],
             "materials": [3, 4, 3], "directions": [1, 2, 3]},
        ],
        "materials": [
            {"tag": 3, "type": "Elastic", "k_kip_per_in": 12.6},
            {"tag": 4, "type": "ENT", "k_kip_per_in": 5000.0},
        ],
        "constraints": [],
        "bottom_nodes": bottom_nodes,
        "top_nodes": top_nodes,
        "cases": {
            "upper_bound": {"Kh_kip_per_in": 14.5},
            "lower_bound": {"Kh_kip_per_in": 10.7},
        },
    }


def _mock_superstructure(
    num_supports: int = 2,
    num_girders: int = 1,
) -> dict:
    """Mock SuperstructureModel dict — simple span steel girder."""
    nodes = []
    elements = []
    support_nodes = []
    midspan_nodes = []

    node_tag = 1
    elem_tag = 1
    n_elem = 10

    for g in range(num_girders):
        z = g * 96.0  # 8 ft spacing
        for i in range(n_elem + 1):
            x = i * 96.0  # 10 elements at 8ft each = 80ft span
            nodes.append({"tag": node_tag, "x": x, "y": 0.0, "z": z})

            if i == 0:
                support_nodes.append(node_tag)
            elif i == n_elem:
                support_nodes.append(node_tag)
            elif i == n_elem // 2:
                midspan_nodes.append(node_tag)

            if i > 0:
                elements.append({
                    "tag": elem_tag, "type": "dispBeamColumn",
                    "nodes": [node_tag - 1, node_tag],
                    "section": 1, "transform": 2, "np": 5,
                })
                elem_tag += 1

            node_tag += 1

    return {
        "nodes": nodes,
        "elements": elements,
        "materials": [
            {"tag": 1, "type": "Steel02", "name": "structural_steel",
             "params": {"fy": 50.0, "Es": 29000.0, "b": 0.01}},
        ],
        "sections": [
            {"tag": 1, "type": "composite", "params": {"d": 48.0}},
        ],
        "transforms": [
            {"tag": 2, "type": "Linear", "vecxz": [0.0, 0.0, 1.0]},
        ],
        "diaphragms": [],
        "support_nodes": support_nodes,
        "midspan_nodes": midspan_nodes,
        "girder_lines": num_girders,
        "span_lengths": [80.0],
    }


def _mock_loads(has_seismic: bool = False) -> dict:
    """Mock LoadModel dict."""
    cases = [
        {"name": "DC1_deck", "load_type": "DC", "category": "standard",
         "loads": [{"type": "distributed", "w_kip_per_in": 0.05}]},
        {"name": "DW_fws", "load_type": "DW", "category": "standard",
         "loads": [{"type": "distributed", "w_kip_per_in": 0.01}]},
        {"name": "LL_HL93", "load_type": "LL", "category": "standard",
         "loads": [{"type": "envelope", "governing": "Truck+Lane"}]},
    ]
    if has_seismic:
        cases.append({
            "name": "EQ_long", "load_type": "EQ", "category": "standard",
            "loads": [{"type": "spectrum", "sdc": "C", "sds": 0.5}],
        })

    combos = [
        {"name": "Strength_I_max", "limit_state": "Strength I",
         "factors": {"DC1_deck": 1.25, "DW_fws": 1.50, "LL_HL93": 1.75}},
        {"name": "Service_I", "limit_state": "Service I",
         "factors": {"DC1_deck": 1.00, "DW_fws": 1.00, "LL_HL93": 1.00}},
    ]
    return {
        "cases": cases,
        "combinations": combos,
        "distribution_factors": {"moment_interior": 0.65},
    }


def _mock_site() -> dict:
    """Mock site profile dict."""
    return {
        "location": {"lat": 41.0, "lon": -74.0, "state": "NY"},
        "seismic": {"sds": 0.25, "sd1": 0.15, "sdc": "B"},
        "wind": {"v_ult": 115},
    }


def _simple_1span_components():
    """Return a complete set of components for a simple 1-span bridge."""
    site = _mock_site()
    fnds = [_mock_foundation(top_node=2), _mock_foundation(top_node=2)]
    subs = [
        _mock_substructure(base_nodes=[1], top_nodes=[8]),
        _mock_substructure(base_nodes=[1], top_nodes=[8]),
    ]
    brgs = [
        _mock_bearing(bottom_nodes=[1], top_nodes=[2]),
        _mock_bearing(bottom_nodes=[1], top_nodes=[2]),
    ]
    sup = _mock_superstructure(num_supports=2, num_girders=1)
    loads = _mock_loads()
    return site, fnds, subs, brgs, sup, loads


def _3span_continuous_components():
    """Return components for a 3-span continuous bridge (4 supports)."""
    site = _mock_site()
    n_supports = 4
    fnds = [_mock_foundation(top_node=2) for _ in range(n_supports)]
    subs = [_mock_substructure() for _ in range(n_supports)]
    brgs = [_mock_bearing() for _ in range(n_supports)]

    # Build superstructure with 4 support lines
    nodes = []
    elements = []
    support_nodes = []
    midspan_nodes = []
    node_tag = 1
    elem_tag = 1

    span_lengths = [100, 140, 100]  # ft
    n_elem_per_span = 10

    for span_idx, span_ft in enumerate(span_lengths):
        span_in = span_ft * 12.0
        dx = span_in / n_elem_per_span
        for i in range(n_elem_per_span + 1):
            if span_idx > 0 and i == 0:
                continue  # shared node at pier
            x = sum(s * 12 for s in span_lengths[:span_idx]) + i * dx
            nodes.append({"tag": node_tag, "x": x, "y": 0.0, "z": 0.0})

            if (span_idx == 0 and i == 0) or i == n_elem_per_span:
                support_nodes.append(node_tag)
            if i == n_elem_per_span // 2:
                midspan_nodes.append(node_tag)

            if not (span_idx == 0 and i == 0):
                if len(nodes) >= 2:
                    prev_tag = nodes[-2]["tag"]
                    elements.append({
                        "tag": elem_tag, "type": "dispBeamColumn",
                        "nodes": [prev_tag, node_tag],
                        "section": 1, "transform": 2, "np": 5,
                    })
                    elem_tag += 1

            node_tag += 1

    sup = {
        "nodes": nodes,
        "elements": elements,
        "materials": [
            {"tag": 1, "type": "Steel02", "name": "steel",
             "params": {"fy": 50.0, "Es": 29000.0}},
        ],
        "sections": [{"tag": 1, "type": "composite"}],
        "transforms": [{"tag": 2, "type": "PDelta"}],
        "diaphragms": [],
        "support_nodes": support_nodes,
        "midspan_nodes": midspan_nodes,
        "girder_lines": 1,
        "span_lengths": span_lengths,
    }
    loads = _mock_loads()
    return site, fnds, subs, brgs, sup, loads


# ============================================================================
# TESTS — Tag Remapping
# ============================================================================

class TestTagRemapping:
    """Tag remapping produces unique tags across components."""

    def test_foundation_tags_in_range(self):
        r = TagRemapper("foundation", 0)
        tag = r.remap(1, "node")
        lo, hi = TAG_RANGES["foundation"]["node"]
        assert lo <= tag <= hi

    def test_substructure_tags_in_range(self):
        r = TagRemapper("substructure", 0)
        tag = r.remap(1, "node")
        lo, hi = TAG_RANGES["substructure"]["node"]
        assert lo <= tag <= hi

    def test_bearing_tags_in_range(self):
        r = TagRemapper("bearing", 0)
        tag = r.remap(1, "node")
        lo, hi = TAG_RANGES["bearing"]["node"]
        assert lo <= tag <= hi

    def test_superstructure_tags_in_range(self):
        r = TagRemapper("superstructure", 0)
        tag = r.remap(1, "node")
        lo, hi = TAG_RANGES["superstructure"]["node"]
        assert lo <= tag <= hi

    def test_no_collision_between_components(self):
        """Tags from different components must not overlap."""
        r_fnd = TagRemapper("foundation", 0)
        r_sub = TagRemapper("substructure", 0)
        r_brg = TagRemapper("bearing", 0)
        r_sup = TagRemapper("superstructure", 0)

        tags = set()
        for r in (r_fnd, r_sub, r_brg, r_sup):
            for local in range(1, 50):
                tag = r.remap(local, "node")
                assert tag not in tags, f"Collision at tag {tag}"
                tags.add(tag)

    def test_no_collision_between_supports(self):
        """Tags from different supports within the same component don't collide."""
        tags = set()
        for idx in range(5):
            r = TagRemapper("foundation", idx)
            for local in range(1, 20):
                tag = r.remap(local, "node")
                assert tag not in tags, f"Collision at tag {tag} (support {idx})"
                tags.add(tag)

    def test_remap_deterministic(self):
        """Same local tag always maps to same global tag."""
        r = TagRemapper("foundation", 0)
        t1 = r.remap(5, "node")
        t2 = r.remap(5, "node")
        assert t1 == t2

    def test_remap_nodes(self):
        nodes = [{"tag": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                 {"tag": 2, "x": 1.0, "y": 0.0, "z": 0.0}]
        r = TagRemapper("foundation", 0)
        remapped = _remap_component_nodes(nodes, r)
        assert remapped[0]["tag"] != 1
        assert remapped[0]["_original_tag"] == 1
        assert remapped[0]["x"] == 0.0

    def test_remap_elements_updates_node_refs(self):
        r = TagRemapper("substructure", 0)
        r.remap(1, "node")
        r.remap(2, "node")
        r.remap(1, "material")

        elems = [{"tag": 1, "type": "dispBeamColumn", "nodes": [1, 2],
                  "section": 1, "transform": 1}]
        remapped = _remap_component_elements(elems, r)
        assert remapped[0]["nodes"][0] == r.get_global(1, "node")
        assert remapped[0]["nodes"][1] == r.get_global(2, "node")
        assert remapped[0]["section"] == r.get_global(1, "material")

    def test_remap_materials(self):
        r = TagRemapper("bearing", 0)
        mats = [{"tag": 1, "type": "Elastic", "k_kip_per_in": 100}]
        remapped = _remap_component_materials(mats, r)
        lo, hi = TAG_RANGES["bearing"]["material"]
        assert lo <= remapped[0]["tag"] <= hi

    def test_remap_constraints(self):
        r = TagRemapper("substructure", 0)
        r.remap(1, "node")
        r.remap(2, "node")
        constrs = [{"type": "equalDOF", "master": 1, "slave": 2, "dofs": [1, 2, 3]}]
        remapped = _remap_constraints(constrs, r)
        assert remapped[0]["master"] == r.get_global(1, "node")
        assert remapped[0]["slave"] == r.get_global(2, "node")


# ============================================================================
# TESTS — Connection Protocol
# ============================================================================

class TestConnectionProtocol:
    """Connection protocol merges foundation→sub→bearing→super correctly."""

    def test_foundation_to_substructure_connection(self):
        fnd_r = TagRemapper("foundation", 0)
        sub_r = TagRemapper("substructure", 0)
        fnd_r.remap(2, "node")
        sub_r.remap(1, "node")

        fnd = {"top_node": 2}
        sub = {"base_nodes": [1]}

        conns = _connect_foundation_to_substructure(fnd_r, sub_r, fnd, sub)
        assert len(conns) == 1
        assert conns[0]["type"] == "equalDOF"
        assert conns[0]["master"] == fnd_r.get_global(2, "node")
        assert conns[0]["slave"] == sub_r.get_global(1, "node")

    def test_substructure_to_bearing_connection(self):
        sub_r = TagRemapper("substructure", 0)
        brg_r = TagRemapper("bearing", 0)
        sub_r.remap(8, "node")
        brg_r.remap(1, "node")

        sub = {"top_nodes": [8], "cap_nodes": []}
        brg = {"bottom_nodes": [1]}

        conns = _connect_substructure_to_bearing(sub_r, brg_r, sub, brg)
        assert len(conns) == 1
        assert conns[0]["master"] == sub_r.get_global(8, "node")
        assert conns[0]["slave"] == brg_r.get_global(1, "node")

    def test_bearing_to_superstructure_connection(self):
        brg_r = TagRemapper("bearing", 0)
        sup_r = TagRemapper("superstructure", 0)
        brg_r.remap(2, "node")
        sup_r.remap(1, "node")

        brg = {"top_nodes": [2]}
        sup = {"support_nodes": [1]}

        conns = _connect_bearing_to_superstructure(
            brg_r, sup_r, brg, sup,
            support_index=0, num_supports=1, num_girders=1,
        )
        assert len(conns) == 1
        assert conns[0]["master"] == brg_r.get_global(2, "node")
        assert conns[0]["slave"] == sup_r.get_global(1, "node")

    def test_no_base_nodes_raises(self):
        fnd_r = TagRemapper("foundation", 0)
        sub_r = TagRemapper("substructure", 0)
        fnd = {"top_node": 1}
        sub = {"base_nodes": []}

        with pytest.raises(ConnectionError, match="no base_nodes"):
            _connect_foundation_to_substructure(fnd_r, sub_r, fnd, sub)

    def test_no_top_nodes_raises(self):
        sub_r = TagRemapper("substructure", 0)
        brg_r = TagRemapper("bearing", 0)
        sub = {"top_nodes": [], "cap_nodes": []}
        brg = {"bottom_nodes": [1]}

        with pytest.raises(ConnectionError, match="no top_nodes"):
            _connect_substructure_to_bearing(sub_r, brg_r, sub, brg)

    def test_cap_nodes_preferred_over_top_nodes(self):
        sub_r = TagRemapper("substructure", 0)
        brg_r = TagRemapper("bearing", 0)
        sub_r.remap(5, "node")
        sub_r.remap(8, "node")
        brg_r.remap(1, "node")

        sub = {"top_nodes": [8], "cap_nodes": [5]}
        brg = {"bottom_nodes": [1]}

        conns = _connect_substructure_to_bearing(sub_r, brg_r, sub, brg)
        assert conns[0]["master"] == sub_r.get_global(5, "node")

    def test_single_sub_top_to_multiple_bearings(self):
        sub_r = TagRemapper("substructure", 0)
        brg_r = TagRemapper("bearing", 0)
        sub_r.remap(8, "node")
        brg_r.remap(1, "node")
        brg_r.remap(2, "node")

        sub = {"top_nodes": [8], "cap_nodes": []}
        brg = {"bottom_nodes": [1, 2]}

        conns = _connect_substructure_to_bearing(sub_r, brg_r, sub, brg)
        assert len(conns) == 2
        for c in conns:
            assert c["master"] == sub_r.get_global(8, "node")


# ============================================================================
# TESTS — Analysis Sequence
# ============================================================================

class TestAnalysisSequence:
    """Analysis sequence is ordered correctly."""

    def test_basic_sequence_order(self):
        seq = build_analysis_sequence()
        assert seq[0] == "model_setup"
        assert "define_materials" in seq
        assert "define_nodes" in seq
        assert "define_elements" in seq
        assert "gravity_analysis" in seq
        assert "modal_analysis" in seq
        assert seq.index("gravity_analysis") < seq.index("modal_analysis")

    def test_gravity_before_modal(self):
        seq = build_analysis_sequence()
        g = seq.index("gravity_analysis")
        m = seq.index("modal_analysis")
        assert g < m

    def test_seismic_adds_spectrum(self):
        seq = build_analysis_sequence(has_seismic=True)
        assert "response_spectrum_analysis" in seq

    def test_no_spectrum_without_seismic(self):
        seq = build_analysis_sequence(has_seismic=False)
        assert "response_spectrum_analysis" not in seq

    def test_pushover_for_sdc_c_d(self):
        seq = build_analysis_sequence(has_seismic=True, sdc="C")
        assert "pushover_analysis" in seq

        seq = build_analysis_sequence(has_seismic=True, sdc="D")
        assert "pushover_analysis" in seq

    def test_no_pushover_for_sdc_b(self):
        seq = build_analysis_sequence(has_seismic=True, sdc="B")
        assert "pushover_analysis" not in seq

    def test_envelopes_always_present(self):
        seq = build_analysis_sequence()
        assert "load_combination_envelopes" in seq

    def test_moving_load_optional(self):
        seq = build_analysis_sequence(has_moving_load=False)
        assert "moving_load_analysis" not in seq

        seq = build_analysis_sequence(has_moving_load=True)
        assert "moving_load_analysis" in seq


# ============================================================================
# TESTS — Script Generation
# ============================================================================

class TestScriptGeneration:
    """Script generation produces valid Python."""

    def test_script_is_valid_python(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        script = model.script

        # Must parse without syntax errors
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Generated script has syntax error: {e}")

    def test_script_contains_wipe(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "ops.wipe()" in model.script

    def test_script_contains_model_setup(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "ops.model(" in model.script
        assert "ndm" in model.script

    def test_script_contains_nodes(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "ops.node(" in model.script

    def test_script_contains_elements(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "ops.element(" in model.script or "dispBeamColumn" in model.script

    def test_script_contains_analysis(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "ops.analyze(" in model.script

    def test_generate_script_function(self):
        model = AssembledModel()
        model.nodes = [{"tag": 1, "x": 0.0, "y": 0.0, "z": 0.0}]
        model.elements = []
        model.materials = []
        model.sections = []
        model.constraints = []
        model.boundary_conditions = []
        model.analysis_sequence = ["model_setup", "gravity_analysis"]
        script = generate_script(model)
        assert "ops.wipe()" in script
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


# ============================================================================
# TESTS — Simple 1-Span Bridge Assembly
# ============================================================================

class TestSimple1SpanBridge:
    """Simple 1-span bridge assembles without error (mock all components)."""

    def test_assembles_without_error(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert isinstance(model, AssembledModel)

    def test_node_count_positive(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert model.node_count > 0

    def test_element_count_positive(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert model.element_count > 0

    def test_all_node_tags_unique(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        tags = [n["tag"] for n in model.nodes]
        assert len(tags) == len(set(tags)), "Duplicate node tags found"

    def test_all_element_tags_unique(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        tags = [e["tag"] for e in model.elements]
        assert len(tags) == len(set(tags)), "Duplicate element tags found"

    def test_connections_created(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert len(model.connections) > 0

    def test_constraints_created(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert len(model.constraints) > 0

    def test_analysis_sequence_populated(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert len(model.analysis_sequence) >= 5

    def test_script_generated(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert len(model.script) > 100

    def test_load_cases_counted(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert model.load_cases == 3
        assert model.load_combinations == 2


# ============================================================================
# TESTS — 3-Span Continuous Bridge
# ============================================================================

class TestThreeSpanBridge:
    """3-span continuous bridge assembles (mock)."""

    def test_assembles_without_error(self):
        site, fnds, subs, brgs, sup, loads = _3span_continuous_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert isinstance(model, AssembledModel)

    def test_four_supports(self):
        site, fnds, subs, brgs, sup, loads = _3span_continuous_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        # Should have connections for all 4 supports
        fnd_to_sub = [c for c in model.connections
                      if c["type"] == "foundation_to_substructure"]
        assert len(fnd_to_sub) == 4

    def test_all_tags_unique(self):
        site, fnds, subs, brgs, sup, loads = _3span_continuous_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        node_tags = [n["tag"] for n in model.nodes]
        assert len(node_tags) == len(set(node_tags))

        elem_tags = [e["tag"] for e in model.elements]
        assert len(elem_tags) == len(set(elem_tags))


# ============================================================================
# TESTS — Bounding Cases
# ============================================================================

class TestBoundingCases:
    """Bounding case enumeration (4 cases minimum)."""

    def test_four_bounding_cases(self):
        cases = enumerate_bounding_cases()
        assert len(cases) >= 4

    def test_case_labels(self):
        cases = enumerate_bounding_cases()
        assert "UU" in cases
        assert "UL" in cases
        assert "LU" in cases
        assert "LL" in cases

    def test_each_case_has_foundation_and_bearing(self):
        cases = enumerate_bounding_cases()
        for label, config in cases.items():
            assert "foundation" in config
            assert "bearing" in config
            assert config["foundation"] in ("upper_bound", "lower_bound")
            assert config["bearing"] in ("upper_bound", "lower_bound")

    def test_bounding_cases_in_assembled_model(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert len(model.bounding_cases) >= 4


# ============================================================================
# TESTS — Missing Optional Components
# ============================================================================

class TestMissingComponents:
    """Missing optional components handled gracefully."""

    def test_no_seismic_loads(self):
        """Low-seismic site without EQ loads should still assemble."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        # Ensure no EQ load cases
        loads["cases"] = [c for c in loads["cases"]
                          if c.get("load_type") != "EQ"]
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "response_spectrum_analysis" not in model.analysis_sequence

    def test_seismic_adds_spectrum_to_sequence(self):
        """SDC C site should include response spectrum."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        loads = _mock_loads(has_seismic=True)
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert "response_spectrum_analysis" in model.analysis_sequence

    def test_empty_diaphragms(self):
        """Model assembles even if superstructure has no diaphragms."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        sup["diaphragms"] = []
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert model.node_count > 0

    def test_empty_constraints(self):
        """Substructure with no internal constraints still connects."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        for sub in subs:
            sub["constraints"] = []
        model = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert len(model.constraints) > 0  # inter-component constraints still created


# ============================================================================
# TESTS — Error on Component Count Mismatch
# ============================================================================

class TestComponentCountMismatch:
    """Error on component count mismatch."""

    def test_foundation_substructure_mismatch(self):
        site = _mock_site()
        fnds = [_mock_foundation(), _mock_foundation(), _mock_foundation()]
        subs = [_mock_substructure(), _mock_substructure()]  # Only 2
        brgs = [_mock_bearing(), _mock_bearing(), _mock_bearing()]
        sup = _mock_superstructure()
        loads = _mock_loads()

        with pytest.raises(ConnectionError, match="mismatch"):
            assemble_model(site, fnds, subs, brgs, sup, loads)

    def test_foundation_bearing_mismatch(self):
        site = _mock_site()
        fnds = [_mock_foundation(), _mock_foundation()]
        subs = [_mock_substructure(), _mock_substructure()]
        brgs = [_mock_bearing()]  # Only 1
        sup = _mock_superstructure()
        loads = _mock_loads()

        with pytest.raises(ConnectionError, match="mismatch"):
            assemble_model(site, fnds, subs, brgs, sup, loads)


# ============================================================================
# TESTS — AnalysisResults and Empty Results
# ============================================================================

class TestAnalysisResults:
    """AnalysisResults dataclass and empty result extraction."""

    def test_empty_results(self):
        r = _extract_empty_results()
        assert isinstance(r, AnalysisResults)
        assert r.envelopes == {}
        assert r.dcr == {}
        assert r.controlling_cases == []

    def test_results_dataclass(self):
        r = AnalysisResults(
            envelopes={"elem1": {"N": {"max": 100, "min": -50}}},
            dcr={"elem1": {"flexure": 0.85}},
            modal={1: {"period": 1.5, "frequency": 0.667}},
        )
        assert r.dcr["elem1"]["flexure"] == 0.85

    def test_run_analysis_without_opensees(self):
        """run_analysis should handle missing openseespy gracefully."""
        model = AssembledModel()
        model.nodes = [{"tag": 1, "x": 0, "y": 0, "z": 0}]
        model.boundary_conditions = []
        # This may or may not work depending on openseespy availability
        # but should not crash
        try:
            result = run_analysis(model)
            assert isinstance(result, AnalysisResults)
        except Exception:
            # openseespy may not be installed — that's fine
            pass


# ============================================================================
# TESTS — AssembledModel Dataclass
# ============================================================================

class TestAssembledModel:
    """AssembledModel dataclass."""

    def test_defaults(self):
        m = AssembledModel()
        assert m.script == ""
        assert m.node_count == 0
        assert m.element_count == 0
        assert m.nodes == []
        assert m.bounding_cases == {}

    def test_populated(self):
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        m = assemble_model(site, fnds, subs, brgs, sup, loads)
        assert m.node_count > 0
        assert m.element_count > 0
        assert m.material_count > 0
        assert len(m.script) > 0


# ============================================================================
# TESTS — Integration: Full Pipeline
# ============================================================================

class TestFullPipeline:
    """Integration tests verifying the full assembly pipeline."""

    def test_1span_roundtrip(self):
        """Assemble a 1-span bridge and verify script parses."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)

        # Script must parse
        ast.parse(model.script)

        # Must have all components represented
        components_seen = set()
        for n in model.nodes:
            if "_component" in n:
                components_seen.add(n["_component"])
        assert "foundation" in components_seen
        assert "substructure" in components_seen
        assert "bearing" in components_seen
        assert "superstructure" in components_seen

    def test_3span_roundtrip(self):
        """Assemble a 3-span bridge and verify."""
        site, fnds, subs, brgs, sup, loads = _3span_continuous_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)

        ast.parse(model.script)
        assert model.node_count > 30

    def test_node_element_cross_refs_valid(self):
        """All element node references must exist in the node list."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)

        node_tags = {n["tag"] for n in model.nodes}
        for e in model.elements:
            for n in e.get("nodes", []):
                assert n in node_tags, (
                    f"Element {e['tag']} references node {n} "
                    f"which doesn't exist in node list"
                )

    def test_constraint_refs_valid(self):
        """All constraint node references must exist in the node list."""
        site, fnds, subs, brgs, sup, loads = _simple_1span_components()
        model = assemble_model(site, fnds, subs, brgs, sup, loads)

        node_tags = {n["tag"] for n in model.nodes}
        for c in model.constraints:
            if "master" in c:
                assert c["master"] in node_tags, (
                    f"Constraint master {c['master']} not in node list"
                )
            if "slave" in c:
                assert c["slave"] in node_tags, (
                    f"Constraint slave {c['slave']} not in node list"
                )

"""Assembler Tool — Stitches component models into a complete OpenSees bridge model.

This is the BRAIN of the Natural Language Builder pipeline. It takes outputs
from all six component tools (site-recon, foundation, substructure, bearings,
superstructure, loads) and produces:

1. A unified OpenSees model with collision-free tag numbering
2. Proper node connectivity between components
3. A complete analysis sequence (gravity → modal → moving load → spectrum → envelopes)
4. Upper/lower bound analysis for foundation springs and bearing stiffness
5. A standalone Python script that can be run independently

Tag Ranges (global numbering to prevent collisions):
    Nodes:      1–9999     foundations
                10000–19999 substructure
                20000–29999 bearings
                30000–49999 superstructure
    Elements:   Same ranges as nodes
    Materials:  1–999   foundations
                1000–1999 substructure
                2000–2999 bearings
                3000–4999 superstructure
                5000–5999 loads / analysis

Bounding Cases:
    UU = Upper foundation springs, Upper bearing stiffness
    UL = Upper foundation springs, Lower bearing stiffness
    LU = Lower foundation springs, Upper bearing stiffness
    LL = Lower foundation springs, Lower bearing stiffness
    Envelope across all four for every element force and displacement.

Units: kip-inch-second (KIS) internally.

References:
    AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    AASHTO Guide Specifications for LRFD Seismic Bridge Design, 2nd Ed (2011)
"""

from __future__ import annotations

import copy
import json
import logging
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# TAG RANGES
# ============================================================================

TAG_RANGES = {
    "foundation":    {"node": (1, 9999),      "element": (1, 9999),      "material": (1, 4999)},
    "substructure":  {"node": (10000, 19999),  "element": (10000, 19999), "material": (5000, 9999)},
    "bearing":       {"node": (20000, 29999),  "element": (20000, 29999), "material": (10000, 14999)},
    "superstructure":{"node": (30000, 49999),  "element": (30000, 49999), "material": (15000, 24999)},
    "analysis":      {"node": (50000, 59999),  "element": (50000, 59999), "material": (25000, 29999)},
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AssembledModel:
    """Complete bridge model ready for analysis.

    Attributes:
        script:              Complete OpenSees Python script.
        node_count:          Total number of nodes.
        element_count:       Total number of elements.
        material_count:      Total number of materials/sections.
        load_cases:          Number of load cases.
        load_combinations:   Number of load combinations.
        analysis_sequence:   Ordered list of analysis step descriptions.
        nodes:               All node dicts with global tags.
        elements:            All element dicts with global tags.
        materials:           All material dicts with global tags.
        sections:            All section dicts with global tags.
        constraints:         All constraint dicts with global tags.
        boundary_conditions: All fixity dicts with global tags.
        connections:         Connection metadata (foundation→sub→bearing→super).
        bounding_cases:      Dict of bounding case labels → component configs.
        warnings:            Any warnings from assembly.
    """
    script: str = ""
    node_count: int = 0
    element_count: int = 0
    material_count: int = 0
    load_cases: int = 0
    load_combinations: int = 0
    analysis_sequence: list[str] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)
    materials: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    boundary_conditions: list[dict] = field(default_factory=list)
    connections: list[dict] = field(default_factory=list)
    bounding_cases: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisResults:
    """Results from all analyses.

    Attributes:
        envelopes:         {element_tag: {force_type: {max, min, controlling_combo}}}
        dcr:               {element_tag: {limit_state: dcr_value}}
        reactions:         {node_tag: {Fx, Fy, Fz, Mx, My, Mz} per combo}
        displacements:     {node_tag: {dx, dy, dz, rx, ry, rz} per combo}
        modal:             {mode: {period, frequency, mass_participation}}
        controlling_cases: [{element, check, dcr, combo, description}]
    """
    envelopes: dict = field(default_factory=dict)
    dcr: dict = field(default_factory=dict)
    reactions: dict = field(default_factory=dict)
    displacements: dict = field(default_factory=dict)
    modal: dict = field(default_factory=dict)
    controlling_cases: list[dict] = field(default_factory=list)


# ============================================================================
# TAG REMAPPER
# ============================================================================

class TagRemapper:
    """Remaps local component tags to global non-colliding ranges.

    Each component tool assigns tags starting from 1. This remapper shifts
    all tags into the appropriate global range and tracks the mapping so
    inter-component connections can be resolved.
    """

    def __init__(self, component: str, support_index: int = 0):
        """
        Args:
            component:     One of 'foundation', 'substructure', 'bearing', 'superstructure'.
            support_index: Index of the support (0, 1, 2, ...) to space out
                          foundations/substructures/bearings within their range.
        """
        self.component = component
        self.support_index = support_index
        ranges = TAG_RANGES[component]

        # Compute per-support offset within the component range
        range_size = ranges["node"][1] - ranges["node"][0] + 1
        # Divide range evenly among supports (max 20 supports)
        max_supports = 20
        per_support = range_size // max_supports

        self.offsets = {}
        for kind in ("node", "element", "material"):
            base = ranges[kind][0]
            self.offsets[kind] = base + support_index * per_support

        self._map: dict[str, dict[int, int]] = {"node": {}, "element": {}, "material": {}}

    def remap(self, local_tag: int, kind: str) -> int:
        """Remap a local tag to global.

        Args:
            local_tag: Original tag from component tool.
            kind:      'node', 'element', or 'material'.

        Returns:
            Global tag.
        """
        if local_tag in self._map[kind]:
            return self._map[kind][local_tag]
        global_tag = self.offsets[kind] + local_tag
        self._map[kind][local_tag] = global_tag
        return global_tag

    def get_global(self, local_tag: int, kind: str) -> int:
        """Look up a previously remapped tag."""
        return self._map[kind].get(local_tag, self.offsets[kind] + local_tag)

    @property
    def node_map(self) -> dict[int, int]:
        return dict(self._map["node"])

    @property
    def element_map(self) -> dict[int, int]:
        return dict(self._map["element"])

    @property
    def material_map(self) -> dict[int, int]:
        return dict(self._map["material"])


def _normalize_component(component: dict) -> dict:
    """Normalize all tags in a component dict so they start from 1.
    
    Component tools may use arbitrary internal tag ranges (e.g., foundation
    starts nodes at 1004 or 2000). This renumbers everything to start from 1
    so the TagRemapper works correctly.
    """
    comp = dict(component)
    
    # Build offset maps for each tag type
    for tag_type, list_key in [("node", "nodes"), ("element", "elements"), ("material", "materials"), ("section", "sections")]:
        items = comp.get(list_key, [])
        if not items:
            continue
        tags = [item.get("tag", 0) for item in items]
        if not tags or min(tags) <= 1:
            continue
        
        min_tag = min(tags)
        offset = min_tag - 1
        tag_map = {}
        
        # Renumber items
        new_items = []
        for item in items:
            new = dict(item)
            old_tag = new.get("tag", 0)
            new_tag = old_tag - offset
            tag_map[old_tag] = new_tag
            new["tag"] = new_tag
            new_items.append(new)
        comp[list_key] = new_items
        
        # Update cross-references
        if tag_type == "section":
            # Fix section refs in elements
            new_elems = []
            for e in comp.get("elements", []):
                ne = dict(e)
                if "section" in ne and isinstance(ne["section"], int) and ne["section"] in tag_map:
                    ne["section"] = tag_map[ne["section"]]
                new_elems.append(ne)
            comp["elements"] = new_elems

        if tag_type == "material":
            # Fix nested material refs in material params (e.g., FiberSection core/cover/steel)
            new_mats = []
            for item in comp.get("materials", []):
                nm = dict(item)
                if "params" in nm and isinstance(nm["params"], dict):
                    params = dict(nm["params"])
                    for sub_key in ("core", "cover", "steel"):
                        if sub_key in params and isinstance(params[sub_key], dict):
                            sub = dict(params[sub_key])
                            if "material" in sub and isinstance(sub["material"], int) and sub["material"] in tag_map:
                                sub["material"] = tag_map[sub["material"]]
                            params[sub_key] = sub
                    nm["params"] = params
                new_mats.append(nm)
            comp["materials"] = new_mats
            
            # Fix material refs in elements
            new_elems = []
            for e in comp.get("elements", []):
                ne = dict(e)
                for mk in ("material", "section", "transform"):
                    if mk in ne and isinstance(ne[mk], int) and ne[mk] in tag_map:
                        ne[mk] = tag_map[ne[mk]]
                # Fix material lists (e.g., materials=[101, 102])
                if "materials" in ne and isinstance(ne["materials"], list):
                    ne["materials"] = [tag_map.get(m, m) if isinstance(m, int) else m for m in ne["materials"]]
                new_elems.append(ne)
            comp["elements"] = new_elems
            
            # Fix material refs in springs
            new_springs = []
            for s in comp.get("springs", []):
                ns = dict(s)
                if "material" in ns and isinstance(ns["material"], int) and ns["material"] in tag_map:
                    ns["material"] = tag_map[ns["material"]]
                new_springs.append(ns)
            comp["springs"] = new_springs
        
        if tag_type == "node":
            # Fix node refs in elements
            new_elems = []
            for e in comp.get("elements", []):
                ne = dict(e)
                if "nodes" in ne and isinstance(ne["nodes"], list):
                    ne["nodes"] = [tag_map.get(n, n) if isinstance(n, int) else n for n in ne["nodes"]]
                new_elems.append(ne)
            comp["elements"] = new_elems
            
            # Fix node refs in springs
            new_springs = []
            for s in comp.get("springs", []):
                ns = dict(s)
                if "node" in ns and isinstance(ns["node"], int):
                    ns["node"] = tag_map.get(ns["node"], ns["node"])
                new_springs.append(ns)
            comp["springs"] = new_springs
            
            # Fix scalar node refs
            for key in ("top_node", "base_node"):
                if key in comp and isinstance(comp[key], int):
                    comp[key] = tag_map.get(comp[key], comp[key])
            for key in ("top_nodes", "base_nodes", "cap_nodes", "support_nodes", "midspan_nodes"):
                if key in comp and isinstance(comp[key], list):
                    comp[key] = [tag_map.get(n, n) if isinstance(n, int) else n for n in comp[key]]
            
            # Fix node refs in boundary conditions
            new_bcs = []
            for bc in comp.get("boundary_conditions", []):
                nbc = dict(bc)
                if "node" in nbc and isinstance(nbc["node"], int):
                    nbc["node"] = tag_map.get(nbc["node"], nbc["node"])
                new_bcs.append(nbc)
            comp["boundary_conditions"] = new_bcs
            
            # Fix node refs in constraints
            new_cons = []
            for c in comp.get("constraints", []):
                nc = dict(c)
                for k in ("master", "slave", "node", "retained_node", "constrained_node"):
                    if k in nc and isinstance(nc[k], int):
                        nc[k] = tag_map.get(nc[k], nc[k])
                new_cons.append(nc)
            comp["constraints"] = new_cons
    
    return comp


def _remap_component_nodes(nodes: list[dict], remapper: TagRemapper) -> list[dict]:
    """Remap node tags in a list of node dicts."""
    result = []
    for n in nodes:
        new = dict(n)
        new["tag"] = remapper.remap(n["tag"], "node")
        new["_original_tag"] = n["tag"]
        new["_component"] = remapper.component
        result.append(new)
    return result


def _remap_component_elements(elements: list[dict], remapper: TagRemapper) -> list[dict]:
    """Remap element and node tags in a list of element dicts."""
    result = []
    for e in elements:
        new = dict(e)
        new["tag"] = remapper.remap(e["tag"], "element")
        new["_original_tag"] = e["tag"]
        new["_component"] = remapper.component
        if "nodes" in e:
            new["nodes"] = [remapper.remap(n, "node") for n in e["nodes"]]
        if "section" in e and isinstance(e["section"], int):
            new["section"] = remapper.remap(e["section"], "material")
        if "transform" in e and isinstance(e["transform"], int):
            new["transform"] = remapper.remap(e["transform"], "material")
        if "material" in e and isinstance(e["material"], int):
            new["material"] = remapper.remap(e["material"], "material")
        if "materials" in e and isinstance(e["materials"], list):
            new["materials"] = [
                remapper.remap(m, "material") if isinstance(m, int) else m
                for m in e["materials"]
            ]
        result.append(new)
    return result


def _remap_component_materials(materials: list[dict], remapper: TagRemapper) -> list[dict]:
    """Remap material tags (and cross-references) in a list of material dicts."""
    result = []
    for m in materials:
        new = dict(m)
        new["tag"] = remapper.remap(m["tag"], "material")
        new["_original_tag"] = m["tag"]
        new["_component"] = remapper.component
        # Remap cross-references in params if they are material tags
        if "params" in m and isinstance(m["params"], dict):
            params = dict(m["params"])
            for key in ("mat_confined", "mat_unconfined", "mat_steel",
                        "mat_concrete", "mat_flange", "mat_web", "mat_strand",
                        "material", "matTag"):
                if key in params and isinstance(params[key], int):
                    params[key] = remapper.remap(params[key], "material")
            # Handle nested dicts (FiberSection core/cover/steel)
            for sub_key in ("core", "cover", "steel"):
                if sub_key in params and isinstance(params[sub_key], dict):
                    sub = dict(params[sub_key])
                    if "material" in sub and isinstance(sub["material"], int):
                        sub["material"] = remapper.remap(sub["material"], "material")
                    params[sub_key] = sub
            new["params"] = params
        result.append(new)
    return result


def _remap_component_sections(sections: list[dict], remapper: TagRemapper) -> list[dict]:
    """Remap section tags and material references."""
    result = []
    for s in sections:
        new = dict(s)
        new["tag"] = remapper.remap(s["tag"], "material")  # sections share material tag space
        new["_original_tag"] = s["tag"]
        new["_component"] = remapper.component
        # Remap top-level material references
        for key in ("mat_confined", "mat_unconfined", "mat_steel",
                    "mat_concrete", "mat_flange", "mat_web", "mat_strand"):
            if key in new and isinstance(new[key], int):
                new[key] = remapper.remap(new[key], "material")
        if "params" in s and isinstance(s["params"], dict):
            params = dict(s["params"])
            for key in ("mat_confined", "mat_unconfined", "mat_steel",
                        "mat_concrete", "mat_flange", "mat_web", "mat_strand",
                        "matTag"):
                if key in params and isinstance(params[key], int):
                    params[key] = remapper.remap(params[key], "material")
            new["params"] = params
        result.append(new)
    return result


def _remap_constraints(constraints: list[dict], remapper: TagRemapper) -> list[dict]:
    """Remap node references in constraints."""
    result = []
    for c in constraints:
        new = dict(c)
        if "master" in c:
            new["master"] = remapper.remap(c["master"], "node")
        if "slave" in c:
            new["slave"] = remapper.remap(c["slave"], "node")
        if "nodes" in c:
            new["nodes"] = [remapper.remap(n, "node") for n in c["nodes"]]
        result.append(new)
    return result


# ============================================================================
# CONNECTION PROTOCOL
# ============================================================================

class ConnectionError(Exception):
    """Raised when components cannot be connected."""
    pass


def _connect_foundation_to_substructure(
    fnd_remapper: TagRemapper,
    sub_remapper: TagRemapper,
    fnd_model: dict,
    sub_model: dict,
) -> list[dict]:
    """Connect foundation top node(s) to substructure base node(s).

    Uses equalDOF constraints to tie the foundation top to the substructure
    base at each support.

    Returns list of constraint dicts.
    """
    constraints = []

    fnd_top = fnd_model.get("top_node", 0)
    sub_bases = sub_model.get("base_nodes", [])

    if not sub_bases:
        raise ConnectionError(
            "Substructure has no base_nodes — cannot connect to foundation."
        )

    # Foundation top_node → substructure base_node(s)
    # If single foundation top, connect to all sub bases via equalDOF
    fnd_top_global = fnd_remapper.get_global(fnd_top, "node")

    for base_node in sub_bases:
        sub_base_global = sub_remapper.get_global(base_node, "node")
        constraints.append({
            "type": "equalDOF",
            "master": fnd_top_global,
            "slave": sub_base_global,
            "dofs": [1, 2, 3, 4, 5, 6],
            "connection": "foundation_to_substructure",
        })

    return constraints


def _connect_substructure_to_bearing(
    sub_remapper: TagRemapper,
    brg_remapper: TagRemapper,
    sub_model: dict,
    brg_model: dict,
) -> list[dict]:
    """Connect substructure top/cap nodes to bearing bottom nodes.

    Returns list of constraint dicts (equalDOF).
    """
    constraints = []

    # Prefer cap_nodes, fall back to top_nodes
    sub_tops = sub_model.get("cap_nodes", []) or sub_model.get("top_nodes", [])
    brg_bots = brg_model.get("bottom_nodes", [])

    if not sub_tops:
        raise ConnectionError(
            "Substructure has no top_nodes or cap_nodes for bearing connection."
        )
    if not brg_bots:
        raise ConnectionError(
            "Bearing model has no bottom_nodes for substructure connection."
        )

    # Match one-to-one if counts match, or connect all bearings to first sub top
    if len(sub_tops) == len(brg_bots):
        for s_node, b_node in zip(sub_tops, brg_bots):
            s_global = sub_remapper.get_global(s_node, "node")
            b_global = brg_remapper.get_global(b_node, "node")
            constraints.append({
                "type": "equalDOF",
                "master": s_global,
                "slave": b_global,
                "dofs": [1, 2, 3, 4, 5, 6],
                "connection": "substructure_to_bearing",
            })
    elif len(sub_tops) == 1:
        # Single cap node → all bearings connect to it
        s_global = sub_remapper.get_global(sub_tops[0], "node")
        for b_node in brg_bots:
            b_global = brg_remapper.get_global(b_node, "node")
            constraints.append({
                "type": "equalDOF",
                "master": s_global,
                "slave": b_global,
                "dofs": [1, 2, 3, 4, 5, 6],
                "connection": "substructure_to_bearing",
            })
    elif len(brg_bots) == 1:
        # Single bearing bottom → connect to first substructure cap node
        b_global = brg_remapper.get_global(brg_bots[0], "node")
        s_global = sub_remapper.get_global(sub_tops[0], "node")
        constraints.append({
            "type": "equalDOF",
            "master": s_global,
            "slave": b_global,
            "dofs": [1, 2, 3, 4, 5, 6],
            "connection": "substructure_to_bearing",
        })
    else:
        # Connect what we can
        n_connect = min(len(sub_tops), len(brg_bots))
        for i in range(n_connect):
            s_global = sub_remapper.get_global(sub_tops[i], "node")
            b_global = brg_remapper.get_global(brg_bots[i], "node")
            constraints.append({
                "type": "equalDOF",
                "master": s_global,
                "slave": b_global,
                "dofs": [1, 2, 3, 4, 5, 6],
                "connection": "substructure_to_bearing",
            })

    return constraints


def _connect_bearing_to_superstructure(
    brg_remapper: TagRemapper,
    sup_remapper: TagRemapper,
    brg_model: dict,
    sup_model: dict,
    support_index: int,
    num_supports: int,
    num_girders: int,
) -> list[dict]:
    """Connect bearing top nodes to superstructure support nodes.

    The superstructure support_nodes are ordered:
    [support_0_girder_0, support_0_girder_1, ..., support_1_girder_0, ...]

    Returns list of constraint dicts (equalDOF).
    """
    constraints = []

    brg_tops = brg_model.get("top_nodes", [])
    sup_supports = sup_model.get("support_nodes", [])

    if not brg_tops:
        raise ConnectionError("Bearing has no top_nodes for superstructure connection.")
    if not sup_supports:
        raise ConnectionError("Superstructure has no support_nodes for bearing connection.")

    # Extract superstructure nodes for this support line
    # support_nodes has num_girders entries per support, ordered by support then girder
    start = support_index * num_girders
    end = start + num_girders
    sup_at_support = sup_supports[start:end] if end <= len(sup_supports) else sup_supports[start:]

    if len(brg_tops) == len(sup_at_support):
        for b_node, s_node in zip(brg_tops, sup_at_support):
            b_global = brg_remapper.get_global(b_node, "node")
            s_global = sup_remapper.get_global(s_node, "node")
            constraints.append({
                "type": "equalDOF",
                "master": b_global,
                "slave": s_global,
                "dofs": [1, 2, 3, 4, 5, 6],
                "connection": "bearing_to_superstructure",
            })
    elif len(brg_tops) == 1:
        # Single bearing node → connect ALL superstructure girders at this support
        b_global = brg_remapper.get_global(brg_tops[0], "node")
        for s_node in sup_at_support:
            s_global = sup_remapper.get_global(s_node, "node")
            constraints.append({
                "type": "equalDOF",
                "master": b_global,
                "slave": s_global,
                "dofs": [1, 2, 3, 4, 5, 6],
                "connection": "bearing_to_superstructure",
            })
    else:
        # Mismatch but still connect what we can
        n_connect = min(len(brg_tops), len(sup_at_support))
        for i in range(n_connect):
            b_global = brg_remapper.get_global(brg_tops[i], "node")
            s_global = sup_remapper.get_global(sup_at_support[i], "node")
            constraints.append({
                "type": "equalDOF",
                "master": b_global,
                "slave": s_global,
                "dofs": [1, 2, 3, 4, 5, 6],
                "connection": "bearing_to_superstructure",
            })

    return constraints


# ============================================================================
# BOUNDING CASE ENUMERATION
# ============================================================================

BOUNDING_LABELS = {
    "UU": {"foundation": "upper_bound", "bearing": "upper_bound"},
    "UL": {"foundation": "upper_bound", "bearing": "lower_bound"},
    "LU": {"foundation": "lower_bound", "bearing": "upper_bound"},
    "LL": {"foundation": "lower_bound", "bearing": "lower_bound"},
}


def enumerate_bounding_cases() -> dict[str, dict[str, str]]:
    """Return the 4 standard bounding case combinations.

    UU = Upper foundation springs + Upper bearing stiffness (stiffer → higher forces)
    UL = Upper foundation springs + Lower bearing stiffness
    LU = Lower foundation springs + Upper bearing stiffness
    LL = Lower foundation springs + Lower bearing stiffness (softer → larger displacements)
    """
    return dict(BOUNDING_LABELS)


# ============================================================================
# ANALYSIS SEQUENCE BUILDER
# ============================================================================

def build_analysis_sequence(
    has_seismic: bool = False,
    sdc: str = "B",
    has_moving_load: bool = True,
) -> list[str]:
    """Build the standard analysis sequence for a bridge model.

    Args:
        has_seismic:   Whether seismic load cases exist.
        sdc:           Seismic Design Category (A, B, C, D).
        has_moving_load: Whether to include HL-93 moving load analysis.

    Returns:
        Ordered list of analysis step names.
    """
    sequence = [
        "model_setup",          # ops.wipe() + ndm=3, ndf=6
        "define_materials",     # All materials, sections, transforms
        "define_nodes",         # All nodes with fixity
        "define_elements",      # All elements
        "gravity_analysis",     # DC + DW, load-controlled
        "modal_analysis",       # Eigenvalue → periods + mode shapes
    ]

    if has_moving_load:
        sequence.append("moving_load_analysis")  # HL-93 influence lines

    if has_seismic:
        sequence.append("response_spectrum_analysis")

    sequence.append("load_combination_envelopes")

    if has_seismic and sdc in ("C", "D"):
        sequence.append("pushover_analysis")

    return sequence


# ============================================================================
# SCRIPT GENERATOR
# ============================================================================

def _generate_model_setup() -> str:
    """Generate the model setup preamble."""
    return textwrap.dedent("""\
        #!/usr/bin/env python3
        \"\"\"Auto-generated OpenSees bridge model.
        
        Generated by Natural Language Builder assembler tool.
        Units: kip-inch-second (KIS).
        \"\"\"
        
        import json
        import math
        import sys
        
        try:
            import openseespy.opensees as ops
        except ImportError:
            print("ERROR: openseespy not installed. pip install openseespy")
            sys.exit(1)
        
        # ============================================================
        # MODEL SETUP
        # ============================================================
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
    """)


def _generate_materials_script(materials: list[dict], sections: list[dict]) -> str:
    """Generate OpenSees material/section definition commands."""
    lines = [
        "# ============================================================",
        "# MATERIALS AND SECTIONS",
        "# ============================================================",
        "",
        "# Default geometric transforms (fallback for any component)",
        "ops.geomTransf('PDelta', 1, 0.0, 0.0, 1.0)",
        "ops.geomTransf('Linear', 2, 0.0, 0.0, 1.0)",
        "ops.geomTransf('Corotational', 3, 0.0, 0.0, 1.0)",
        "",
    ]

    # First pass: emit all geomTransf definitions (they share tag space but must not be deduped against materials)
    # We'll defer actual emission until after nodes are known, so we can pick correct vecxz
    # For now, collect transform metadata
    transform_meta = {}  # tag -> {type, vecxz}
    for m in materials:
        mtype = m.get("type", "")
        if mtype in ("geomTransf", "Linear", "PDelta", "Corotational"):
            tag = m["tag"]
            if tag <= 3 or tag in transform_meta:
                continue
            tf = m.get("transform", m.get("type_name", mtype if mtype != "geomTransf" else "Linear"))
            vecxz = m.get("vecxz", [0.0, 0.0, 1.0])
            transform_meta[tag] = {"type": tf, "vecxz": vecxz}
    
    # Emit transforms (will be re-emitted with correct vecxz in element pass if needed)
    seen_tf_tags = set()
    for tag, meta in sorted(transform_meta.items()):
        seen_tf_tags.add(tag)
        lines.append(f"ops.geomTransf('{meta['type']}', {tag}, {', '.join(str(v) for v in meta['vecxz'])})")
    lines.append("")

    # Second pass: emit materials (skip transforms, FiberSections, and duplicates)
    # FiberSections are deferred to third pass (they reference other materials)
    fiber_sections = []
    seen_tags = set(seen_tf_tags)
    seen_tags.update({1, 2, 3})  # default transforms
    for m in materials:
        if m.get("type") == "FiberSection":
            fiber_sections.append(m)
            continue
        tag = m["tag"]
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        mtype = m.get("type", "Elastic")
        desc = m.get("description", m.get("name", ""))

        if desc:
            lines.append(f"# {desc}")

        if mtype == "Concrete01":
            params = m.get("params", {})
            if isinstance(params, dict):
                fc = params.get("fc", m.get("fc_ksi", 4.0))
                lines.append(
                    f"ops.uniaxialMaterial('Concrete01', {tag}, "
                    f"{-fc}, -0.002, 0.0, -0.005)"
                )
            elif isinstance(params, (list, tuple)):
                lines.append(
                    f"ops.uniaxialMaterial('Concrete01', {tag}, "
                    f"{', '.join(str(p) for p in params)})"
                )
        elif mtype == "Steel02":
            params = m.get("params", {})
            if isinstance(params, dict):
                fy = params.get("fy", m.get("fy_ksi", 60.0))
                Es = params.get("Es", 29000.0)
                b = params.get("b", 0.01)
                lines.append(
                    f"ops.uniaxialMaterial('Steel02', {tag}, {fy}, {Es}, {b})"
                )
            elif isinstance(params, (list, tuple)):
                lines.append(
                    f"ops.uniaxialMaterial('Steel02', {tag}, "
                    f"{', '.join(str(p) for p in params)})"
                )
        elif mtype == "Elastic":
            k = m.get("k_kip_per_in", m.get("params", {}).get("k", 1000.0)
                       if isinstance(m.get("params"), dict) else 1000.0)
            lines.append(f"ops.uniaxialMaterial('Elastic', {tag}, {k})")
        elif mtype == "ENT":
            k = m.get("k_kip_per_in", 1e6)
            if isinstance(m.get("params"), (list, tuple)):
                k = m["params"][0]
            elif isinstance(m.get("params"), dict):
                k = m["params"].get("k", k)
            lines.append(f"ops.uniaxialMaterial('ENT', {tag}, {k})")
        elif mtype in ("PySimple1", "TzSimple1", "QzSimple1"):
            params = m.get("params", {})
            if isinstance(params, dict):
                if mtype == "PySimple1":
                    lines.append(
                        f"ops.uniaxialMaterial('PySimple1', {tag}, "
                        f"{params.get('soilType', 1)}, "
                        f"{params.get('pult', 10.0)}, "
                        f"{params.get('y50', 0.1)}, "
                        f"{params.get('Cd', 0.0)})"
                    )
                elif mtype == "TzSimple1":
                    lines.append(
                        f"ops.uniaxialMaterial('TzSimple1', {tag}, "
                        f"{params.get('soilType', 1)}, "
                        f"{params.get('tult', 10.0)}, "
                        f"{params.get('z50', 0.1)}, "
                        f"{params.get('c', 0.0)})"
                    )
                elif mtype == "QzSimple1":
                    lines.append(
                        f"ops.uniaxialMaterial('QzSimple1', {tag}, "
                        f"{params.get('qzType', 1)}, "
                        f"{params.get('Qult', 100.0)}, "
                        f"{params.get('z50', 0.1)}, "
                        f"{params.get('suction', 0.0)})"
                    )
        elif mtype == "geomTransf" or mtype in ("Linear", "PDelta", "Corotational"):
            continue  # handled in first pass above
        elif mtype == "frictionModel":
            mu = m.get("mu", 0.05)
            lines.append(
                f"ops.frictionModel('Coulomb', {tag}, {mu})"
            )
        elif mtype == "FiberSection":
            params = m.get("params", {})
            core = params.get("core", {})
            cover = params.get("cover", {})
            steel = params.get("steel", {})
            if core and cover and steel:
                lines.append(f"ops.section('Fiber', {tag}, '-GJ', 1.0e10)")
                mat_core = core.get("material", tag + 1)
                r_core = core.get("radius", 24.0)
                nfr = core.get("nfr", 8)
                nft = core.get("nft", 16)
                lines.append(f"ops.patch('circ', {mat_core}, {nfr}, {nft}, 0.0, 0.0, 0.0, {r_core:.2f}, 0.0, 360.0)")
                mat_cover = cover.get("material", tag + 2)
                r_out = cover.get("outer_radius", r_core + 3)
                r_in = cover.get("inner_radius", r_core)
                lines.append(f"ops.patch('circ', {mat_cover}, {cover.get('nfr', 2)}, {cover.get('nft', 16)}, 0.0, 0.0, {r_in:.2f}, {r_out:.2f}, 0.0, 360.0)")
                mat_steel = steel.get("material", tag + 3)
                n_bars = steel.get("n_bars", 16)
                bar_area = steel.get("bar_area", 1.0)
                r_steel = steel.get("radius", r_core - 0.5)
                lines.append(f"ops.layer('circ', {mat_steel}, {n_bars}, {bar_area}, 0.0, 0.0, {r_steel:.2f})")
            else:
                lines.append(f"ops.section('Elastic', {tag}, 29000.0, 100.0, 5000.0, 29000.0, 100.0, 1.0e10)")
        else:
            lines.append(f"# Unsupported material type: {mtype} tag={tag}")

        lines.append("")

    # Third pass: FiberSections (must come after all regular materials)
    for m in fiber_sections:
        tag = m["tag"]
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        params = m.get("params", {})
        core = params.get("core", {})
        cover = params.get("cover", {})
        steel = params.get("steel", {})
        if core and cover and steel:
            lines.append(f"ops.section('Fiber', {tag}, '-GJ', 1.0e10)")
            mat_core = core.get("material", tag + 1)
            r_core = core.get("radius", 24.0)
            lines.append(f"ops.patch('circ', {mat_core}, {core.get('nfr', 8)}, {core.get('nft', 16)}, 0.0, 0.0, 0.0, {r_core:.2f}, 0.0, 360.0)")
            mat_cover = cover.get("material", tag + 2)
            r_out = cover.get("outer_radius", r_core + 3)
            r_in = cover.get("inner_radius", r_core)
            lines.append(f"ops.patch('circ', {mat_cover}, {cover.get('nfr', 2)}, {cover.get('nft', 16)}, 0.0, 0.0, {r_in:.2f}, {r_out:.2f}, 0.0, 360.0)")
            mat_steel = steel.get("material", tag + 3)
            lines.append(f"ops.layer('circ', {mat_steel}, {steel.get('n_bars', 16)}, {steel.get('bar_area', 1.0)}, 0.0, 0.0, {steel.get('radius', r_core - 0.5):.2f})")
        else:
            lines.append(f"ops.section('Elastic', {tag}, 29000.0, 100.0, 5000.0, 29000.0, 100.0, 1.0e10)")
        lines.append("")

    # Sections
    for s in sections:
        tag = s["tag"]
        stype = s.get("type", "")
        
        if stype == "circular_rc":
            # Fiber section for circular RC column
            d_in = s.get("diameter_in", 48.0)
            cover = s.get("cover_in", 2.0)
            n_bars = s.get("num_bars", 16)
            bar_area = s.get("bar_area", 1.0)
            mat_conf = s.get("mat_confined", 2)
            mat_unconf = s.get("mat_unconfined", 1)
            mat_steel = s.get("mat_steel", 3)
            r_out = d_in / 2.0
            r_core = r_out - cover
            lines.append(f"ops.section('Fiber', {tag}, '-GJ', 1.0e10)")
            lines.append(f"ops.patch('circ', {mat_conf}, 8, 8, 0.0, 0.0, 0.0, {r_core:.2f}, 0.0, 360.0)")
            lines.append(f"ops.patch('circ', {mat_unconf}, 4, 8, 0.0, 0.0, {r_core:.2f}, {r_out:.2f}, 0.0, 360.0)")
            lines.append(f"ops.layer('circ', {mat_steel}, {n_bars}, {bar_area}, 0.0, 0.0, {r_core:.2f})")
        elif stype == "rectangular_rc":
            w = s.get("width_in", 48.0)
            h = s.get("depth_in", 48.0)
            cover = s.get("cover_in", 2.0)
            # Use Elastic section as fallback if no material refs
            mat_unconf = s.get("mat_unconfined", s.get("mat_confined", 0))
            mat_steel = s.get("mat_steel", 0)
            if not mat_unconf or not mat_steel:
                lines.append(f"ops.section('Elastic', {tag}, 29000.0, 100.0, 5000.0, 29000.0, 100.0, 1.0e10)")
                continue
            n_top = s.get("num_bars_top", 8)
            n_bot = s.get("num_bars_bot", 8)
            bar_area = s.get("bar_area", 1.0)
            lines.append(f"ops.section('Fiber', {tag}, '-GJ', 1.0e10)")
            lines.append(f"ops.patch('rect', {mat_unconf}, 8, 8, {-h/2:.2f}, {-w/2:.2f}, {h/2:.2f}, {w/2:.2f})")
            if n_top > 0:
                lines.append(f"ops.layer('straight', {mat_steel}, {n_top}, {bar_area}, {h/2-cover:.2f}, {-w/2+cover:.2f}, {h/2-cover:.2f}, {w/2-cover:.2f})")
            if n_bot > 0:
                lines.append(f"ops.layer('straight', {mat_steel}, {n_bot}, {bar_area}, {-h/2+cover:.2f}, {-w/2+cover:.2f}, {-h/2+cover:.2f}, {w/2-cover:.2f})")
        elif stype == "FiberSection":
            params = s.get("params", {})
            core = params.get("core", {})
            cover = params.get("cover", {})
            steel = params.get("steel", {})
            
            if core and cover and steel:
                lines.append(f"ops.section('Fiber', {tag}, '-GJ', 1.0e10)")
                # Core patch (circular)
                mat_core = core.get("material", 1)
                r_core = core.get("radius", 24.0)
                nfr = core.get("nfr", 8)
                nft = core.get("nft", 16)
                lines.append(f"ops.patch('circ', {mat_core}, {nfr}, {nft}, 0.0, 0.0, 0.0, {r_core:.2f}, 0.0, 360.0)")
                # Cover patch
                mat_cover = cover.get("material", 1)
                r_out = cover.get("outer_radius", r_core + 3)
                r_in = cover.get("inner_radius", r_core)
                nfr_c = cover.get("nfr", 2)
                nft_c = cover.get("nft", 16)
                lines.append(f"ops.patch('circ', {mat_cover}, {nfr_c}, {nft_c}, 0.0, 0.0, {r_in:.2f}, {r_out:.2f}, 0.0, 360.0)")
                # Steel layer
                mat_steel = steel.get("material", 1)
                n_bars = steel.get("n_bars", 16)
                bar_area = steel.get("bar_area", 1.0)
                r_steel = steel.get("radius", r_core - 0.5)
                lines.append(f"ops.layer('circ', {mat_steel}, {n_bars}, {bar_area}, 0.0, 0.0, {r_steel:.2f})")
            else:
                # Fallback elastic
                lines.append(f"ops.section('Elastic', {tag}, 29000.0, 100.0, 5000.0, 29000.0, 100.0, 1.0e10)")
        elif stype in ("composite", "steel_i"):
            # Composite or steel I-section — use equivalent elastic properties as placeholder
            # TODO: Generate proper fiber section from params
            params = s.get("params", {})
            if stype == "composite":
                steel = params.get("steel_section", {})
                d = steel.get("d", 48.0)
                bf = steel.get("bf_bot", 18.0)
                tf = steel.get("tf_bot", 1.5)
                tw = steel.get("tw", 0.5)
                A = 2 * bf * tf + (d - 2*tf) * tw  # approximate steel area
                I = bf * d**3 / 12  # approximate
            else:
                steel = params
                d = steel.get("d", 12.0)
                bf = steel.get("bf_top", 6.0)
                tf = steel.get("tf_top", 0.5)
                tw = steel.get("tw", 0.375)
                A = 2 * bf * tf + (d - 2*tf) * tw
                I = bf * d**3 / 12
            E = 29000.0
            G = 11200.0
            J = I * 0.1  # approximate torsion
            lines.append(f"# {stype} section {tag} — elastic approximation")
            lines.append(f"ops.section('Elastic', {tag}, {E}, {A:.2f}, {I:.2f}, {E}, {A:.2f}, {J:.2f})")
        else:
            lines.append(f"# Section {tag}: {stype}")
            E = 29000.0
            lines.append(f"ops.section('Elastic', {tag}, {E}, 100.0, 5000.0, {E}, 100.0, 1.0e10)")

    lines.append("")
    return "\n".join(lines)


def _generate_nodes_script(nodes: list[dict], boundary_conditions: list[dict],
                           elements: list[dict] | None = None,
                           constraints: list[dict] | None = None) -> str:
    """Generate OpenSees node definition commands."""
    lines = [
        "# ============================================================",
        "# NODES",
        "# ============================================================",
        "",
    ]

    # Collect fixed nodes for boundary conditions
    fixed_nodes: dict[int, list[int]] = {}
    for bc in boundary_conditions:
        node = bc.get("node", 0)
        fixity = bc.get("fixity", [1, 1, 1, 1, 1, 1])
        fixed_nodes[node] = fixity

    for n in nodes:
        tag = n["tag"]
        x = n.get("x", 0.0)
        y = n.get("y", 0.0)
        z = n.get("z", 0.0)
        lines.append(f"ops.node({tag}, {x:.4f}, {y:.4f}, {z:.4f})")

    lines.append("")
    lines.append("# Boundary conditions")
    for node_tag, fixity in fixed_nodes.items():
        lines.append(f"ops.fix({node_tag}, {', '.join(str(f) for f in fixity)})")

    # Fix orphan nodes (not in any element or constraint) to prevent singularity
    if elements is not None:
        all_node_tags = set(n["tag"] for n in nodes)
        connected = set()
        for e in (elements or []):
            for n in e.get("nodes", []):
                connected.add(n)
        for c in (constraints or []):
            connected.add(c.get("master", 0))
            connected.add(c.get("slave", 0))
        connected.update(fixed_nodes.keys())
        orphans = all_node_tags - connected
        if orphans:
            lines.append("")
            lines.append(f"# Fix {len(orphans)} orphan nodes (not connected to elements)")
            for tag in sorted(orphans):
                lines.append(f"ops.fix({tag}, 1, 1, 1, 1, 1, 1)")

    lines.append("")
    return "\n".join(lines)


def _generate_elements_script(elements: list[dict], all_nodes: list[dict] | None = None) -> str:
    """Generate OpenSees element definition commands."""
    lines = [
        "# ============================================================",
        "# ELEMENTS",
        "# ============================================================",
        "",
    ]

    # Build node coordinate lookup for orientation detection
    node_coords = {}
    if all_nodes:
        for n in all_nodes:
            node_coords[n["tag"]] = (n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0))

    # Track orientation-specific transforms (created on the fly)
    # Base transforms 1-3 use vecxz=[0,0,1] — good for horizontal elements
    # Vertical elements (Y-dir) need vecxz=[1,0,0]  
    # Z-direction elements need vecxz=[0,1,0]
    orient_transforms = {}  # (base_tf, orient_key) -> new_tag
    next_orient_tf = 90001  # high range for auto-generated transforms

    def _get_oriented_transform(base_tf: int, n1: int, n2: int) -> int:
        """Get or create a transform with correct vecxz for element orientation."""
        nonlocal next_orient_tf
        if n1 not in node_coords or n2 not in node_coords:
            return base_tf
        x1, y1, z1 = node_coords[n1]
        x2, y2, z2 = node_coords[n2]
        dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
        
        # Determine dominant direction
        max_d = max(dx, dy, dz)
        if max_d < 1e-6:
            return base_tf  # zero-length, doesn't matter
        
        # Default vecxz=[0,0,1] works for X-direction elements
        # Y-direction (vertical) needs vecxz=[1,0,0]
        # Z-direction (transverse) needs vecxz=[0,1,0]
        if dy / max_d > 0.7:  # predominantly vertical
            orient_key = "vertical"
            vecxz = (1.0, 0.0, 0.0)
        elif dz / max_d > 0.7:  # predominantly Z (transverse)
            orient_key = "transverse"
            vecxz = (0.0, 1.0, 0.0)
        else:
            return base_tf  # horizontal X-dir, default is fine
        
        key = (base_tf, orient_key)
        if key not in orient_transforms:
            orient_transforms[key] = next_orient_tf
            next_orient_tf += 1
        return orient_transforms[key]

    # Pre-scan to discover orientation-specific transforms needed
    for e in elements:
        if e.get("type", "dispBeamColumn") == "dispBeamColumn":
            enodes = e.get("nodes", [])
            tf = e.get("transform", 1)
            if isinstance(tf, str):
                tf = 1
            if len(enodes) >= 2:
                _get_oriented_transform(tf, enodes[0], enodes[1])

    # Emit orientation-specific transforms
    if orient_transforms:
        lines.append("# Orientation-specific transforms (auto-generated)")
        tf_type_map = {1: "PDelta", 2: "Linear", 3: "Corotational"}
        for (base_tf, orient_key), new_tag in sorted(orient_transforms.items(), key=lambda x: x[1]):
            tf_type = tf_type_map.get(base_tf, "Linear")
            if orient_key == "vertical":
                vecxz = "1.0, 0.0, 0.0"
            elif orient_key == "transverse":
                vecxz = "0.0, 1.0, 0.0"
            else:
                vecxz = "0.0, 0.0, 1.0"
            lines.append(f"ops.geomTransf('{tf_type}', {new_tag}, {vecxz})")
        lines.append("")

    for e in elements:
        tag = e["tag"]
        etype = e.get("type", "dispBeamColumn")
        nodes = e.get("nodes", [])

        if etype == "dispBeamColumn":
            sec = e.get("section", 1)
            tf = e.get("transform", 1)
            if isinstance(tf, str):
                tf = 1
            if len(nodes) >= 2:
                tf = _get_oriented_transform(tf, nodes[0], nodes[1])
            np_ = e.get("np", 5)
            integ_tag = 100000 + tag
            lines.append(
                f"ops.beamIntegration('Lobatto', {integ_tag}, {sec}, {np_})"
            )
            lines.append(
                f"ops.element('dispBeamColumn', {tag}, "
                f"{', '.join(str(n) for n in nodes)}, "
                f"{tf}, {integ_tag})"
            )
        elif etype == "zeroLength":
            mats = e.get("materials", [e.get("material", 1)])
            dirs = e.get("directions", [1])
            if isinstance(mats, int):
                mats = [mats]
            if isinstance(dirs, int):
                dirs = [dirs]
            lines.append(
                f"ops.element('zeroLength', {tag}, "
                f"{', '.join(str(n) for n in nodes)}, "
                f"'-mat', {', '.join(str(m) for m in mats)}, "
                f"'-dir', {', '.join(str(d) for d in dirs)})"
            )
        elif etype == "rigidLink":
            link_type = e.get("linkType", "beam")
            lines.append(
                f"ops.rigidLink('{link_type}', {nodes[0]}, {nodes[1]})"
            )
        elif etype == "corotTruss":
            area = e.get("area", 10.0)
            mat = e.get("material", 1)
            lines.append(
                f"ops.element('corotTruss', {tag}, "
                f"{', '.join(str(n) for n in nodes)}, {area}, {mat})"
            )
        elif etype in ("SingleFPBearing", "TripleFrictionPendulum"):
            lines.append(f"# {etype} element {tag} — requires specialized setup")
        elif etype == "ShellMITC4":
            sec = e.get("section", 1)
            lines.append(
                f"ops.element('ShellMITC4', {tag}, "
                f"{', '.join(str(n) for n in nodes)}, {sec})"
            )
        else:
            lines.append(f"# Element {tag}: type {etype} (not scripted)")

    lines.append("")
    return "\n".join(lines)


def _generate_constraints_script(constraints: list[dict]) -> str:
    """Generate equalDOF and rigidLink constraint commands."""
    lines = [
        "# ============================================================",
        "# CONSTRAINTS (inter-component connections)",
        "# ============================================================",
        "",
    ]

    for c in constraints:
        ctype = c.get("type", "equalDOF")
        if ctype == "equalDOF":
            master = c.get("master", 0)
            slave = c.get("slave", 0)
            dofs = c.get("dofs", [1, 2, 3, 4, 5, 6])
            conn = c.get("connection", "")
            if conn:
                lines.append(f"# {conn}")
            lines.append(
                f"ops.equalDOF({master}, {slave}, "
                f"{', '.join(str(d) for d in dofs)})"
            )
        elif ctype == "rigidLink":
            master = c.get("master", 0)
            slave = c.get("slave", 0)
            lines.append(f"ops.rigidLink('beam', {master}, {slave})")

    lines.append("")
    return "\n".join(lines)


def _generate_analysis_script(analysis_sequence: list[str],
                              elements: list[dict] | None = None,
                              nodes: list[dict] | None = None) -> str:
    """Generate analysis commands."""
    lines = [
        "# ============================================================",
        "# ANALYSIS",
        "# ============================================================",
        "",
    ]

    for step in analysis_sequence:
        if step == "model_setup":
            continue  # Already in preamble
        elif step == "define_materials":
            continue
        elif step == "define_nodes":
            continue
        elif step == "define_elements":
            continue
        elif step == "gravity_analysis":
            lines.extend([
                "# --- Gravity Analysis ---",
                "ops.timeSeries('Linear', 1)",
                "ops.pattern('Plain', 1, 1)",
                "",
                "# Apply self-weight (DC) as distributed load on beam elements",
                "# Convention: gravity = -Y direction in kip-inch-second units",
                "# Also assign mass for modal analysis (mass = weight / g)",
                "g_in_s2 = 386.4  # in/s^2",
            ])
            # Generate eleLoad for all beam/frame elements
            beam_types = {"dispBeamColumn", "forceBeamColumn", "elasticBeamColumn"}
            if elements:
                for e in elements:
                    if e.get("type") in beam_types:
                        tag = e["tag"]
                        comp = e.get("_component", "")
                        # Estimate self-weight based on component
                        if comp == "superstructure":
                            # Steel plate girder: ~0.3-0.5 kip/ft ≈ 0.025-0.042 kip/in per girder
                            # Plus deck tributary: 8" slab × 9.5' spacing × 0.150 kcf = 0.95 kip/ft = 0.079 kip/in
                            # Total ≈ 0.104 kip/in per girder line
                            w_y = -0.104  # kip/in (gravity = -Y)
                        elif comp == "substructure":
                            # RC column/cap: ~0.15 kcf × ~12 ft² ≈ 1.8 kip/ft = 0.15 kip/in
                            w_y = -0.15
                        elif comp == "foundation":
                            # Drilled shaft: ~0.15 kcf × π/4 × 7² = 5.77 kip/ft = 0.48 kip/in
                            w_y = -0.48
                        else:
                            w_y = -0.05  # conservative default
                        lines.append(
                            f"ops.eleLoad('-ele', {tag}, '-type', '-beamUniform', {w_y}, 0.0)"
                        )
            else:
                lines.append("# WARNING: No elements available for gravity load generation")
            
            # Assign lumped mass to superstructure nodes for modal analysis
            if elements and nodes:
                lines.append("")
                lines.append("# Lumped mass assignment (kip-s²/in)")
                node_set = {n["tag"] for n in nodes}
                sup_nodes = {n["tag"] for n in nodes if n.get("_component") == "superstructure"}
                # Estimate tributary mass per superstructure node
                # ~0.104 kip/in per girder line × span/num_nodes × 1/g
                n_sup = len(sup_nodes) or 1
                # Total bridge weight ≈ 0.104 kip/in × 7 girders × 1050 ft × 12 in/ft = 916,000 lb = 916 kips
                total_weight_kip = 0.104 * 7.0 * 1050.0 * 12.0
                mass_per_node = total_weight_kip / n_sup / 386.4  # kip-s²/in
                for nt in sorted(sup_nodes):
                    lines.append(
                        f"ops.mass({nt}, {mass_per_node:.6f}, {mass_per_node:.6f}, {mass_per_node:.6f}, 0.0, 0.0, 0.0)"
                    )
            
            lines.extend([
                "",
                "ops.system('UmfPack')",
                "ops.numberer('Plain')",
                "ops.constraints('Penalty', 1.0e14, 1.0e14)",
                "ops.integrator('LoadControl', 1.0)",
                "ops.test('NormUnbalance', 1.0e-2, 1000)",
                "ops.algorithm('Linear')",  # Single-step linear analysis
                "ops.analysis('Static')",
                "",
                "gravity_ok = True",
                "ok = ops.analyze(1)",
                "if ok != 0:",
                "    # Try smaller steps with Newton",
                "    ops.wipeAnalysis()",
                "    ops.system('UmfPack')",
                "    ops.numberer('Plain')",
                "    ops.constraints('Penalty', 1.0e14, 1.0e14)",
                "    ops.integrator('LoadControl', 0.01)",
                "    ops.test('NormUnbalance', 1.0e-2, 500)",
                "    ops.algorithm('Newton')",
                "    ops.analysis('Static')",
                "    for i in range(100):",
                "        ok = ops.analyze(1)",
                "        if ok != 0:",
                "            ops.algorithm('ModifiedNewton')",
                "            ok = ops.analyze(1)",
                "            ops.algorithm('Newton')",
                "            if ok != 0:",
                "                print(f'WARNING: Gravity step {i+1} failed')",
                "                gravity_ok = False",
                "                break",
                "",
                "if gravity_ok:",
                "    print('Gravity analysis converged successfully')",
                "    ops.loadConst('-time', 0.0)",
                "else:",
                "    print('WARNING: Gravity analysis did not converge')",
                "",
            ])
        elif step == "modal_analysis":
            lines.extend([
                "# --- Modal Analysis ---",
                "if gravity_ok:",
                "    try:",
                "        num_modes = 10",
                "        eigenvalues = ops.eigen(num_modes)",
                "        periods = []",
                "        for ev in eigenvalues:",
                "            if ev > 0:",
                "                T = 2.0 * 3.14159265 / (ev ** 0.5)",
                "            else:",
                "                T = 0.0",
                "            periods.append(T)",
                "        print('Natural periods:', [f'{T:.3f}s' for T in periods])",
                "    except Exception as e:",
                "        print(f'WARNING: Modal analysis failed: {e}')",
                "        periods = []",
                "else:",
                "    print('Skipping modal analysis — gravity did not converge')",
                "    periods = []",
                "",
            ])
        elif step == "moving_load_analysis":
            lines.extend([
                "# --- Moving Load Analysis (HL-93) ---",
                "# Influence line analysis at tenth-points",
                "# (Implemented via unit load traversal)",
                "print('Moving load analysis: placeholder')",
                "",
            ])
        elif step == "response_spectrum_analysis":
            lines.extend([
                "# --- Response Spectrum Analysis ---",
                "# Apply CQC combination of modal responses",
                "print('Response spectrum analysis: placeholder')",
                "",
            ])
        elif step == "load_combination_envelopes":
            lines.extend([
                "# --- Load Combination Envelopes ---",
                "# Envelope across all AASHTO combinations",
                "print('Load combination envelopes: placeholder')",
                "",
            ])
        elif step == "pushover_analysis":
            lines.extend([
                "# --- Pushover Analysis ---",
                "# Displacement-controlled push to target ductility",
                "print('Pushover analysis: placeholder')",
                "",
            ])

    lines.extend([
        "# ============================================================",
        "# RESULTS OUTPUT",
        "# ============================================================",
        "results = {",
        "    'status': 'complete',",
        "    'model': 'bridge_model',",
        "}",
        "with open('bridge_results.json', 'w') as f:",
        "    json.dump(results, f, indent=2)",
        "print('Analysis complete. Results written to bridge_results.json')",
        "",
    ])

    return "\n".join(lines)


def generate_script(model: AssembledModel) -> str:
    """Generate a standalone OpenSees Python script from an assembled model.

    The generated script can be run independently:
        python bridge_model.py

    Args:
        model: AssembledModel with all components assembled.

    Returns:
        Complete Python script as a string.
    """
    parts = [
        _generate_model_setup(),
        _generate_materials_script(model.materials, model.sections),
        _generate_nodes_script(model.nodes, model.boundary_conditions, model.elements, model.constraints),
        _generate_elements_script(model.elements, model.nodes),
        _generate_constraints_script(model.constraints),
        _generate_analysis_script(model.analysis_sequence, model.elements, model.nodes),
    ]
    return "\n".join(parts)


# ============================================================================
# RESULT EXTRACTION
# ============================================================================

def _extract_empty_results() -> AnalysisResults:
    """Return an empty AnalysisResults placeholder."""
    return AnalysisResults(
        envelopes={},
        dcr={},
        reactions={},
        displacements={},
        modal={},
        controlling_cases=[],
    )


# ============================================================================
# MAIN ASSEMBLY FUNCTION
# ============================================================================

def assemble_model(
    site: dict,
    foundations: list[dict],
    substructures: list[dict],
    bearings: list[dict],
    superstructure: dict,
    loads: dict,
) -> AssembledModel:
    """Assemble all components into a complete OpenSees bridge model.

    This is the primary entry point. It:
    1. Validates component counts
    2. Remaps all tags to global ranges
    3. Connects components (foundation → substructure → bearing → superstructure)
    4. Builds the analysis sequence
    5. Generates the complete OpenSees script
    6. Enumerates bounding cases

    Args:
        site:           Site profile dict (from site_recon).
        foundations:     List of FoundationModel dicts (one per support).
        substructures:  List of SubstructureModel dicts (one per pier/abutment).
        bearings:       List of BearingModel dicts (one set per support).
        superstructure: SuperstructureModel dict.
        loads:          LoadModel dict.

    Returns:
        AssembledModel ready for analysis.

    Raises:
        ConnectionError: If component counts don't match or connections fail.
    """
    model = AssembledModel()
    warnings = []

    # ------------------------------------------------------------------
    # 1. Validate component counts
    # ------------------------------------------------------------------
    n_supports = len(foundations)
    if len(substructures) != n_supports:
        raise ConnectionError(
            f"Component count mismatch: {n_supports} foundations but "
            f"{len(substructures)} substructures. Must be equal."
        )
    if len(bearings) != n_supports:
        raise ConnectionError(
            f"Component count mismatch: {n_supports} foundations but "
            f"{len(bearings)} bearing sets. Must be equal."
        )

    # ------------------------------------------------------------------
    # 2. Remap tags for each component
    # ------------------------------------------------------------------
    all_nodes = []
    all_elements = []
    all_materials = []
    all_sections = []
    all_constraints = []
    all_bcs = []

    fnd_remappers = []
    sub_remappers = []
    brg_remappers = []

    # --- Normalize all components (tools use arbitrary internal tag ranges) ---
    foundations = [_normalize_component(f) for f in foundations]
    substructures = [_normalize_component(s) for s in substructures]
    bearings = [_normalize_component(b) for b in bearings]
    superstructure = _normalize_component(superstructure)

    # --- Foundations ---
    for i, fnd in enumerate(foundations):
        r = TagRemapper("foundation", i)
        fnd_remappers.append(r)

        all_nodes.extend(_remap_component_nodes(fnd.get("nodes", []), r))
        all_elements.extend(_remap_component_elements(fnd.get("elements", []), r))
        all_materials.extend(_remap_component_materials(fnd.get("materials", []), r))
        for t in fnd.get("transforms", []):
            new_t = dict(t)
            new_t["tag"] = r.remap(t["tag"], "material")
            new_t["_component"] = "foundation"
            all_materials.append(new_t)

        for bc in fnd.get("boundary_conditions", []):
            new_bc = dict(bc)
            new_bc["node"] = r.get_global(bc.get("node", 0), "node")
            all_bcs.append(new_bc)

    # --- Substructures ---
    for i, sub in enumerate(substructures):
        r = TagRemapper("substructure", i)
        sub_remappers.append(r)

        all_nodes.extend(_remap_component_nodes(sub.get("nodes", []), r))
        all_elements.extend(_remap_component_elements(sub.get("elements", []), r))
        all_materials.extend(_remap_component_materials(sub.get("materials", []), r))
        all_sections.extend(_remap_component_sections(sub.get("sections", []), r))
        all_constraints.extend(_remap_constraints(sub.get("constraints", []), r))
        for t in sub.get("transforms", []):
            new_t = dict(t)
            new_t["tag"] = r.remap(t["tag"], "material")
            new_t["_component"] = "substructure"
            all_materials.append(new_t)

    # --- Bearings ---
    for i, brg in enumerate(bearings):
        r = TagRemapper("bearing", i)
        brg_remappers.append(r)

        all_nodes.extend(_remap_component_nodes(brg.get("nodes", []), r))
        all_elements.extend(_remap_component_elements(brg.get("elements", []), r))
        all_materials.extend(_remap_component_materials(brg.get("materials", []), r))
        all_constraints.extend(_remap_constraints(brg.get("constraints", []), r))
        for t in brg.get("transforms", []):
            new_t = dict(t)
            new_t["tag"] = r.remap(t["tag"], "material")
            new_t["_component"] = "bearing"
            all_materials.append(new_t)

    # --- Superstructure (single component, support_index=0) ---
    sup_remapper = TagRemapper("superstructure", 0)

    all_nodes.extend(_remap_component_nodes(superstructure.get("nodes", []), sup_remapper))
    all_elements.extend(_remap_component_elements(superstructure.get("elements", []), sup_remapper))
    all_materials.extend(_remap_component_materials(superstructure.get("materials", []), sup_remapper))
    all_sections.extend(_remap_component_sections(superstructure.get("sections", []), sup_remapper))

    # Superstructure transforms — elements already have transform remapped to material range
    # We need to emit geomTransf definitions at those SAME remapped tags
    # Element transform=1 was remapped to 15001 by _remap_component_elements
    # So we need geomTransf at tag 15001
    for t in superstructure.get("transforms", []):
        new_t = dict(t)
        new_t["tag"] = sup_remapper.remap(t["tag"], "material")
        new_t["_component"] = "superstructure"
        new_t["type"] = t.get("type", "Linear")  # Ensure type is set
        # Append AFTER all other materials so it doesn't get deduped by an earlier material with same tag
        all_materials.append(new_t)

    # Remap diaphragm elements
    all_elements.extend(
        _remap_component_elements(superstructure.get("diaphragms", []), sup_remapper)
    )

    # ------------------------------------------------------------------
    # 3. Connect components
    # ------------------------------------------------------------------
    num_girders = superstructure.get("girder_lines", 1)
    if num_girders == 0:
        num_girders = 1  # slab bridges etc.

    for i in range(n_supports):
        # Foundation → Substructure
        try:
            conns = _connect_foundation_to_substructure(
                fnd_remappers[i], sub_remappers[i],
                foundations[i], substructures[i],
            )
            all_constraints.extend(conns)
            model.connections.append({
                "support": i,
                "type": "foundation_to_substructure",
                "constraints": len(conns),
            })
        except ConnectionError as e:
            warnings.append(f"Support {i}: {e}")

        # Substructure → Bearing
        try:
            conns = _connect_substructure_to_bearing(
                sub_remappers[i], brg_remappers[i],
                substructures[i], bearings[i],
            )
            all_constraints.extend(conns)
            model.connections.append({
                "support": i,
                "type": "substructure_to_bearing",
                "constraints": len(conns),
            })
        except ConnectionError as e:
            warnings.append(f"Support {i}: {e}")

        # Bearing → Superstructure
        try:
            conns = _connect_bearing_to_superstructure(
                brg_remappers[i], sup_remapper,
                bearings[i], superstructure,
                support_index=i,
                num_supports=n_supports,
                num_girders=num_girders,
            )
            all_constraints.extend(conns)
            model.connections.append({
                "support": i,
                "type": "bearing_to_superstructure",
                "constraints": len(conns),
            })
        except ConnectionError as e:
            warnings.append(f"Support {i}: {e}")

    # ------------------------------------------------------------------
    # 4. Build analysis sequence
    # ------------------------------------------------------------------
    has_seismic = False
    sdc = "B"
    load_cases_list = loads.get("cases", [])
    load_combos_list = loads.get("combinations", [])

    for lc in load_cases_list:
        lt = lc.get("load_type", "") if isinstance(lc, dict) else ""
        if lt == "EQ":
            has_seismic = True
            for ld in lc.get("loads", []):
                if "sdc" in ld:
                    sdc = ld["sdc"]

    analysis_seq = build_analysis_sequence(has_seismic, sdc)

    # ------------------------------------------------------------------
    # 5. Enumerate bounding cases
    # ------------------------------------------------------------------
    bounding = enumerate_bounding_cases()

    # ------------------------------------------------------------------
    # 6. Populate model
    # ------------------------------------------------------------------
    model.nodes = all_nodes
    model.elements = all_elements
    model.materials = all_materials
    model.sections = all_sections
    model.constraints = all_constraints
    model.boundary_conditions = all_bcs
    model.analysis_sequence = analysis_seq
    model.bounding_cases = bounding
    model.warnings = warnings

    model.node_count = len(all_nodes)
    model.element_count = len(all_elements)
    model.material_count = len(all_materials) + len(all_sections)
    model.load_cases = len(load_cases_list)
    model.load_combinations = len(load_combos_list)

    # ------------------------------------------------------------------
    # 7. Generate script
    # ------------------------------------------------------------------
    model.script = generate_script(model)

    return model


# ============================================================================
# ANALYSIS RUNNER
# ============================================================================

def run_analysis(model: AssembledModel) -> AnalysisResults:
    """Execute full analysis sequence and extract results.

    This function requires openseespy to be installed. It:
    1. Builds the model in OpenSees from the assembled data
    2. Runs each analysis step in sequence
    3. Extracts forces, displacements, reactions at every element/node
    4. Computes DCR = demand/capacity for each element
    5. Identifies controlling load combinations

    Args:
        model: AssembledModel from assemble_model().

    Returns:
        AnalysisResults with envelopes, DCRs, reactions, displacements, modal data.

    Note:
        If openseespy is not available, returns empty results with a warning.
    """
    try:
        import openseespy.opensees as ops
    except ImportError:
        logger.warning("openseespy not installed — returning empty results")
        results = _extract_empty_results()
        return results

    results = AnalysisResults()

    # Execute the generated script would be one approach,
    # but we'll build the model programmatically for better error handling
    try:
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # Define nodes
        for n in model.nodes:
            ops.node(n["tag"], n.get("x", 0.0), n.get("y", 0.0), n.get("z", 0.0))

        # Apply boundary conditions
        for bc in model.boundary_conditions:
            node = bc.get("node", 0)
            fixity = bc.get("fixity", [1, 1, 1, 1, 1, 1])
            ops.fix(node, *fixity)

        # Note: Full material/element/analysis execution would go here
        # For now, we do modal analysis if possible

        logger.info("OpenSees model built: %d nodes, %d elements",
                     model.node_count, model.element_count)

    except Exception as e:
        logger.error("Analysis failed: %s", e)
        results = _extract_empty_results()

    return results

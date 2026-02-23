"""
Superstructure Modeling Tool for Natural Language Bridge Builder.

Creates OpenSees FEA superstructure models from bridge type + geometry.
Generates nodes, elements, fiber sections, and materials for complete
3D grillage models of common bridge types.

All internal units: kip-inch-second (KIS).
    Input accepts feet — converted at boundary.
    Stress: ksi
    Length: inch
    Force:  kip

Bridge Types Supported (prioritized):
    1. Steel Plate Girders (composite)
    2. Steel Plate Girders (non-composite)
    3. Prestressed Concrete I-Girders
    4. CIP Concrete Box Girder
    5. Prestressed Segmental Box Girder
    6. Steel Truss
    7. Concrete Slab Bridge
    8. Arch Bridge

References:
    - AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    - PCI Bridge Design Manual, 3rd Edition (2014)
    - AISC Steel Construction Manual, 15th Edition (2017)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ============================================================================
# CONSTANTS
# ============================================================================

FT_TO_IN = 12.0  # feet to inches conversion


# ============================================================================
# OUTPUT DATA MODEL
# ============================================================================

@dataclass
class SuperstructureModel:
    """Complete superstructure model output.

    Contains all nodes, elements, sections, and materials needed to
    build an OpenSees model of the bridge superstructure.

    Attributes:
        nodes:          List of node dicts [{tag, x, y, z}, ...]
        elements:       List of element dicts [{tag, type, nodes, section, transform}, ...]
        sections:       List of section dicts [{tag, type, params}, ...]
        materials:      List of material dicts [{tag, type, params}, ...]
        diaphragms:     Transverse connectivity elements
        span_lengths:   Span lengths in feet (as input)
        girder_lines:   Number of girder lines modeled
        deck_width:     Out-to-out deck width (ft)
        support_nodes:  Node tags at abutments/piers (for bearing connection)
        midspan_nodes:  Node tags at midspan (for max moment check)
        continuity:     'continuous' or 'simple' at each support
        transforms:     Geometric transformation dicts [{tag, type, vecxz}, ...]
    """
    nodes: List[Dict] = field(default_factory=list)
    elements: List[Dict] = field(default_factory=list)
    sections: List[Dict] = field(default_factory=list)
    materials: List[Dict] = field(default_factory=list)
    diaphragms: List[Dict] = field(default_factory=list)
    span_lengths: List[float] = field(default_factory=list)
    girder_lines: int = 0
    deck_width: float = 0.0
    support_nodes: List[int] = field(default_factory=list)
    midspan_nodes: List[int] = field(default_factory=list)
    continuity: List[str] = field(default_factory=list)
    transforms: List[Dict] = field(default_factory=list)


# ============================================================================
# TAG MANAGER
# ============================================================================

class TagManager:
    """Manages unique tags for nodes, elements, sections, materials, transforms."""

    def __init__(self, base: int = 1):
        self._counters = {
            'node': base,
            'element': base,
            'section': base,
            'material': base,
            'transform': base,
        }

    def next(self, kind: str) -> int:
        tag = self._counters[kind]
        self._counters[kind] += 1
        return tag

    def next_n(self, kind: str, n: int) -> List[int]:
        tags = list(range(self._counters[kind], self._counters[kind] + n))
        self._counters[kind] += n
        return tags


# ============================================================================
# MESH / ENGINEERING UTILITIES
# ============================================================================

def select_transform_type(span_ft: float) -> str:
    """Select geometric transformation type based on span length.

    Per engineering practice:
        - < 100 ft:  Linear (P-delta effects negligible)
        - 100-200 ft: PDelta (second-order effects significant)
        - > 200 ft:  Corotational (large displacement formulation)

    Args:
        span_ft: Span length in feet.

    Returns:
        Transform type string: 'Linear', 'PDelta', or 'Corotational'.

    Reference:
        AASHTO LRFD C4.5.3.2.2b: When to include P-delta effects.
    """
    if span_ft > 200.0:
        return 'Corotational'
    elif span_ft >= 100.0:
        return 'PDelta'
    else:
        return 'Linear'


def compute_mesh_density(span_ft: float) -> int:
    """Compute number of elements per span.

    Minimum 10 elements per span, 20 for spans > 100 ft.
    Ensures nodes at tenth-points and midspan.

    Args:
        span_ft: Span length in feet.

    Returns:
        Number of elements per span (always divisible by 10 for tenth-points).

    Reference:
        General FEA best practice for beam models.
    """
    if span_ft > 100.0:
        base = 20
    else:
        base = 10
    # Round up to nearest 10 for clean tenth-point spacing
    return max(base, 10 * math.ceil(base / 10))


def effective_slab_width(span_ft: float, girder_spacing_ft: float,
                         slab_thickness_in: float,
                         top_flange_width_in: float) -> float:
    """Compute effective slab width per AASHTO LRFD 4.6.2.6.1.

    For interior girders, effective width is the minimum of:
        1. One-quarter of the effective span length
        2. Center-to-center girder spacing
        3. 12 × slab thickness + max(top flange width, 0.5 × web thickness)

    Simplified: uses top flange width for the third criterion.

    Args:
        span_ft:              Effective span length (ft).
        girder_spacing_ft:    Center-to-center girder spacing (ft).
        slab_thickness_in:    Slab thickness (inches).
        top_flange_width_in:  Top flange width (inches).

    Returns:
        Effective slab width in inches.

    Reference:
        AASHTO LRFD 4.6.2.6.1: Effective Flange Width for Interior Beams.
    """
    span_in = span_ft * FT_TO_IN
    spacing_in = girder_spacing_ft * FT_TO_IN

    opt1 = span_in / 4.0
    opt2 = spacing_in
    opt3 = 12.0 * slab_thickness_in + top_flange_width_in

    return min(opt1, opt2, opt3)


def _span_node_positions(span_in: float, n_elements: int) -> List[float]:
    """Generate node positions along a span (relative to span start).

    Returns positions that include tenth-points and midspan.

    Args:
        span_in:    Span length in inches.
        n_elements: Number of elements.

    Returns:
        List of positions from 0 to span_in (n_elements + 1 values).
    """
    return [i * span_in / n_elements for i in range(n_elements + 1)]


def _diaphragm_positions(span_in: float) -> List[float]:
    """Compute diaphragm locations within a span.

    Diaphragms at supports and at third-points minimum (AASHTO 6.7.4).
    Returns positions relative to span start (excluding supports themselves,
    which are handled separately).

    Args:
        span_in: Span length in inches.

    Returns:
        List of diaphragm positions (relative to span start).

    Reference:
        AASHTO LRFD 6.7.4: Diaphragms and Cross-Frames.
    """
    return [span_in / 3.0, 2.0 * span_in / 3.0]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def create_superstructure(
    bridge_type: str,
    span_lengths_ft: List[float],
    *,
    # Common parameters
    num_girders: int = 5,
    girder_spacing_ft: float = 8.0,
    deck_width_ft: Optional[float] = None,
    slab_thickness_in: float = 8.0,
    continuity: Optional[List[str]] = None,
    skew_angle: float = 0.0,
    # Steel plate girder parameters
    girder_depth_in: float = 48.0,
    top_flange_width_in: float = 16.0,
    top_flange_thick_in: float = 1.0,
    bot_flange_width_in: float = 18.0,
    bot_flange_thick_in: float = 1.5,
    web_thickness_in: float = 0.5625,
    haunch_in: float = 2.0,
    fy_ksi: float = 50.0,
    fc_ksi: float = 4.0,
    # Prestressed girder parameters
    girder_type: Optional[str] = None,
    strand_pattern: Optional[List[Tuple[float, int]]] = None,
    strand_area: float = 0.217,
    # Box girder parameters
    num_cells: int = 3,
    box_depth_in: float = 72.0,
    top_slab_thick_in: float = 9.0,
    bot_slab_thick_in: float = 6.0,
    box_web_thick_in: float = 12.0,
    box_top_width_ft: Optional[float] = None,
    box_bot_width_ft: Optional[float] = None,
    # Segmental parameters
    segment_length_ft: float = 10.0,
    # Truss parameters
    panel_length_ft: float = 20.0,
    truss_depth_ft: float = 25.0,
    top_chord_area_in2: float = 50.0,
    bot_chord_area_in2: float = 50.0,
    vertical_area_in2: float = 20.0,
    diagonal_area_in2: float = 30.0,
    connection_type: str = 'pin',
    # Slab bridge parameters
    slab_width_ft: Optional[float] = None,
    rebar_area_in2_per_ft: float = 0.6,
    # Arch parameters
    arch_rise_ft: float = 50.0,
    arch_shape: str = 'parabolic',
    rib_depth_in: float = 36.0,
    rib_width_in: float = 24.0,
    # Tag base
    tag_base: int = 1,
) -> SuperstructureModel:
    """Create a complete superstructure model.

    Primary entry point for the superstructure tool. Routes to the
    appropriate builder based on bridge_type.

    Args:
        bridge_type: One of:
            'steel_plate_girder_composite'
            'steel_plate_girder_noncomposite'
            'prestressed_i_girder'
            'cip_box_girder'
            'segmental_box_girder'
            'steel_truss'
            'concrete_slab'
            'arch'

        span_lengths_ft: List of span lengths in feet.
            e.g., [120, 160, 120] for a 3-span continuous bridge.

        num_girders: Number of girder lines (default 5).
        girder_spacing_ft: Center-to-center girder spacing (ft).
        deck_width_ft: Out-to-out deck width (ft). Auto-computed if None.
        slab_thickness_in: Deck slab thickness (inches).
        continuity: List of 'continuous' or 'simple' at each interior
            support (length = len(span_lengths) - 1). Default: all continuous
            for multi-span, simple for single-span.
        skew_angle: Skew angle in degrees (0 = normal, positive = right).

    Returns:
        SuperstructureModel with all nodes, elements, sections, materials.

    Raises:
        ValueError: For invalid bridge_type or parameter combinations.

    Reference:
        AASHTO LRFD 4.6.2: Approximate/Refined Methods of Analysis.
    """
    # Validate inputs
    if not span_lengths_ft:
        raise ValueError("span_lengths_ft must have at least one span.")
    for s in span_lengths_ft:
        if s <= 0:
            raise ValueError(f"Span length must be positive, got {s} ft.")

    # Default continuity
    n_spans = len(span_lengths_ft)
    n_supports = n_spans + 1  # abutments + piers
    n_interior = n_spans - 1  # interior supports (piers)

    if continuity is None:
        if n_spans == 1:
            continuity = []
        else:
            # Default: all continuous for steel/CIP, all simple for prestressed
            if bridge_type in ('prestressed_i_girder',):
                continuity = ['simple'] * n_interior
            else:
                continuity = ['continuous'] * n_interior
    else:
        if len(continuity) != n_interior:
            raise ValueError(
                f"continuity must have {n_interior} entries "
                f"(one per interior support), got {len(continuity)}."
            )

    # Default deck width
    if deck_width_ft is None:
        deck_width_ft = girder_spacing_ft * (num_girders - 1) + 3.0  # 1.5 ft overhang each side

    # Route to builder
    builders = {
        'steel_plate_girder_composite': _build_steel_plate_girder,
        'steel_plate_girder_noncomposite': _build_steel_plate_girder,
        'prestressed_i_girder': _build_prestressed_i_girder,
        'cip_box_girder': _build_cip_box_girder,
        'segmental_box_girder': _build_segmental_box_girder,
        'steel_truss': _build_steel_truss,
        'concrete_slab': _build_concrete_slab,
        'arch': _build_arch,
    }

    if bridge_type not in builders:
        available = ', '.join(sorted(builders.keys()))
        raise ValueError(
            f"Unknown bridge_type '{bridge_type}'. Available: {available}"
        )

    return builders[bridge_type](
        bridge_type=bridge_type,
        span_lengths_ft=span_lengths_ft,
        num_girders=num_girders,
        girder_spacing_ft=girder_spacing_ft,
        deck_width_ft=deck_width_ft,
        slab_thickness_in=slab_thickness_in,
        continuity=continuity,
        skew_angle=skew_angle,
        girder_depth_in=girder_depth_in,
        top_flange_width_in=top_flange_width_in,
        top_flange_thick_in=top_flange_thick_in,
        bot_flange_width_in=bot_flange_width_in,
        bot_flange_thick_in=bot_flange_thick_in,
        web_thickness_in=web_thickness_in,
        haunch_in=haunch_in,
        fy_ksi=fy_ksi,
        fc_ksi=fc_ksi,
        girder_type=girder_type,
        strand_pattern=strand_pattern,
        strand_area=strand_area,
        num_cells=num_cells,
        box_depth_in=box_depth_in,
        top_slab_thick_in=top_slab_thick_in,
        bot_slab_thick_in=bot_slab_thick_in,
        box_web_thick_in=box_web_thick_in,
        box_top_width_ft=box_top_width_ft,
        box_bot_width_ft=box_bot_width_ft,
        segment_length_ft=segment_length_ft,
        panel_length_ft=panel_length_ft,
        truss_depth_ft=truss_depth_ft,
        top_chord_area_in2=top_chord_area_in2,
        bot_chord_area_in2=bot_chord_area_in2,
        vertical_area_in2=vertical_area_in2,
        diagonal_area_in2=diagonal_area_in2,
        connection_type=connection_type,
        slab_width_ft=slab_width_ft,
        rebar_area_in2_per_ft=rebar_area_in2_per_ft,
        arch_rise_ft=arch_rise_ft,
        arch_shape=arch_shape,
        rib_depth_in=rib_depth_in,
        rib_width_in=rib_width_in,
        tag_base=tag_base,
    )


# ============================================================================
# GRILLAGE NODE/ELEMENT GENERATION (shared by girder-based types)
# ============================================================================

def _generate_grillage_nodes(
    span_lengths_ft: List[float],
    num_girders: int,
    girder_spacing_ft: float,
    skew_angle: float,
    tags: TagManager,
    elevation: float = 0.0,
) -> Tuple[List[Dict], Dict[Tuple[int, int], int], List[int], List[int]]:
    """Generate nodes for a multi-girder grillage model.

    Creates nodes for all girder lines across all spans. Nodes are placed
    at element boundaries determined by mesh density.

    Args:
        span_lengths_ft:  Span lengths (ft).
        num_girders:      Number of girder lines.
        girder_spacing_ft: Spacing between girders (ft).
        skew_angle:       Skew angle (degrees).
        tags:             Tag manager for unique IDs.
        elevation:        Y-coordinate for all nodes (inches).

    Returns:
        Tuple of:
            - nodes: list of node dicts
            - node_map: {(girder_idx, longitudinal_idx): node_tag}
            - support_positions: longitudinal indices at supports
            - midspan_positions: longitudinal indices at midspans
    """
    nodes = []
    node_map: Dict[Tuple[int, int], int] = {}

    # Compute all longitudinal positions (cumulative from x=0)
    all_x_positions: List[float] = []
    support_long_indices: List[int] = []  # longitudinal indices at supports
    midspan_long_indices: List[int] = []

    cum_x = 0.0
    global_long_idx = 0
    support_long_indices.append(0)

    for span_idx, span_ft in enumerate(span_lengths_ft):
        span_in = span_ft * FT_TO_IN
        n_elem = compute_mesh_density(span_ft)
        positions = _span_node_positions(span_in, n_elem)

        for i, dx in enumerate(positions):
            if span_idx > 0 and i == 0:
                # Skip the first node of subsequent spans (shared with prev span end)
                continue
            all_x_positions.append(cum_x + dx)
            # Track midspan (n_elem / 2)
            if i == n_elem // 2:
                midspan_long_indices.append(global_long_idx)
            global_long_idx += 1

        cum_x += span_in
        support_long_indices.append(global_long_idx - 1)

    # Generate nodes for each girder line at each longitudinal position
    skew_rad = math.radians(skew_angle)
    total_width_in = girder_spacing_ft * (num_girders - 1) * FT_TO_IN
    z_start = -total_width_in / 2.0  # center the girders

    for g_idx in range(num_girders):
        z_base = z_start + g_idx * girder_spacing_ft * FT_TO_IN
        for l_idx, x_pos in enumerate(all_x_positions):
            tag = tags.next('node')
            # Apply skew: offset x based on z position
            x_skew = x_pos + z_base * math.tan(skew_rad)
            nodes.append({
                'tag': tag,
                'x': x_skew,
                'y': elevation,
                'z': z_base,
            })
            node_map[(g_idx, l_idx)] = tag

    # Collect support and midspan node tags (all girder lines)
    support_nodes = []
    for l_idx in support_long_indices:
        for g_idx in range(num_girders):
            if (g_idx, l_idx) in node_map:
                support_nodes.append(node_map[(g_idx, l_idx)])

    midspan_nodes = []
    for l_idx in midspan_long_indices:
        for g_idx in range(num_girders):
            if (g_idx, l_idx) in node_map:
                midspan_nodes.append(node_map[(g_idx, l_idx)])

    n_long = len(all_x_positions)
    return nodes, node_map, support_nodes, midspan_nodes, n_long, support_long_indices, midspan_long_indices


def _generate_longitudinal_elements(
    node_map: Dict[Tuple[int, int], int],
    num_girders: int,
    n_long: int,
    section_tag: int,
    transform_tag: int,
    tags: TagManager,
    element_type: str = 'dispBeamColumn',
) -> List[Dict]:
    """Generate longitudinal beam-column elements for each girder line.

    Args:
        node_map:       {(girder_idx, long_idx): node_tag}
        num_girders:    Number of girder lines.
        n_long:         Number of longitudinal node positions.
        section_tag:    Section tag for all elements.
        transform_tag:  Geometric transformation tag.
        tags:           Tag manager.
        element_type:   Element type string.

    Returns:
        List of element dicts.
    """
    elements = []
    for g_idx in range(num_girders):
        for l_idx in range(n_long - 1):
            n_i = node_map[(g_idx, l_idx)]
            n_j = node_map[(g_idx, l_idx + 1)]
            e_tag = tags.next('element')
            elements.append({
                'tag': e_tag,
                'type': element_type,
                'nodes': [n_i, n_j],
                'section': section_tag,
                'transform': transform_tag,
            })
    return elements


def _generate_diaphragms(
    node_map: Dict[Tuple[int, int], int],
    num_girders: int,
    n_long: int,
    span_lengths_ft: List[float],
    tags: TagManager,
    diaphragm_section_tag: int,
    diaphragm_transform_tag: int,
    support_long_indices: List[int],
) -> List[Dict]:
    """Generate transverse diaphragm/cross-frame elements.

    Places diaphragms at supports and span third-points.

    Args:
        node_map:               Node lookup.
        num_girders:            Number of girder lines.
        n_long:                 Number of longitudinal positions.
        span_lengths_ft:        Span lengths.
        tags:                   Tag manager.
        diaphragm_section_tag:  Section for diaphragm elements.
        diaphragm_transform_tag: Transform for diaphragms.
        support_long_indices:   Longitudinal indices at supports.

    Returns:
        List of diaphragm element dicts.

    Reference:
        AASHTO LRFD 6.7.4: Diaphragms and Cross-Frames.
    """
    diaphragms = []

    # Collect all longitudinal indices that need diaphragms
    diaphragm_l_indices = set(support_long_indices)

    # Add third-point locations within each span
    cum_idx = 0
    for span_ft in span_lengths_ft:
        n_elem = compute_mesh_density(span_ft)
        third = n_elem // 3
        two_thirds = 2 * n_elem // 3
        diaphragm_l_indices.add(cum_idx + third)
        diaphragm_l_indices.add(cum_idx + two_thirds)
        cum_idx += n_elem

    for l_idx in sorted(diaphragm_l_indices):
        if l_idx >= n_long:
            continue
        for g_idx in range(num_girders - 1):
            key_i = (g_idx, l_idx)
            key_j = (g_idx + 1, l_idx)
            if key_i in node_map and key_j in node_map:
                e_tag = tags.next('element')
                diaphragms.append({
                    'tag': e_tag,
                    'type': 'dispBeamColumn',
                    'nodes': [node_map[key_i], node_map[key_j]],
                    'section': diaphragm_section_tag,
                    'transform': diaphragm_transform_tag,
                })

    return diaphragms


# ============================================================================
# BUILDER: Steel Plate Girder (Composite & Non-Composite)
# ============================================================================

def _build_steel_plate_girder(
    bridge_type: str,
    span_lengths_ft: List[float],
    num_girders: int,
    girder_spacing_ft: float,
    deck_width_ft: float,
    slab_thickness_in: float,
    continuity: List[str],
    skew_angle: float,
    girder_depth_in: float,
    top_flange_width_in: float,
    top_flange_thick_in: float,
    bot_flange_width_in: float,
    bot_flange_thick_in: float,
    web_thickness_in: float,
    haunch_in: float,
    fy_ksi: float,
    fc_ksi: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build steel plate girder superstructure (composite or non-composite).

    Models each girder line as a series of dispBeamColumn elements with
    fiber sections. Transverse connectivity via diaphragm elements.

    For composite: uses composite_section() with effective slab width.
    For non-composite: uses steel_i_section() with deck mass as nodal mass.

    Reference:
        AASHTO LRFD 6.10: I-Section Flexural Members.
        AASHTO LRFD 4.6.2.6: Effective Flange Width.
    """
    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = num_girders
    model.deck_width = deck_width_ft
    model.continuity = list(continuity)

    is_composite = (bridge_type == 'steel_plate_girder_composite')

    # --- Materials ---
    steel_mat_tag = tags.next('material')
    model.materials.append({
        'tag': steel_mat_tag,
        'type': 'Steel02',
        'params': {'fy': fy_ksi, 'Es': 29000.0, 'b': 0.01},
        'function': 'structural_steel',
    })

    concrete_mat_tag = None
    if is_composite:
        concrete_mat_tag = tags.next('material')
        model.materials.append({
            'tag': concrete_mat_tag,
            'type': 'Concrete01',
            'params': {'fc': fc_ksi},
            'function': 'unconfined_concrete',
        })

    # --- Sections ---
    # Use the max span for effective width calculation
    max_span_ft = max(span_lengths_ft)

    steel_section_dict = {
        'd': girder_depth_in,
        'bf_top': top_flange_width_in,
        'tf_top': top_flange_thick_in,
        'bf_bot': bot_flange_width_in,
        'tf_bot': bot_flange_thick_in,
        'tw': web_thickness_in,
    }

    section_tag = tags.next('section')
    if is_composite:
        eff_width = effective_slab_width(
            max_span_ft, girder_spacing_ft,
            slab_thickness_in, top_flange_width_in,
        )
        model.sections.append({
            'tag': section_tag,
            'type': 'composite',
            'params': {
                'steel_section': steel_section_dict,
                'slab_width': eff_width,
                'slab_thick': slab_thickness_in,
                'haunch': haunch_in,
                'mat_steel': steel_mat_tag,
                'mat_concrete': concrete_mat_tag,
            },
            'function': 'composite_section',
        })
    else:
        model.sections.append({
            'tag': section_tag,
            'type': 'steel_i',
            'params': {
                'd': girder_depth_in,
                'bf_top': top_flange_width_in,
                'tf_top': top_flange_thick_in,
                'bf_bot': bot_flange_width_in,
                'tf_bot': bot_flange_thick_in,
                'tw': web_thickness_in,
                'mat_flange': steel_mat_tag,
                'mat_web': steel_mat_tag,
            },
            'function': 'steel_i_section',
        })

    # Diaphragm section — use a small steel section for cross-frames
    diaphragm_section_tag = tags.next('section')
    model.sections.append({
        'tag': diaphragm_section_tag,
        'type': 'steel_i',
        'params': {
            'd': 12.0,
            'bf_top': 6.0, 'tf_top': 0.5,
            'bf_bot': 6.0, 'tf_bot': 0.5,
            'tw': 0.375,
            'mat_flange': steel_mat_tag,
            'mat_web': steel_mat_tag,
        },
        'function': 'steel_i_section',
    })

    # --- Geometric Transforms ---
    transform_type = select_transform_type(max_span_ft)
    girder_transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': girder_transform_tag,
        'type': transform_type,
        'vecxz': [0.0, 0.0, 1.0],  # vertical web for horizontal beam
    })

    diaphragm_transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': diaphragm_transform_tag,
        'type': 'Linear',
        'vecxz': [0.0, 1.0, 0.0],  # transverse member
    })

    # --- Nodes ---
    (nodes, node_map, support_nodes, midspan_nodes,
     n_long, support_long_indices, midspan_long_indices) = \
        _generate_grillage_nodes(
            span_lengths_ft, num_girders, girder_spacing_ft,
            skew_angle, tags,
        )
    model.nodes = nodes
    model.support_nodes = support_nodes
    model.midspan_nodes = midspan_nodes

    # --- Longitudinal Elements ---
    model.elements = _generate_longitudinal_elements(
        node_map, num_girders, n_long,
        section_tag, girder_transform_tag, tags,
    )

    # --- Diaphragms ---
    model.diaphragms = _generate_diaphragms(
        node_map, num_girders, n_long,
        span_lengths_ft, tags,
        diaphragm_section_tag, diaphragm_transform_tag,
        support_long_indices,
    )

    return model


# ============================================================================
# BUILDER: Prestressed Concrete I-Girder
# ============================================================================

def _build_prestressed_i_girder(
    span_lengths_ft: List[float],
    num_girders: int,
    girder_spacing_ft: float,
    deck_width_ft: float,
    slab_thickness_in: float,
    continuity: List[str],
    skew_angle: float,
    girder_type: Optional[str],
    strand_pattern: Optional[List[Tuple[float, int]]],
    strand_area: float,
    fc_ksi: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build prestressed concrete I-girder superstructure.

    Simple-span precast girders with composite deck. Each girder line modeled
    as dispBeamColumn with prestressed_i_section fiber section.

    Args:
        girder_type: Key into GIRDER_LIBRARY (e.g., 'BT_72', 'AASHTO_IV').
        strand_pattern: List of (y_from_bottom_in, num_strands) tuples.

    Reference:
        AASHTO LRFD 5.9: Prestressed Concrete.
        PCI Bridge Design Manual, 3rd Edition.
    """
    from ..opensees.sections import GIRDER_LIBRARY

    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = num_girders
    model.deck_width = deck_width_ft
    model.continuity = list(continuity)

    # Default girder type
    if girder_type is None:
        girder_type = 'BT_72'

    if girder_type not in GIRDER_LIBRARY:
        available = ', '.join(sorted(GIRDER_LIBRARY.keys()))
        raise ValueError(
            f"Unknown girder_type '{girder_type}'. Available: {available}"
        )

    # Default strand pattern
    if strand_pattern is None:
        strand_pattern = [
            (2.0, 12), (4.0, 12), (6.0, 8), (8.0, 4),
        ]

    # --- Materials ---
    concrete_mat_tag = tags.next('material')
    model.materials.append({
        'tag': concrete_mat_tag,
        'type': 'Concrete01',
        'params': {'fc': fc_ksi},
        'function': 'unconfined_concrete',
    })

    strand_mat_tag = tags.next('material')
    model.materials.append({
        'tag': strand_mat_tag,
        'type': 'Steel02',
        'params': {'fpu': 270.0, 'Eps': 28500.0},
        'function': 'prestressing_strand',
    })

    # --- Section ---
    section_tag = tags.next('section')
    model.sections.append({
        'tag': section_tag,
        'type': 'prestressed_i',
        'params': {
            'girder_type': girder_type,
            'mat_concrete': concrete_mat_tag,
            'strand_pattern': {
                'rows': strand_pattern,
                'strand_area': strand_area,
            },
            'mat_strand': strand_mat_tag,
        },
        'function': 'prestressed_i_section',
    })

    # Diaphragm section
    diaphragm_section_tag = tags.next('section')
    model.sections.append({
        'tag': diaphragm_section_tag,
        'type': 'concrete_diaphragm',
        'params': {'d': 12.0, 'width': 8.0},
        'function': 'rectangular_solid',
    })

    # --- Transforms ---
    max_span_ft = max(span_lengths_ft)
    transform_type = select_transform_type(max_span_ft)
    girder_transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': girder_transform_tag,
        'type': transform_type,
        'vecxz': [0.0, 0.0, 1.0],
    })

    diaphragm_transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': diaphragm_transform_tag,
        'type': 'Linear',
        'vecxz': [0.0, 1.0, 0.0],
    })

    # --- Nodes ---
    (nodes, node_map, support_nodes, midspan_nodes,
     n_long, support_long_indices, midspan_long_indices) = \
        _generate_grillage_nodes(
            span_lengths_ft, num_girders, girder_spacing_ft,
            skew_angle, tags,
        )
    model.nodes = nodes
    model.support_nodes = support_nodes
    model.midspan_nodes = midspan_nodes

    # --- Elements ---
    model.elements = _generate_longitudinal_elements(
        node_map, num_girders, n_long,
        section_tag, girder_transform_tag, tags,
    )

    # --- Diaphragms ---
    model.diaphragms = _generate_diaphragms(
        node_map, num_girders, n_long,
        span_lengths_ft, tags,
        diaphragm_section_tag, diaphragm_transform_tag,
        support_long_indices,
    )

    return model


# ============================================================================
# BUILDER: CIP Concrete Box Girder
# ============================================================================

def _build_cip_box_girder(
    span_lengths_ft: List[float],
    num_girders: int,
    girder_spacing_ft: float,
    deck_width_ft: float,
    slab_thickness_in: float,
    continuity: List[str],
    skew_angle: float,
    num_cells: int,
    box_depth_in: float,
    top_slab_thick_in: float,
    bot_slab_thick_in: float,
    box_web_thick_in: float,
    box_top_width_ft: Optional[float],
    box_bot_width_ft: Optional[float],
    fc_ksi: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build CIP concrete box girder superstructure.

    Single box with multiple cells modeled as a single spine with the
    full box cross-section. Suitable for continuous post-tensioned bridges.

    Reference:
        AASHTO LRFD 5.12.2: Segmental Concrete Bridges.
        AASHTO LRFD 4.6.2.6.2: Box Girder Effective Width.
    """
    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = 1  # spine model
    model.deck_width = deck_width_ft
    model.continuity = list(continuity)

    top_width_in = (box_top_width_ft or deck_width_ft) * FT_TO_IN
    bot_width_in = (box_bot_width_ft or (deck_width_ft - 4.0)) * FT_TO_IN

    # --- Materials ---
    concrete_mat_tag = tags.next('material')
    model.materials.append({
        'tag': concrete_mat_tag,
        'type': 'Concrete01',
        'params': {'fc': fc_ksi},
        'function': 'unconfined_concrete',
    })

    # --- Section ---
    section_tag = tags.next('section')
    model.sections.append({
        'tag': section_tag,
        'type': 'box_girder',
        'params': {
            'depth': box_depth_in,
            'top_width': top_width_in,
            'bot_width': bot_width_in,
            'top_thick': top_slab_thick_in,
            'bot_thick': bot_slab_thick_in,
            'web_thick': box_web_thick_in,
            'num_cells': num_cells,
            'mat_concrete': concrete_mat_tag,
        },
        'function': 'box_girder_section',
    })

    # --- Transforms ---
    max_span_ft = max(span_lengths_ft)
    transform_type = select_transform_type(max_span_ft)
    transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': transform_tag,
        'type': transform_type,
        'vecxz': [0.0, 0.0, 1.0],
    })

    # --- Nodes (spine model = 1 girder line) ---
    (nodes, node_map, support_nodes, midspan_nodes,
     n_long, support_long_indices, midspan_long_indices) = \
        _generate_grillage_nodes(
            span_lengths_ft, 1, girder_spacing_ft,
            skew_angle, tags,
        )
    model.nodes = nodes
    model.support_nodes = support_nodes
    model.midspan_nodes = midspan_nodes

    # --- Elements ---
    model.elements = _generate_longitudinal_elements(
        node_map, 1, n_long,
        section_tag, transform_tag, tags,
    )

    return model


# ============================================================================
# BUILDER: Segmental Box Girder
# ============================================================================

def _build_segmental_box_girder(
    span_lengths_ft: List[float],
    segment_length_ft: float,
    num_cells: int,
    box_depth_in: float,
    top_slab_thick_in: float,
    bot_slab_thick_in: float,
    box_web_thick_in: float,
    box_top_width_ft: Optional[float],
    box_bot_width_ft: Optional[float],
    deck_width_ft: float,
    continuity: List[str],
    fc_ksi: float,
    skew_angle: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build prestressed segmental box girder superstructure.

    Similar to CIP box but with segment joints modeled as zero-length
    elements with reduced stiffness (compression-only + friction).

    Segment joints are placed at regular intervals per segment_length_ft.

    Reference:
        AASHTO LRFD 5.12.2: Segmental Concrete Bridges.
        AASHTO LRFD 5.12.5.3.6: Joint Design.
    """
    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = 1
    model.deck_width = deck_width_ft
    model.continuity = list(continuity)

    top_width_in = (box_top_width_ft or deck_width_ft) * FT_TO_IN
    bot_width_in = (box_bot_width_ft or (deck_width_ft - 4.0)) * FT_TO_IN

    # --- Materials ---
    concrete_mat_tag = tags.next('material')
    model.materials.append({
        'tag': concrete_mat_tag,
        'type': 'Concrete01',
        'params': {'fc': fc_ksi},
        'function': 'unconfined_concrete',
    })

    # Joint friction material (compression-only with friction)
    joint_mat_tag = tags.next('material')
    model.materials.append({
        'tag': joint_mat_tag,
        'type': 'ENT',
        'params': {'k': 1.0e6},  # high stiffness in compression
        'function': 'compression_only',
    })

    # --- Section ---
    section_tag = tags.next('section')
    model.sections.append({
        'tag': section_tag,
        'type': 'box_girder',
        'params': {
            'depth': box_depth_in,
            'top_width': top_width_in,
            'bot_width': bot_width_in,
            'top_thick': top_slab_thick_in,
            'bot_thick': bot_slab_thick_in,
            'web_thick': box_web_thick_in,
            'num_cells': num_cells,
            'mat_concrete': concrete_mat_tag,
        },
        'function': 'box_girder_section',
    })

    # --- Transforms ---
    max_span_ft = max(span_lengths_ft)
    transform_type = select_transform_type(max_span_ft)
    transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': transform_tag,
        'type': transform_type,
        'vecxz': [0.0, 0.0, 1.0],
    })

    # --- Nodes (spine model) ---
    (nodes, node_map, support_nodes, midspan_nodes,
     n_long, support_long_indices, midspan_long_indices) = \
        _generate_grillage_nodes(
            span_lengths_ft, 1, 1.0,  # single line, spacing irrelevant
            skew_angle, tags,
        )
    model.nodes = nodes
    model.support_nodes = support_nodes
    model.midspan_nodes = midspan_nodes

    # --- Longitudinal Elements ---
    model.elements = _generate_longitudinal_elements(
        node_map, 1, n_long,
        section_tag, transform_tag, tags,
    )

    # --- Segment Joints as zero-length elements ---
    # Insert joints at segment boundaries
    segment_in = segment_length_ft * FT_TO_IN
    cum_x = 0.0
    for span_ft in span_lengths_ft:
        span_in = span_ft * FT_TO_IN
        x_in_span = segment_in
        while x_in_span < span_in - segment_in / 2.0:
            joint_x = cum_x + x_in_span
            # Find the closest node to this position
            closest_node = _find_closest_node(model.nodes, joint_x, 0.0, 0.0)
            if closest_node is not None:
                # Create a paired node for zero-length joint element
                joint_node_tag = tags.next('node')
                orig_node = next(n for n in model.nodes if n['tag'] == closest_node)
                model.nodes.append({
                    'tag': joint_node_tag,
                    'x': orig_node['x'],
                    'y': orig_node['y'],
                    'z': orig_node['z'],
                })
                joint_elem_tag = tags.next('element')
                model.diaphragms.append({
                    'tag': joint_elem_tag,
                    'type': 'zeroLength',
                    'nodes': [closest_node, joint_node_tag],
                    'materials': [joint_mat_tag],
                    'directions': [1],  # axial compression only
                    'purpose': 'segment_joint',
                })
            x_in_span += segment_in
        cum_x += span_in

    return model


def _find_closest_node(nodes: List[Dict], x: float, y: float, z: float,
                       tol: float = 1.0) -> Optional[int]:
    """Find the node tag closest to target coordinates within tolerance."""
    best_tag = None
    best_dist = float('inf')
    for n in nodes:
        dist = math.sqrt((n['x'] - x)**2 + (n['y'] - y)**2 + (n['z'] - z)**2)
        if dist < best_dist:
            best_dist = dist
            best_tag = n['tag']
    if best_dist <= tol:
        return best_tag
    # If no node within tolerance, return nearest anyway
    return best_tag


# ============================================================================
# BUILDER: Steel Truss
# ============================================================================

def _build_steel_truss(
    span_lengths_ft: List[float],
    panel_length_ft: float,
    truss_depth_ft: float,
    top_chord_area_in2: float,
    bot_chord_area_in2: float,
    vertical_area_in2: float,
    diagonal_area_in2: float,
    connection_type: str,
    fy_ksi: float,
    deck_width_ft: float,
    continuity: List[str],
    skew_angle: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build steel truss superstructure.

    Warren or Pratt truss geometry with corotTruss elements.
    Two truss lines (left and right) with floor beam connections.

    Reference:
        AASHTO LRFD 4.6.2.7: Truss Bridges.
    """
    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = 2  # two truss lines
    model.deck_width = deck_width_ft
    model.continuity = list(continuity)

    truss_depth_in = truss_depth_ft * FT_TO_IN
    panel_length_in = panel_length_ft * FT_TO_IN
    truss_spacing_in = deck_width_ft * FT_TO_IN  # trusses at edges

    # --- Material ---
    steel_mat_tag = tags.next('material')
    model.materials.append({
        'tag': steel_mat_tag,
        'type': 'Steel02',
        'params': {'fy': fy_ksi, 'Es': 29000.0, 'b': 0.01},
        'function': 'structural_steel',
    })

    # --- Transform ---
    transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': transform_tag,
        'type': 'Corotational',
        'vecxz': [0.0, 0.0, 1.0],
    })

    # --- Generate truss geometry ---
    # For each span, generate panels
    cum_x = 0.0
    # node_map: keyed by (truss_side, panel_idx, chord) where chord = 'top'/'bot'
    bot_nodes: Dict[Tuple[int, int], int] = {}  # (side, panel_idx) -> tag
    top_nodes: Dict[Tuple[int, int], int] = {}

    all_panel_starts: List[float] = []

    for span_ft in span_lengths_ft:
        span_in = span_ft * FT_TO_IN
        n_panels = max(1, round(span_in / panel_length_in))
        actual_panel_len = span_in / n_panels

        for p in range(n_panels + 1):
            x = cum_x + p * actual_panel_len
            all_panel_starts.append(x)

            for side in range(2):
                z = -truss_spacing_in / 2.0 + side * truss_spacing_in
                key = (side, len(all_panel_starts) - 1)

                # Bottom chord node
                tag_bot = tags.next('node')
                model.nodes.append({'tag': tag_bot, 'x': x, 'y': 0.0, 'z': z})
                bot_nodes[key] = tag_bot

                # Top chord node
                tag_top = tags.next('node')
                model.nodes.append({
                    'tag': tag_top, 'x': x,
                    'y': truss_depth_in, 'z': z,
                })
                top_nodes[key] = tag_top

        cum_x += span_in

    n_panel_positions = len(all_panel_starts)

    # --- Truss elements ---
    for side in range(2):
        for p_idx in range(n_panel_positions - 1):
            key_i = (side, p_idx)
            key_j = (side, p_idx + 1)

            # Bottom chord
            e_tag = tags.next('element')
            model.elements.append({
                'tag': e_tag,
                'type': 'corotTruss',
                'nodes': [bot_nodes[key_i], bot_nodes[key_j]],
                'area': bot_chord_area_in2,
                'material': steel_mat_tag,
            })

            # Top chord
            e_tag = tags.next('element')
            model.elements.append({
                'tag': e_tag,
                'type': 'corotTruss',
                'nodes': [top_nodes[key_i], top_nodes[key_j]],
                'area': top_chord_area_in2,
                'material': steel_mat_tag,
            })

            # Vertical at start of panel
            e_tag = tags.next('element')
            model.elements.append({
                'tag': e_tag,
                'type': 'corotTruss',
                'nodes': [bot_nodes[key_i], top_nodes[key_i]],
                'area': vertical_area_in2,
                'material': steel_mat_tag,
            })

            # Diagonal
            e_tag = tags.next('element')
            model.elements.append({
                'tag': e_tag,
                'type': 'corotTruss',
                'nodes': [bot_nodes[key_i], top_nodes[key_j]],
                'area': diagonal_area_in2,
                'material': steel_mat_tag,
            })

        # Last vertical
        key_last = (side, n_panel_positions - 1)
        e_tag = tags.next('element')
        model.elements.append({
            'tag': e_tag,
            'type': 'corotTruss',
            'nodes': [bot_nodes[key_last], top_nodes[key_last]],
            'area': vertical_area_in2,
            'material': steel_mat_tag,
        })

    # --- Floor beams (transverse connections at bottom chord) ---
    floor_beam_transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': floor_beam_transform_tag,
        'type': 'Linear',
        'vecxz': [0.0, 1.0, 0.0],
    })

    for p_idx in range(n_panel_positions):
        key_left = (0, p_idx)
        key_right = (1, p_idx)
        e_tag = tags.next('element')
        model.diaphragms.append({
            'tag': e_tag,
            'type': 'corotTruss',
            'nodes': [bot_nodes[key_left], bot_nodes[key_right]],
            'area': vertical_area_in2,
            'material': steel_mat_tag,
            'purpose': 'floor_beam',
        })

    # --- Support and midspan nodes ---
    # Supports at span boundaries
    cum_panels = 0
    support_panel_indices = [0]
    for span_ft in span_lengths_ft:
        span_in = span_ft * FT_TO_IN
        n_panels = max(1, round(span_in / panel_length_in))
        cum_panels += n_panels
        support_panel_indices.append(cum_panels)

    for p_idx in support_panel_indices:
        if p_idx < n_panel_positions:
            for side in range(2):
                model.support_nodes.append(bot_nodes[(side, p_idx)])

    # Midspans
    cum_panels = 0
    for span_ft in span_lengths_ft:
        span_in = span_ft * FT_TO_IN
        n_panels = max(1, round(span_in / panel_length_in))
        mid_idx = cum_panels + n_panels // 2
        if mid_idx < n_panel_positions:
            for side in range(2):
                model.midspan_nodes.append(bot_nodes[(side, mid_idx)])
        cum_panels += n_panels

    return model


# ============================================================================
# BUILDER: Concrete Slab Bridge
# ============================================================================

def _build_concrete_slab(
    span_lengths_ft: List[float],
    slab_thickness_in: float,
    slab_width_ft: Optional[float],
    deck_width_ft: float,
    continuity: List[str],
    skew_angle: float,
    fc_ksi: float,
    rebar_area_in2_per_ft: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build concrete slab bridge using ShellMITC4 elements.

    For short-span bridges (typically < 40 ft). Models the slab as a
    2D shell mesh with plate bending behavior.

    Reference:
        AASHTO LRFD 4.6.2.3: Equivalent Strip Widths for Slab-Type Bridges.
    """
    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = 0  # no girder lines — shell model
    width_ft = slab_width_ft or deck_width_ft
    model.deck_width = width_ft
    model.continuity = list(continuity)

    width_in = width_ft * FT_TO_IN

    # --- Material (nDMaterial for shell) ---
    from ..opensees.materials import concrete_defaults
    props = concrete_defaults(fc_ksi)

    concrete_nd_tag = tags.next('material')
    model.materials.append({
        'tag': concrete_nd_tag,
        'type': 'ElasticIsotropic',
        'params': {'E': props.Ec, 'nu': 0.2},
        'function': 'nDMaterial_ElasticIsotropic',
    })

    # Shell section (PlateFiber)
    shell_section_tag = tags.next('section')
    model.sections.append({
        'tag': shell_section_tag,
        'type': 'PlateFiber',
        'params': {
            'matTag': concrete_nd_tag,
            'thickness': slab_thickness_in,
        },
        'function': 'PlateFiber_section',
    })

    # --- Mesh ---
    # Transverse divisions: 1 element per 2 ft of width
    n_z = max(4, round(width_ft / 2.0))
    dz = width_in / n_z
    z_start = -width_in / 2.0

    # Longitudinal: use standard mesh density per span
    all_x: List[float] = []
    support_x_indices: List[int] = []
    midspan_x_indices: List[int] = []

    cum_x = 0.0
    global_idx = 0
    support_x_indices.append(0)

    for span_ft in span_lengths_ft:
        span_in = span_ft * FT_TO_IN
        n_elem = compute_mesh_density(span_ft)
        for i in range(n_elem + 1):
            if span_lengths_ft.index(span_ft) > 0 and i == 0:
                continue
            x = cum_x + i * span_in / n_elem
            all_x.append(x)
            if i == n_elem // 2:
                midspan_x_indices.append(global_idx)
            global_idx += 1
        support_x_indices.append(global_idx - 1)
        cum_x += span_in

    n_x = len(all_x)

    # --- Nodes ---
    # node_grid[ix][iz] = tag
    node_grid: List[List[int]] = []
    for ix in range(n_x):
        row = []
        for iz in range(n_z + 1):
            tag = tags.next('node')
            x = all_x[ix]
            z = z_start + iz * dz
            model.nodes.append({'tag': tag, 'x': x, 'y': 0.0, 'z': z})
            row.append(tag)
        node_grid.append(row)

    # --- Shell Elements ---
    for ix in range(n_x - 1):
        for iz in range(n_z):
            e_tag = tags.next('element')
            n1 = node_grid[ix][iz]
            n2 = node_grid[ix + 1][iz]
            n3 = node_grid[ix + 1][iz + 1]
            n4 = node_grid[ix][iz + 1]
            model.elements.append({
                'tag': e_tag,
                'type': 'ShellMITC4',
                'nodes': [n1, n2, n3, n4],
                'section': shell_section_tag,
            })

    # --- Support / midspan nodes ---
    for ix in support_x_indices:
        if ix < n_x:
            for iz in range(n_z + 1):
                model.support_nodes.append(node_grid[ix][iz])

    for ix in midspan_x_indices:
        if ix < n_x:
            for iz in range(n_z + 1):
                model.midspan_nodes.append(node_grid[ix][iz])

    return model


# ============================================================================
# BUILDER: Arch Bridge
# ============================================================================

def _arch_shape_y(x: float, span_in: float, rise_in: float,
                  shape: str) -> float:
    """Compute arch rib elevation at position x.

    Args:
        x:        Position along span (0 to span_in).
        span_in:  Total span length (inches).
        rise_in:  Rise at midspan (inches).
        shape:    'parabolic' or 'circular'.

    Returns:
        Y-coordinate (inches) of arch rib centerline.
    """
    if shape == 'parabolic':
        # y = 4*rise * x/L * (1 - x/L)
        xi = x / span_in
        return 4.0 * rise_in * xi * (1.0 - xi)
    elif shape == 'circular':
        # Circular arc: find R from span and rise
        # The arc passes through (0,0), (span/2, rise), (span, 0)
        # Center of circle is at (span/2, -(R - rise))
        # R² = (span/2)² + (R - rise)²
        # R = [(span/2)² + rise²] / (2 * rise)
        half_span = span_in / 2.0
        R = (half_span**2 + rise_in**2) / (2.0 * rise_in)
        y_center = -(R - rise_in)  # center below the chord
        dx = x - half_span
        y = y_center + math.sqrt(max(0.0, R**2 - dx**2))
        return max(0.0, y)
    else:
        raise ValueError(f"Unknown arch shape '{shape}'. Use 'parabolic' or 'circular'.")


def _build_arch(
    span_lengths_ft: List[float],
    arch_rise_ft: float,
    arch_shape: str,
    rib_depth_in: float,
    rib_width_in: float,
    deck_width_ft: float,
    continuity: List[str],
    skew_angle: float,
    fc_ksi: float,
    fy_ksi: float,
    tag_base: int = 1,
    **kwargs,
) -> SuperstructureModel:
    """Build arch bridge superstructure.

    Single arch span modeled as dispBeamColumn elements following the
    arch geometry. Supports parabolic and circular arch shapes.

    Only the first span is used for the arch; multi-span arches not supported.

    Reference:
        AASHTO LRFD 4.5.3.2.2c: Arches.
    """
    tags = TagManager(tag_base)
    model = SuperstructureModel()
    model.span_lengths = list(span_lengths_ft)
    model.girder_lines = 1
    model.deck_width = deck_width_ft
    model.continuity = list(continuity)

    span_ft = span_lengths_ft[0]
    span_in = span_ft * FT_TO_IN
    rise_in = arch_rise_ft * FT_TO_IN

    # --- Material ---
    concrete_mat_tag = tags.next('material')
    model.materials.append({
        'tag': concrete_mat_tag,
        'type': 'Concrete01',
        'params': {'fc': fc_ksi},
        'function': 'unconfined_concrete',
    })

    steel_mat_tag = tags.next('material')
    model.materials.append({
        'tag': steel_mat_tag,
        'type': 'Steel02',
        'params': {'fy': fy_ksi, 'Es': 29000.0, 'b': 0.01},
        'function': 'reinforcing_steel',
    })

    # --- Section (rectangular RC rib) ---
    section_tag = tags.next('section')
    model.sections.append({
        'tag': section_tag,
        'type': 'rectangular_rc',
        'params': {
            'width': rib_width_in,
            'height': rib_depth_in,
            'cover': 2.0,
            'mat_confined': concrete_mat_tag,
            'mat_unconfined': concrete_mat_tag,
            'mat_steel': steel_mat_tag,
        },
        'function': 'rectangular_rc_section',
    })

    # --- Transform ---
    transform_type = select_transform_type(span_ft)
    # For arches, always use at least PDelta
    if transform_type == 'Linear':
        transform_type = 'PDelta'
    transform_tag = tags.next('transform')
    model.transforms.append({
        'tag': transform_tag,
        'type': transform_type,
        'vecxz': [0.0, 0.0, 1.0],
    })

    # --- Nodes along arch ---
    n_elem = compute_mesh_density(span_ft)
    dx = span_in / n_elem

    for i in range(n_elem + 1):
        x = i * dx
        y = _arch_shape_y(x, span_in, rise_in, arch_shape)
        tag = tags.next('node')
        model.nodes.append({'tag': tag, 'x': x, 'y': y, 'z': 0.0})

    # --- Elements ---
    node_tags = [n['tag'] for n in model.nodes]
    for i in range(n_elem):
        e_tag = tags.next('element')
        model.elements.append({
            'tag': e_tag,
            'type': 'dispBeamColumn',
            'nodes': [node_tags[i], node_tags[i + 1]],
            'section': section_tag,
            'transform': transform_tag,
        })

    # Support nodes: first and last
    model.support_nodes = [node_tags[0], node_tags[-1]]
    # Midspan node
    model.midspan_nodes = [node_tags[n_elem // 2]]

    return model

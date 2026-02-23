"""
OpenSees Fiber Section Builders for Bridge Engineering.

All internal units: kip-inch-second (KIS).
    Stress: ksi
    Length: inch
    Area:   in²

Each function creates an OpenSees Fiber section using ops.section('Fiber', ...)
with appropriate ops.patch() and ops.layer() commands.

Coordinate convention:
    y-axis = vertical (depth direction)
    z-axis = horizontal (width direction)
    Origin = centroid of section

References:
    - AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    - PCI Bridge Design Manual, 3rd Edition (2014)
    - AISC Steel Construction Manual, 15th Edition (2017)
    - Caltrans Seismic Design Criteria v2.0 (2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import openseespy.opensees as ops


# ============================================================================
# STANDARD GIRDER DIMENSIONS
# ============================================================================

# AASHTO/PCI standard girder types: {name: {d, bt, bb, tw, tf_top, tf_bot, ...}}
GIRDER_LIBRARY: Dict[str, Dict[str, float]] = {
    "AASHTO_I":    {"d": 28.0, "bt": 12.0, "bb": 16.0, "tw": 6.0,
                    "tf_top": 4.0, "tf_bot": 5.0},
    "AASHTO_II":   {"d": 36.0, "bt": 12.0, "bb": 18.0, "tw": 6.0,
                    "tf_top": 6.0, "tf_bot": 6.0},
    "AASHTO_III":  {"d": 45.0, "bt": 16.0, "bb": 22.0, "tw": 7.0,
                    "tf_top": 7.0, "tf_bot": 7.0},
    "AASHTO_IV":   {"d": 54.0, "bt": 20.0, "bb": 26.0, "tw": 8.0,
                    "tf_top": 8.0, "tf_bot": 8.0},
    "AASHTO_V":    {"d": 63.0, "bt": 42.0, "bb": 28.0, "tw": 8.0,
                    "tf_top": 5.0, "tf_bot": 8.0},
    "AASHTO_VI":   {"d": 72.0, "bt": 42.0, "bb": 28.0, "tw": 8.0,
                    "tf_top": 5.0, "tf_bot": 8.0},
    "BT_54":       {"d": 54.0, "bt": 42.0, "bb": 26.0, "tw": 6.0,
                    "tf_top": 3.5, "tf_bot": 8.0},
    "BT_63":       {"d": 63.0, "bt": 42.0, "bb": 26.0, "tw": 6.0,
                    "tf_top": 3.5, "tf_bot": 9.0},
    "BT_72":       {"d": 72.0, "bt": 42.0, "bb": 26.0, "tw": 6.0,
                    "tf_top": 3.5, "tf_bot": 10.0},
    "NU_900":      {"d": 35.4, "bt": 39.4, "bb": 18.1, "tw": 5.9,
                    "tf_top": 3.1, "tf_bot": 7.1},
    "NU_1100":     {"d": 43.3, "bt": 39.4, "bb": 18.1, "tw": 5.9,
                    "tf_top": 3.1, "tf_bot": 7.1},
    "NU_1350":     {"d": 53.1, "bt": 39.4, "bb": 18.1, "tw": 5.9,
                    "tf_top": 3.1, "tf_bot": 7.1},
    "NU_1600":     {"d": 63.0, "bt": 39.4, "bb": 18.1, "tw": 5.9,
                    "tf_top": 3.1, "tf_bot": 7.1},
    "NU_2000":     {"d": 78.7, "bt": 39.4, "bb": 18.1, "tw": 5.9,
                    "tf_top": 3.1, "tf_bot": 7.1},
}


# ============================================================================
# STEEL I-SECTION
# ============================================================================

def steel_i_section(tag: int, d: float, bf_top: float, tf_top: float,
                    bf_bot: float, tf_bot: float, tw: float,
                    mat_flange: int, mat_web: int,
                    nf_flange: int = 4, nf_web: int = 8) -> int:
    """Create fiber section for a steel I-shape (plate girder or rolled).

    The section is modeled with rectangular fiber patches for each component:
    top flange, bottom flange, and web.

    Coordinate system:
        y=0 at centroid, positive up.
        z=0 at center of web.

    Args:
        tag:        Section tag.
        d:          Total depth (inches).
        bf_top:     Top flange width (inches).
        tf_top:     Top flange thickness (inches).
        bf_bot:     Bottom flange width (inches).
        tf_bot:     Bottom flange thickness (inches).
        tw:         Web thickness (inches).
        mat_flange: Material tag for flanges.
        mat_web:    Material tag for web.
        nf_flange:  Number of fibers across flange thickness. Default: 4.
        nf_web:     Number of fibers along web depth. Default: 8.

    Returns:
        Section tag.

    Reference:
        AASHTO LRFD 6.10: I-Section Flexural Members.
        AISC 15th Ed Chapter F: Design of Members for Flexure.
    """
    # Compute centroid (measured from bottom)
    # Areas
    A_tf = bf_top * tf_top
    A_bf = bf_bot * tf_bot
    d_web = d - tf_top - tf_bot
    A_w = d_web * tw

    y_bf = tf_bot / 2.0
    y_w = tf_bot + d_web / 2.0
    y_tf = d - tf_top / 2.0

    A_total = A_tf + A_bf + A_w
    if A_total <= 0:
        raise ValueError("Section has zero or negative area.")
    y_bar = (A_bf * y_bf + A_w * y_w + A_tf * y_tf) / A_total

    # Convert to centroidal coordinates (y=0 at centroid)
    ops.section('Fiber', tag)

    # Bottom flange: rect patch
    yI_bf = (tf_bot / 2.0) - y_bar  # center of bottom flange
    ops.patch('rect', mat_flange, nf_flange, max(2, int(bf_bot / 2)),
              yI_bf - tf_bot / 2.0, -bf_bot / 2.0,
              yI_bf + tf_bot / 2.0, bf_bot / 2.0)

    # Web
    yI_w_bot = tf_bot - y_bar
    yI_w_top = (d - tf_top) - y_bar
    ops.patch('rect', mat_web, nf_web, 1,
              yI_w_bot, -tw / 2.0,
              yI_w_top, tw / 2.0)

    # Top flange
    yI_tf = (d - tf_top / 2.0) - y_bar
    ops.patch('rect', mat_flange, nf_flange, max(2, int(bf_top / 2)),
              yI_tf - tf_top / 2.0, -bf_top / 2.0,
              yI_tf + tf_top / 2.0, bf_top / 2.0)

    return tag


# ============================================================================
# COMPOSITE SECTION
# ============================================================================

def composite_section(tag: int, steel_section: Dict[str, float],
                      slab_width: float, slab_thick: float,
                      haunch: float, mat_steel: int,
                      mat_concrete: int,
                      nf_slab: int = 4, nf_flange: int = 4,
                      nf_web: int = 8) -> int:
    """Create composite steel + concrete slab fiber section.

    Models a steel I-girder acting compositely with a concrete deck slab.
    The slab is placed above the steel section with an optional haunch.

    Args:
        tag:            Section tag.
        steel_section:  Dict with keys: 'd', 'bf_top', 'tf_top', 'bf_bot',
                        'tf_bot', 'tw' (all in inches).
        slab_width:     Effective slab width (inches). Per AASHTO 4.6.2.6.
        slab_thick:     Slab thickness (inches).
        haunch:         Haunch depth between top flange and slab bottom (inches).
        mat_steel:      Material tag for steel.
        mat_concrete:   Material tag for concrete (slab).
        nf_slab:        Fibers through slab thickness. Default: 4.
        nf_flange:      Fibers through flange thickness. Default: 4.
        nf_web:         Fibers along web depth. Default: 8.

    Returns:
        Section tag.

    Reference:
        AASHTO LRFD 6.10.1.1: Composite Sections.
        AASHTO LRFD 4.6.2.6: Effective Flange Width.
    """
    d = steel_section['d']
    bf_top = steel_section['bf_top']
    tf_top = steel_section['tf_top']
    bf_bot = steel_section['bf_bot']
    tf_bot = steel_section['tf_bot']
    tw = steel_section['tw']

    # Total composite depth
    total_depth = d + haunch + slab_thick

    # Compute composite centroid (from bottom of steel)
    A_bf = bf_bot * tf_bot
    d_web = d - tf_top - tf_bot
    A_w = d_web * tw
    A_tf = bf_top * tf_top
    A_slab = slab_width * slab_thick

    y_bf = tf_bot / 2.0
    y_w = tf_bot + d_web / 2.0
    y_tf = d - tf_top / 2.0
    y_slab = d + haunch + slab_thick / 2.0

    A_total = A_bf + A_w + A_tf + A_slab
    y_bar = (A_bf * y_bf + A_w * y_w + A_tf * y_tf + A_slab * y_slab) / A_total

    ops.section('Fiber', tag)

    # Bottom flange
    yI_bf = tf_bot / 2.0 - y_bar
    ops.patch('rect', mat_steel, nf_flange, max(2, int(bf_bot / 2)),
              yI_bf - tf_bot / 2.0, -bf_bot / 2.0,
              yI_bf + tf_bot / 2.0, bf_bot / 2.0)

    # Web
    yI_w_bot = tf_bot - y_bar
    yI_w_top = (d - tf_top) - y_bar
    ops.patch('rect', mat_steel, nf_web, 1,
              yI_w_bot, -tw / 2.0,
              yI_w_top, tw / 2.0)

    # Top flange
    yI_tf = (d - tf_top / 2.0) - y_bar
    ops.patch('rect', mat_steel, nf_flange, max(2, int(bf_top / 2)),
              yI_tf - tf_top / 2.0, -bf_top / 2.0,
              yI_tf + tf_top / 2.0, bf_top / 2.0)

    # Concrete slab
    yI_slab_bot = (d + haunch) - y_bar
    yI_slab_top = (d + haunch + slab_thick) - y_bar
    nf_slab_z = max(4, int(slab_width / 6))
    ops.patch('rect', mat_concrete, nf_slab, nf_slab_z,
              yI_slab_bot, -slab_width / 2.0,
              yI_slab_top, slab_width / 2.0)

    return tag


# ============================================================================
# CIRCULAR RC SECTION
# ============================================================================

def circular_rc_section(tag: int, diameter: float, cover: float,
                        num_bars: int, bar_area: float,
                        mat_confined: int, mat_unconfined: int,
                        mat_steel: int,
                        n_core_circ: int = 16, n_core_rad: int = 8,
                        n_cover_circ: int = 16, n_cover_rad: int = 2) -> int:
    """Create fiber section for a circular reinforced concrete column.

    Divides the section into:
    1. Core (confined concrete): inside the reinforcing cage
    2. Cover (unconfined concrete): outside the cage to surface
    3. Reinforcing steel: single layer of bars on a circle

    Args:
        tag:             Section tag.
        diameter:        Outer diameter (inches).
        cover:           Clear cover to outside of hoop/spiral (inches).
        num_bars:        Number of longitudinal bars.
        bar_area:        Area of each bar (in²). e.g., #8 = 0.79 in².
        mat_confined:    Material tag for confined concrete (core).
        mat_unconfined:  Material tag for unconfined concrete (cover).
        mat_steel:       Material tag for reinforcing steel.
        n_core_circ:     Circumferential divisions for core. Default: 16.
        n_core_rad:      Radial divisions for core. Default: 8.
        n_cover_circ:    Circumferential divisions for cover. Default: 16.
        n_cover_rad:     Radial divisions for cover. Default: 2.

    Returns:
        Section tag.

    Reference:
        AASHTO LRFD 5.6.4: Compression Members.
        Caltrans SDC Section 3.7: Concrete Column Design.
    """
    radius = diameter / 2.0
    core_radius = radius - cover  # radius to center of hoop

    ops.section('Fiber', tag)

    # Core concrete: circular patch from center to core_radius
    ops.patch('circ', mat_confined, n_core_circ, n_core_rad,
              0.0, 0.0, 0.0, core_radius, 0.0, 360.0)

    # Cover concrete: annular ring from core_radius to outer radius
    ops.patch('circ', mat_unconfined, n_cover_circ, n_cover_rad,
              0.0, 0.0, core_radius, radius, 0.0, 360.0)

    # Reinforcing steel: single circular layer
    bar_radius = radius - cover - 0.5  # approximate to center of bar
    # Ensure bar_radius doesn't go negative
    bar_radius = max(bar_radius, core_radius * 0.8)
    ops.layer('circ', mat_steel, num_bars, bar_area,
              0.0, 0.0, bar_radius, 0.0, 360.0)

    return tag


# ============================================================================
# RECTANGULAR RC SECTION
# ============================================================================

@dataclass
class BarLayout:
    """Defines reinforcing bar layout for one face of a rectangular section.

    Attributes:
        num_bars: Number of bars on this face.
        bar_area: Area of each bar (in²).
        face:     "top", "bottom", "left", "right", or "corner".
    """
    num_bars: int
    bar_area: float
    face: str = "bottom"


def rectangular_rc_section(tag: int, width: float, height: float,
                           cover: float, bars_layout: List[BarLayout],
                           mat_confined: int, mat_unconfined: int,
                           mat_steel: int,
                           nfy_core: int = 10, nfz_core: int = 10,
                           nf_cover: int = 2) -> int:
    """Create fiber section for a rectangular reinforced concrete column.

    Divides the section into confined core, four cover patches (top, bottom,
    left, right), and reinforcing bar layers.

    Args:
        tag:            Section tag.
        width:          Section width, z-direction (inches).
        height:         Section height, y-direction (inches).
        cover:          Clear cover (inches).
        bars_layout:    List of BarLayout objects defining reinforcement.
        mat_confined:   Material tag for confined concrete.
        mat_unconfined: Material tag for unconfined concrete.
        mat_steel:      Material tag for reinforcing steel.
        nfy_core:       Fiber divisions in y for core. Default: 10.
        nfz_core:       Fiber divisions in z for core. Default: 10.
        nf_cover:       Fiber divisions through cover thickness. Default: 2.

    Returns:
        Section tag.

    Reference:
        AASHTO LRFD 5.6.4: Compression Members.
        ACI 318-19 Section 22.4: Axial Strength.
    """
    hw = height / 2.0
    ww = width / 2.0
    c = cover

    # Core boundaries (inside cover)
    core_y_top = hw - c
    core_y_bot = -hw + c
    core_z_left = -ww + c
    core_z_right = ww - c

    ops.section('Fiber', tag)

    # Confined core
    ops.patch('rect', mat_confined, nfy_core, nfz_core,
              core_y_bot, core_z_left, core_y_top, core_z_right)

    # Cover patches (unconfined)
    nf_z_cover = max(2, int(width / 3))
    nf_y_cover = max(2, int(height / 3))

    # Bottom cover
    ops.patch('rect', mat_unconfined, nf_cover, nf_z_cover,
              -hw, -ww, core_y_bot, ww)
    # Top cover
    ops.patch('rect', mat_unconfined, nf_cover, nf_z_cover,
              core_y_top, -ww, hw, ww)
    # Left cover (between core top and bottom)
    ops.patch('rect', mat_unconfined, nf_y_cover, nf_cover,
              core_y_bot, -ww, core_y_top, core_z_left)
    # Right cover
    ops.patch('rect', mat_unconfined, nf_y_cover, nf_cover,
              core_y_bot, core_z_right, core_y_top, ww)

    # Reinforcing bars
    bar_offset = cover + 0.5  # approximate to bar center
    for bl in bars_layout:
        if bl.face == "bottom":
            ops.layer('straight', mat_steel, bl.num_bars, bl.bar_area,
                      -hw + bar_offset, -ww + bar_offset,
                      -hw + bar_offset, ww - bar_offset)
        elif bl.face == "top":
            ops.layer('straight', mat_steel, bl.num_bars, bl.bar_area,
                      hw - bar_offset, -ww + bar_offset,
                      hw - bar_offset, ww - bar_offset)
        elif bl.face == "left":
            ops.layer('straight', mat_steel, bl.num_bars, bl.bar_area,
                      -hw + bar_offset, -ww + bar_offset,
                      hw - bar_offset, -ww + bar_offset)
        elif bl.face == "right":
            ops.layer('straight', mat_steel, bl.num_bars, bl.bar_area,
                      -hw + bar_offset, ww - bar_offset,
                      hw - bar_offset, ww - bar_offset)

    return tag


# ============================================================================
# BOX GIRDER SECTION
# ============================================================================

@dataclass
class TendonProfile:
    """Tendon location within a section.

    Attributes:
        y:     Vertical position from section centroid (inches).
        z:     Horizontal position from section center (inches).
        area:  Tendon area (in²).
    """
    y: float
    z: float
    area: float


def box_girder_section(tag: int, depth: float, top_width: float,
                       bot_width: float, top_thick: float,
                       bot_thick: float, web_thick: float,
                       num_cells: int, mat_concrete: int,
                       tendons: Optional[List[TendonProfile]] = None,
                       mat_strand: Optional[int] = None,
                       nf_slab: int = 4, nf_web: int = 8) -> int:
    """Create fiber section for a post-tensioned concrete box girder.

    Models single or multi-cell box with top slab, bottom slab, and webs.
    Optional tendons modeled as steel layers.

    Args:
        tag:          Section tag.
        depth:        Total depth (inches).
        top_width:    Top slab width (inches).
        bot_width:    Bottom slab width (inches).
        top_thick:    Top slab thickness (inches).
        bot_thick:    Bottom slab thickness (inches).
        web_thick:    Individual web thickness (inches).
        num_cells:    Number of cells (1 for single-cell, etc.).
        mat_concrete: Material tag for concrete.
        tendons:      Optional list of TendonProfile for PT strands.
        mat_strand:   Material tag for tendons. Required if tendons provided.
        nf_slab:      Fibers through slab thickness. Default: 4.
        nf_web:       Fibers along web depth. Default: 8.

    Returns:
        Section tag.

    Reference:
        AASHTO LRFD 5.12.2: Segmental Concrete Bridges.
        AASHTO LRFD 4.6.2.6.2: Effective Flange Width for Box Girders.
    """
    # Number of webs = num_cells + 1
    num_webs = num_cells + 1
    web_depth = depth - top_thick - bot_thick

    # Centroid estimate (from bottom)
    A_top = top_width * top_thick
    A_bot = bot_width * bot_thick
    A_webs = num_webs * web_thick * web_depth
    y_top = depth - top_thick / 2.0
    y_bot = bot_thick / 2.0
    y_web = bot_thick + web_depth / 2.0
    A_total = A_top + A_bot + A_webs
    y_bar = (A_top * y_top + A_bot * y_bot + A_webs * y_web) / A_total

    ops.section('Fiber', tag)

    # Top slab
    nfz_top = max(4, int(top_width / 6))
    ops.patch('rect', mat_concrete, nf_slab, nfz_top,
              (depth - top_thick) - y_bar, -top_width / 2.0,
              depth - y_bar, top_width / 2.0)

    # Bottom slab
    nfz_bot = max(4, int(bot_width / 6))
    ops.patch('rect', mat_concrete, nf_slab, nfz_bot,
              -y_bar, -bot_width / 2.0,
              bot_thick - y_bar, bot_width / 2.0)

    # Webs — distribute across width
    web_spacing = (bot_width - web_thick) / max(1, num_webs - 1) if num_webs > 1 else 0
    web_start_z = -bot_width / 2.0 + web_thick / 2.0

    for i in range(num_webs):
        if num_webs > 1:
            z_center = web_start_z + i * web_spacing
        else:
            z_center = 0.0

        y_bot_web = bot_thick - y_bar
        y_top_web = (depth - top_thick) - y_bar

        ops.patch('rect', mat_concrete, nf_web, 1,
                  y_bot_web, z_center - web_thick / 2.0,
                  y_top_web, z_center + web_thick / 2.0)

    # Tendons
    if tendons and mat_strand is not None:
        for t in tendons:
            # Single fiber per tendon
            ops.fiber(t.y - y_bar + depth / 2.0, t.z, t.area, mat_strand)

    return tag


# ============================================================================
# PRESTRESSED I-SECTION
# ============================================================================

@dataclass
class StrandPattern:
    """Strand pattern for prestressed girder.

    Attributes:
        rows: List of (y_from_bottom, num_strands) tuples.
        strand_area: Area per strand (in²). Default: 0.217 in² (0.6" dia).
        debond: Optional dict of {row_index: debond_length_inches}.
    """
    rows: List[Tuple[float, int]]
    strand_area: float = 0.217  # 0.6" diameter strand
    debond: Optional[Dict[int, float]] = None


def prestressed_i_section(tag: int, girder_type: str,
                          mat_concrete: int,
                          strand_pattern: StrandPattern,
                          mat_strand: int,
                          nf_flange: int = 4, nf_web: int = 8) -> int:
    """Create fiber section for a prestressed concrete I-girder.

    Supports AASHTO, BT (Bulb-Tee), and NU (Nebraska University) girder types.
    Girder dimensions are looked up from the built-in GIRDER_LIBRARY.

    Args:
        tag:            Section tag.
        girder_type:    Key into GIRDER_LIBRARY (e.g., "BT_72", "AASHTO_IV").
        mat_concrete:   Material tag for concrete.
        strand_pattern: StrandPattern defining row positions and counts.
        mat_strand:     Material tag for prestressing strand.
        nf_flange:      Fibers through flange thickness. Default: 4.
        nf_web:         Fibers along web depth. Default: 8.

    Returns:
        Section tag.

    Raises:
        KeyError: If girder_type not found in GIRDER_LIBRARY.

    Reference:
        PCI Bridge Design Manual 3rd Ed, Chapter 3.
        AASHTO LRFD 5.9: Prestressed Concrete.
    """
    if girder_type not in GIRDER_LIBRARY:
        available = ", ".join(sorted(GIRDER_LIBRARY.keys()))
        raise KeyError(
            f"Girder type '{girder_type}' not found. Available: {available}"
        )

    g = GIRDER_LIBRARY[girder_type]
    d = g['d']
    bt = g['bt']      # top flange width
    bb = g['bb']      # bottom flange width
    tw = g['tw']       # web thickness
    tf_top = g['tf_top']
    tf_bot = g['tf_bot']
    d_web = d - tf_top - tf_bot

    # Compute centroid (from bottom)
    A_bf = bb * tf_bot
    A_w = d_web * tw
    A_tf = bt * tf_top

    y_bf = tf_bot / 2.0
    y_w = tf_bot + d_web / 2.0
    y_tf = d - tf_top / 2.0

    A_total = A_bf + A_w + A_tf
    y_bar = (A_bf * y_bf + A_w * y_w + A_tf * y_tf) / A_total

    ops.section('Fiber', tag)

    # Bottom flange
    nfz_bf = max(2, int(bb / 4))
    ops.patch('rect', mat_concrete, nf_flange, nfz_bf,
              -y_bar, -bb / 2.0,
              tf_bot - y_bar, bb / 2.0)

    # Web
    ops.patch('rect', mat_concrete, nf_web, 1,
              tf_bot - y_bar, -tw / 2.0,
              (d - tf_top) - y_bar, tw / 2.0)

    # Top flange
    nfz_tf = max(2, int(bt / 4))
    ops.patch('rect', mat_concrete, nf_flange, nfz_tf,
              (d - tf_top) - y_bar, -bt / 2.0,
              d - y_bar, bt / 2.0)

    # Strands
    for y_from_bot, num_strands in strand_pattern.rows:
        y_centroidal = y_from_bot - y_bar
        # Distribute strands across bottom flange width
        z_spread = bb * 0.7  # 70% of bottom flange width
        if num_strands > 0:
            ops.layer('straight', mat_strand, num_strands,
                      strand_pattern.strand_area,
                      y_centroidal, -z_spread / 2.0,
                      y_centroidal, z_spread / 2.0)

    return tag

"""Substructure modeling tool for OpenSees FEA models.

Creates columns, bents, wall piers, pile bents, and abutments from
engineering parameters. Generates OpenSees nodes, elements, sections,
and materials for nonlinear analysis.

Supported types:
    - Single Column (circular or rectangular, dispBeamColumn)
    - Multi-Column Bent (columns + cap beam)
    - Wall Pier (solid rectangular)
    - Pile Bent (piles above ground + cap beam)
    - Integral Abutment (monolithic with backfill springs)
    - Stub/Seat Abutment (conventional with bearing seats)

Key engineering:
    - P-Δ geometric transform always for columns
    - Plastic hinge zone mesh refinement (Paulay & Priestley Lp formula)
    - Mander confined concrete (auto-computed from transverse steel)
    - Cracked section stiffness flags (0.5 EIg columns, 0.35 EIg caps)
    - Minimum 6 elements per column, 2 per plastic hinge zone

Units: kip-inch-second internally. Accepts ft input, converts internally.

References:
    - AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    - AASHTO Guide Specifications for LRFD Seismic Bridge Design, 2nd Ed (2011)
    - Paulay, T. & Priestley, M.J.N. (1992). Seismic Design of RC and
      Masonry Buildings. Wiley.
    - Mander, J.B., Priestley, M.J.N., Park, R. (1988). Theoretical
      Stress-Strain Model for Confined Concrete.
    - ACI 318-19: Building Code Requirements for Structural Concrete.
    - Caltrans Seismic Design Criteria v2.0 (2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FT_TO_IN = 12.0
KSF_TO_KSI = 1.0 / 144.0
PCF_TO_PCI = 1.0 / 1728.0

# Reinforcing bar areas (in²) — ASTM standard
REBAR_AREAS: dict[str, float] = {
    "#3": 0.11, "#4": 0.20, "#5": 0.31, "#6": 0.44, "#7": 0.60,
    "#8": 0.79, "#9": 1.00, "#10": 1.27, "#11": 1.56, "#14": 2.25,
    "#18": 4.00,
}

REBAR_DIAMETERS: dict[str, float] = {
    "#3": 0.375, "#4": 0.500, "#5": 0.625, "#6": 0.750, "#7": 0.875,
    "#8": 1.000, "#9": 1.128, "#10": 1.270, "#11": 1.410, "#14": 1.693,
    "#18": 2.257,
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnShape(str, Enum):
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"


class SubstructureType(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN_BENT = "multi_column_bent"
    WALL_PIER = "wall_pier"
    PILE_BENT = "pile_bent"
    INTEGRAL_ABUTMENT = "integral_abutment"
    SEAT_ABUTMENT = "seat_abutment"


# ---------------------------------------------------------------------------
# Data classes — inputs
# ---------------------------------------------------------------------------

@dataclass
class ColumnConfig:
    """Column definition.

    Attributes:
        shape: "circular" or "rectangular".
        diameter_in: Diameter for circular columns (inches).
        width_in: Width for rectangular columns (inches, along bridge).
        depth_in: Depth for rectangular columns (inches, transverse).
        height_ft: Column clear height (ft). Converted internally to inches.
        fc_ksi: Concrete compressive strength (ksi).
        fy_ksi: Longitudinal rebar yield strength (ksi).
        num_bars: Number of longitudinal bars.
        bar_size: Bar designation, e.g. "#8".
        cover_in: Clear cover to hoops/spirals (inches).
        fy_transverse_ksi: Transverse steel yield strength (ksi).
        rho_s: Volumetric ratio of transverse reinforcement.
        cracked: Whether to flag for cracked section stiffness (0.5 EIg).
    """
    shape: str = "circular"
    diameter_in: float = 48.0
    width_in: float = 48.0
    depth_in: float = 48.0
    height_ft: float = 20.0
    fc_ksi: float = 4.0
    fy_ksi: float = 60.0
    num_bars: int = 16
    bar_size: str = "#8"
    cover_in: float = 2.0
    fy_transverse_ksi: float = 60.0
    rho_s: float = 0.01
    cracked: bool = True

    @property
    def height_in(self) -> float:
        return self.height_ft * FT_TO_IN

    @property
    def bar_area(self) -> float:
        return REBAR_AREAS.get(self.bar_size, 0.79)

    @property
    def bar_diameter(self) -> float:
        return REBAR_DIAMETERS.get(self.bar_size, 1.0)


@dataclass
class CapBeamConfig:
    """Cap beam definition.

    Attributes:
        width_in: Cap width along bridge (inches).
        depth_in: Cap depth (inches).
        fc_ksi: Concrete f'c (ksi).
        fy_ksi: Rebar yield strength (ksi).
        num_bars_top: Top bars count.
        num_bars_bot: Bottom bars count.
        bar_size: Bar designation.
        cover_in: Clear cover (inches).
        cracked: Whether to flag for cracked stiffness (0.35 EIg).
    """
    width_in: float = 60.0
    depth_in: float = 48.0
    fc_ksi: float = 4.0
    fy_ksi: float = 60.0
    num_bars_top: int = 8
    num_bars_bot: int = 8
    bar_size: str = "#9"
    cover_in: float = 2.0
    cracked: bool = True


@dataclass
class WallPierConfig:
    """Wall pier definition.

    Attributes:
        height_ft: Pier height (ft).
        width_in: Width along bridge axis (inches).
        thickness_in: Thickness transverse to bridge (inches).
        fc_ksi: Concrete f'c (ksi).
        fy_ksi: Rebar yield strength (ksi).
        num_bars_face: Bars per face (width direction).
        bar_size: Bar designation.
        cover_in: Clear cover (inches).
        fy_transverse_ksi: Transverse steel yield (ksi).
        rho_s: Volumetric transverse steel ratio.
    """
    height_ft: float = 25.0
    width_in: float = 240.0  # 20 ft
    thickness_in: float = 48.0  # 4 ft
    fc_ksi: float = 4.0
    fy_ksi: float = 60.0
    num_bars_face: int = 20
    bar_size: str = "#9"
    cover_in: float = 2.0
    fy_transverse_ksi: float = 60.0
    rho_s: float = 0.006

    @property
    def height_in(self) -> float:
        return self.height_ft * FT_TO_IN

    @property
    def bar_area(self) -> float:
        return REBAR_AREAS.get(self.bar_size, 1.0)

    @property
    def bar_diameter(self) -> float:
        return REBAR_DIAMETERS.get(self.bar_size, 1.128)

    @property
    def is_slender(self) -> bool:
        """Wall is slender if h/t > 25 per ACI 318."""
        return self.height_in / self.thickness_in > 25.0


@dataclass
class PileBentConfig:
    """Pile bent configuration.

    Attributes:
        pile_type: "HP", "pipe", "precast" (string tag for downstream).
        pile_diameter_in: Pile outer diameter or depth (inches).
        pile_wall_thickness_in: For pipe piles (inches).
        pile_count: Number of piles.
        spacing_ft: Center-to-center pile spacing (ft).
        free_height_ft: Height above ground to cap bottom (ft).
        cap: Cap beam configuration.
    """
    pile_type: str = "HP"
    pile_diameter_in: float = 14.0
    pile_wall_thickness_in: float = 0.5
    pile_count: int = 5
    spacing_ft: float = 6.0
    free_height_ft: float = 10.0
    cap: CapBeamConfig = field(default_factory=CapBeamConfig)


@dataclass
class IntegralAbutmentConfig:
    """Integral abutment configuration.

    Attributes:
        backwall_height_ft: Height from seat to top of backwall (ft).
        seat_width_in: Seat width along bridge (inches).
        wingwall_length_ft: Wingwall length (ft).
        skew_deg: Skew angle (degrees).
        backfill_gamma_pcf: Backfill unit weight (pcf).
        backfill_phi_deg: Backfill friction angle (degrees).
        fc_ksi: Concrete f'c for backwall (ksi).
        num_springs: Number of backfill springs behind backwall.
    """
    backwall_height_ft: float = 6.0
    seat_width_in: float = 36.0
    wingwall_length_ft: float = 15.0
    skew_deg: float = 0.0
    backfill_gamma_pcf: float = 120.0
    backfill_phi_deg: float = 34.0
    fc_ksi: float = 4.0
    num_springs: int = 5

    @property
    def backwall_height_in(self) -> float:
        return self.backwall_height_ft * FT_TO_IN


@dataclass
class SeatAbutmentConfig:
    """Stub/seat abutment configuration.

    Attributes:
        seat_width_in: Seat width along bridge (inches).
        backwall_height_ft: Height above seat (ft).
        bearing_locations_in: Transverse locations for bearings (inches from CL).
        fc_ksi: Concrete f'c (ksi).
    """
    seat_width_in: float = 36.0
    backwall_height_ft: float = 5.0
    bearing_locations_in: list[float] = field(default_factory=lambda: [-36.0, 36.0])
    fc_ksi: float = 4.0

    @property
    def backwall_height_in(self) -> float:
        return self.backwall_height_ft * FT_TO_IN


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class SubstructureModel:
    """Complete substructure model output.

    All tags are integers for direct OpenSees use.
    All coordinates in kip-inch-second.
    """
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    materials: list[dict] = field(default_factory=list)
    top_nodes: list[int] = field(default_factory=list)
    base_nodes: list[int] = field(default_factory=list)
    cap_nodes: list[int] = field(default_factory=list)
    plastic_hinge_elements: list[int] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    springs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    substructure_type: str = ""
    cracked_stiffness: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"SubstructureModel ({self.substructure_type}): "
            f"{len(self.nodes)} nodes, {len(self.elements)} elements, "
            f"{len(self.sections)} sections, {len(self.materials)} materials | "
            f"top={self.top_nodes}, base={self.base_nodes}"
        )


# ---------------------------------------------------------------------------
# Tag allocator
# ---------------------------------------------------------------------------

class TagAllocator:
    """Sequential tag allocator for OpenSees objects."""

    def __init__(self, start: int = 1):
        self._next = start

    def next(self, count: int = 1) -> int | list[int]:
        if count == 1:
            tag = self._next
            self._next += 1
            return tag
        tags = list(range(self._next, self._next + count))
        self._next += count
        return tags

    @property
    def current(self) -> int:
        return self._next


# ---------------------------------------------------------------------------
# Plastic hinge length
# ---------------------------------------------------------------------------

def plastic_hinge_length(L_in: float, fy_ksi: float, db_in: float) -> float:
    """Compute plastic hinge length per Paulay & Priestley (1992).

    Lp = 0.08L + 0.15 * fy * db

    Args:
        L_in: Member clear height (inches).
        fy_ksi: Rebar yield strength (ksi).
        db_in: Longitudinal bar diameter (inches).

    Returns:
        Plastic hinge length Lp (inches).

    Reference:
        Paulay & Priestley (1992), Eq. 4.30.
        Caltrans SDC §5.3.4.
    """
    return 0.08 * L_in + 0.15 * fy_ksi * db_in


# ---------------------------------------------------------------------------
# Mesh generation — column
# ---------------------------------------------------------------------------

def _column_mesh_lengths(
    height_in: float,
    Lp: float,
    min_elements: int = 6,
    min_ph_elements: int = 2,
) -> tuple[list[float], list[bool]]:
    """Generate element lengths along column height with plastic hinge refinement.

    Places refined elements at top and bottom plastic hinge zones,
    uniform elements in the middle elastic region.

    Args:
        height_in: Total column height (inches).
        Lp: Plastic hinge length (inches).
        min_elements: Minimum total elements. Default: 6.
        min_ph_elements: Minimum elements per plastic hinge zone. Default: 2.

    Returns:
        Tuple of (lengths, is_ph) where:
            lengths: List of element lengths (inches).
            is_ph: List of booleans indicating plastic hinge zone.
    """
    # Clamp Lp so two PH zones don't exceed column height
    Lp = min(Lp, height_in * 0.4)

    # Elements in each PH zone
    ph_len = Lp / min_ph_elements

    # Elastic region
    elastic_len = height_in - 2.0 * Lp
    n_elastic = max(min_elements - 2 * min_ph_elements, 2)
    el_len = elastic_len / n_elastic

    lengths: list[float] = []
    is_ph: list[bool] = []

    # Bottom PH zone
    for _ in range(min_ph_elements):
        lengths.append(ph_len)
        is_ph.append(True)

    # Elastic zone
    for _ in range(n_elastic):
        lengths.append(el_len)
        is_ph.append(False)

    # Top PH zone
    for _ in range(min_ph_elements):
        lengths.append(ph_len)
        is_ph.append(True)

    return lengths, is_ph


# ---------------------------------------------------------------------------
# Mander confinement helper
# ---------------------------------------------------------------------------

def _mander_confinement(
    fc_ksi: float,
    fy_trans_ksi: float,
    rho_s: float,
    config: str = "circular",
) -> tuple[float, float, float]:
    """Compute confined concrete parameters per Mander et al. (1988).

    Returns (fcc, ecc, ecu).
    """
    ke = 0.95 if config == "circular" else 0.75
    fl = 0.5 * ke * rho_s * fy_trans_ksi

    ratio = fl / fc_ksi if fc_ksi > 0 else 0
    fcc = fc_ksi * (-1.254 + 2.254 * math.sqrt(1.0 + 7.94 * ratio) - 2.0 * ratio)

    eps_co = 0.002
    ecc = eps_co * (1.0 + 5.0 * (fcc / fc_ksi - 1.0))

    eps_su = 0.09  # Grade 60 ultimate strain
    ecu = 0.004 + 1.4 * rho_s * fy_trans_ksi * eps_su / fcc

    return fcc, ecc, ecu


# ---------------------------------------------------------------------------
# Backfill passive pressure
# ---------------------------------------------------------------------------

def _backfill_spring_params(
    height_in: float,
    width_in: float,
    gamma_pci: float,
    phi_deg: float,
    num_springs: int,
) -> list[dict]:
    """Compute nonlinear backfill spring parameters for integral abutments.

    Uses log-spiral passive pressure theory (simplified Rankine Kp)
    distributed over the backwall height.

    Each spring uses a HyperbolicGap-like formulation:
        F = Fult * y / (y + y_ref)  for y > 0 (compression only)

    Args:
        height_in: Backwall height (inches).
        width_in: Backwall width (inches, tributary per spring).
        gamma_pci: Backfill unit weight (kip/in³).
        phi_deg: Backfill friction angle (degrees).
        num_springs: Number of springs.

    Returns:
        List of dicts with {depth_in, Fult, y50, Kini, spring_type}.
    """
    phi_rad = math.radians(phi_deg)
    # Rankine passive pressure coefficient
    Kp = (1.0 + math.sin(phi_rad)) / (1.0 - math.sin(phi_rad))

    trib_height = height_in / num_springs
    springs = []

    for i in range(num_springs):
        depth = (i + 0.5) * trib_height  # depth from top of wall

        # Passive pressure at this depth: pp = Kp * gamma * depth
        pp_ksi = Kp * gamma_pci * depth  # ksi (stress)

        # Force: pp * trib_area
        trib_area = trib_height * width_in  # in²
        Fult = pp_ksi * trib_area  # kip
        Fult = max(Fult, 0.001)

        # y50: displacement at 50% mobilization (~0.01H to 0.04H typical)
        y50 = 0.02 * height_in  # 2% of wall height
        y50 = max(y50, 0.1)

        # Initial stiffness: Kini ~ 2 * Fult / y50
        Kini = 2.0 * Fult / y50

        springs.append({
            "depth_in": depth,
            "Fult_kip": Fult,
            "y50_in": y50,
            "Kini_kip_per_in": Kini,
            "spring_type": "backfill_passive",
        })

    return springs


# ===================================================================
# BUILDERS
# ===================================================================

def build_single_column(
    col: ColumnConfig,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    tag_start: int = 1,
) -> SubstructureModel:
    """Build a single column substructure model.

    Creates a dispBeamColumn column with:
    - Confined core + unconfined cover (Mander model)
    - Steel02 reinforcing steel
    - PDelta geometric transform
    - Plastic hinge zone mesh refinement

    Args:
        col: ColumnConfig with column parameters.
        base_x, base_y, base_z: Base node coordinates (inches).
        tag_start: Starting tag for all OpenSees objects.

    Returns:
        SubstructureModel with nodes, elements, sections, materials.
    """
    tags = TagAllocator(tag_start)
    model = SubstructureModel(substructure_type="single_column")

    height = col.height_in
    config = "circular" if col.shape == "circular" else "rectangular"

    # --- Materials ---
    # Unconfined concrete (cover)
    mat_unconf = tags.next()
    model.materials.append({
        "tag": mat_unconf,
        "type": "Concrete01",
        "name": "unconfined_concrete",
        "fc_ksi": col.fc_ksi,
    })

    # Confined concrete (core)
    fcc, ecc, ecu = _mander_confinement(
        col.fc_ksi, col.fy_transverse_ksi, col.rho_s, config
    )
    mat_conf = tags.next()
    model.materials.append({
        "tag": mat_conf,
        "type": "Concrete01",
        "name": "confined_concrete",
        "fcc_ksi": fcc,
        "ecc": ecc,
        "ecu": ecu,
    })

    # Reinforcing steel
    mat_steel = tags.next()
    model.materials.append({
        "tag": mat_steel,
        "type": "Steel02",
        "name": "reinforcing_steel",
        "fy_ksi": col.fy_ksi,
    })

    # --- Section ---
    sec_tag = tags.next()
    if col.shape == "circular":
        model.sections.append({
            "tag": sec_tag,
            "type": "circular_rc",
            "diameter_in": col.diameter_in,
            "cover_in": col.cover_in,
            "num_bars": col.num_bars,
            "bar_area": col.bar_area,
            "mat_confined": mat_conf,
            "mat_unconfined": mat_unconf,
            "mat_steel": mat_steel,
        })
    else:
        model.sections.append({
            "tag": sec_tag,
            "type": "rectangular_rc",
            "width_in": col.width_in,
            "depth_in": col.depth_in,
            "cover_in": col.cover_in,
            "num_bars": col.num_bars,
            "bar_area": col.bar_area,
            "mat_confined": mat_conf,
            "mat_unconfined": mat_unconf,
            "mat_steel": mat_steel,
        })

    # --- Geometric transform (PDelta always) ---
    trans_tag = tags.next()
    model.materials.append({
        "tag": trans_tag,
        "type": "geomTransf",
        "transform": "PDelta",
    })

    # --- Mesh ---
    Lp = plastic_hinge_length(height, col.fy_ksi, col.bar_diameter)
    lengths, is_ph = _column_mesh_lengths(height, Lp)

    # --- Nodes ---
    base_node_tag = tags.next()
    model.nodes.append({
        "tag": base_node_tag, "x": base_x, "y": base_y, "z": base_z,
    })
    model.base_nodes.append(base_node_tag)

    cum_y = base_y
    prev_node = base_node_tag

    for i, (el_len, ph) in enumerate(zip(lengths, is_ph)):
        cum_y += el_len
        node_tag = tags.next()
        model.nodes.append({
            "tag": node_tag, "x": base_x, "y": cum_y, "z": base_z,
        })

        elem_tag = tags.next()
        model.elements.append({
            "tag": elem_tag,
            "type": "dispBeamColumn",
            "nodes": [prev_node, node_tag],
            "section": sec_tag,
            "transform": trans_tag,
            "integration": "Lobatto",
            "np": 5,
        })

        if ph:
            model.plastic_hinge_elements.append(elem_tag)

        prev_node = node_tag

    # Top node
    model.top_nodes.append(prev_node)

    # Cracked stiffness flags
    if col.cracked:
        model.cracked_stiffness = {"columns": 0.5, "note": "0.5*EIg per ACI 318"}

    return model


def build_multi_column_bent(
    num_columns: int,
    spacing_ft: float,
    col: ColumnConfig,
    cap: CapBeamConfig,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    tag_start: int = 1,
) -> SubstructureModel:
    """Build a multi-column bent with cap beam.

    Creates multiple columns at regular spacing with a cap beam connecting
    their tops. Rigid offsets at column-cap connections if cap depth exceeds
    column dimension.

    Args:
        num_columns: Number of columns.
        spacing_ft: Column center-to-center spacing (ft).
        col: Column configuration (shared by all columns).
        cap: Cap beam configuration.
        base_x, base_y, base_z: Origin (left column base).
        tag_start: Starting tag.

    Returns:
        SubstructureModel.
    """
    tags = TagAllocator(tag_start)
    model = SubstructureModel(substructure_type="multi_column_bent")

    spacing_in = spacing_ft * FT_TO_IN
    height = col.height_in
    config = "circular" if col.shape == "circular" else "rectangular"

    # --- Materials ---
    mat_unconf = tags.next()
    model.materials.append({
        "tag": mat_unconf, "type": "Concrete01",
        "name": "unconfined_concrete_col", "fc_ksi": col.fc_ksi,
    })

    fcc, ecc, ecu = _mander_confinement(
        col.fc_ksi, col.fy_transverse_ksi, col.rho_s, config,
    )
    mat_conf = tags.next()
    model.materials.append({
        "tag": mat_conf, "type": "Concrete01",
        "name": "confined_concrete_col", "fcc_ksi": fcc,
        "ecc": ecc, "ecu": ecu,
    })

    mat_steel = tags.next()
    model.materials.append({
        "tag": mat_steel, "type": "Steel02",
        "name": "reinforcing_steel", "fy_ksi": col.fy_ksi,
    })

    # Cap beam materials (unconfined only — cap is capacity-protected)
    mat_unconf_cap = tags.next()
    model.materials.append({
        "tag": mat_unconf_cap, "type": "Concrete01",
        "name": "unconfined_concrete_cap", "fc_ksi": cap.fc_ksi,
    })

    mat_steel_cap = tags.next()
    model.materials.append({
        "tag": mat_steel_cap, "type": "Steel02",
        "name": "reinforcing_steel_cap", "fy_ksi": cap.fy_ksi,
    })

    # --- Column section ---
    sec_col = tags.next()
    if col.shape == "circular":
        model.sections.append({
            "tag": sec_col, "type": "circular_rc",
            "diameter_in": col.diameter_in,
            "cover_in": col.cover_in,
            "num_bars": col.num_bars, "bar_area": col.bar_area,
            "mat_confined": mat_conf, "mat_unconfined": mat_unconf,
            "mat_steel": mat_steel,
        })
    else:
        model.sections.append({
            "tag": sec_col, "type": "rectangular_rc",
            "width_in": col.width_in, "depth_in": col.depth_in,
            "cover_in": col.cover_in,
            "num_bars": col.num_bars, "bar_area": col.bar_area,
            "mat_confined": mat_conf, "mat_unconfined": mat_unconf,
            "mat_steel": mat_steel,
        })

    # --- Cap section ---
    sec_cap = tags.next()
    cap_bar_area = REBAR_AREAS.get(cap.bar_size, 1.0)
    model.sections.append({
        "tag": sec_cap, "type": "rectangular_rc",
        "width_in": cap.width_in, "depth_in": cap.depth_in,
        "cover_in": cap.cover_in,
        "num_bars_top": cap.num_bars_top,
        "num_bars_bot": cap.num_bars_bot,
        "bar_area": cap_bar_area,
        "mat_unconfined": mat_unconf_cap, "mat_steel": mat_steel_cap,
    })

    # --- Transforms ---
    trans_col = tags.next()
    model.materials.append({
        "tag": trans_col, "type": "geomTransf", "transform": "PDelta",
    })
    trans_cap = tags.next()
    model.materials.append({
        "tag": trans_cap, "type": "geomTransf", "transform": "Linear",
    })

    # --- Rigid offset check ---
    col_dim = col.diameter_in if col.shape == "circular" else col.depth_in
    use_rigid_offset = cap.depth_in > col_dim

    # --- Build columns ---
    Lp = plastic_hinge_length(height, col.fy_ksi, col.bar_diameter)
    lengths, is_ph = _column_mesh_lengths(height, Lp)

    column_top_nodes: list[int] = []

    for c_idx in range(num_columns):
        z_col = base_z + c_idx * spacing_in

        # Base node
        bnode = tags.next()
        model.nodes.append({
            "tag": bnode, "x": base_x, "y": base_y, "z": z_col,
        })
        model.base_nodes.append(bnode)

        cum_y = base_y
        prev = bnode

        for el_len, ph in zip(lengths, is_ph):
            cum_y += el_len
            nnode = tags.next()
            model.nodes.append({"tag": nnode, "x": base_x, "y": cum_y, "z": z_col})

            etag = tags.next()
            model.elements.append({
                "tag": etag, "type": "dispBeamColumn",
                "nodes": [prev, nnode], "section": sec_col,
                "transform": trans_col, "integration": "Lobatto", "np": 5,
            })
            if ph:
                model.plastic_hinge_elements.append(etag)
            prev = nnode

        column_top_nodes.append(prev)

    # --- Cap beam ---
    # Cap is at column top elevation + half cap depth (if rigid offset)
    cap_y = base_y + height
    if use_rigid_offset:
        cap_y_center = cap_y + cap.depth_in / 2.0

        # Add rigid link nodes at cap centerline for each column
        cap_link_nodes: list[int] = []
        for c_idx, col_top in enumerate(column_top_nodes):
            z_col = base_z + c_idx * spacing_in
            cap_node = tags.next()
            model.nodes.append({
                "tag": cap_node, "x": base_x, "y": cap_y_center, "z": z_col,
            })
            cap_link_nodes.append(cap_node)

            # Rigid link from column top to cap centerline
            model.constraints.append({
                "type": "rigidLink",
                "master": col_top,
                "slave": cap_node,
                "dofs": [1, 2, 3, 4, 5, 6],
            })
    else:
        cap_link_nodes = column_top_nodes

    # Cap beam elements between column nodes
    n_cap_elem_per_span = 4  # elements between columns along cap
    for i in range(num_columns - 1):
        z_start = base_z + i * spacing_in
        z_end = base_z + (i + 1) * spacing_in
        node_start = cap_link_nodes[i]
        node_end = cap_link_nodes[i + 1]

        prev_cap = node_start
        for j in range(1, n_cap_elem_per_span):
            frac = j / n_cap_elem_per_span
            z_int = z_start + frac * (z_end - z_start)
            y_cap = cap_y if not use_rigid_offset else cap_y + cap.depth_in / 2.0

            int_node = tags.next()
            model.nodes.append({
                "tag": int_node, "x": base_x, "y": y_cap, "z": z_int,
            })
            model.cap_nodes.append(int_node)

            etag = tags.next()
            model.elements.append({
                "tag": etag, "type": "dispBeamColumn",
                "nodes": [prev_cap, int_node], "section": sec_cap,
                "transform": trans_cap, "integration": "Lobatto", "np": 5,
            })
            prev_cap = int_node

        # Last segment to next column
        etag = tags.next()
        model.elements.append({
            "tag": etag, "type": "dispBeamColumn",
            "nodes": [prev_cap, node_end], "section": sec_cap,
            "transform": trans_cap, "integration": "Lobatto", "np": 5,
        })

    # Cap nodes include the column-top cap nodes
    for cn in cap_link_nodes:
        if cn not in model.cap_nodes:
            model.cap_nodes.append(cn)

    # Top nodes = cap nodes (for bearing placement)
    model.top_nodes = list(model.cap_nodes)

    # Cracked stiffness
    if col.cracked or cap.cracked:
        model.cracked_stiffness = {
            "columns": 0.5 if col.cracked else 1.0,
            "cap_beam": 0.35 if cap.cracked else 1.0,
            "note": "Per ACI 318: 0.5EIg columns, 0.35EIg cap beams",
        }

    return model


def build_wall_pier(
    wall: WallPierConfig,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    tag_start: int = 1,
) -> SubstructureModel:
    """Build a solid wall pier model.

    Models as dispBeamColumn with rectangular fiber section.
    Checks for wall slenderness (h/t > 25).

    Args:
        wall: WallPierConfig.
        base_x, base_y, base_z: Base coordinates.
        tag_start: Starting tag.

    Returns:
        SubstructureModel.
    """
    tags = TagAllocator(tag_start)
    model = SubstructureModel(substructure_type="wall_pier")

    height = wall.height_in
    config = "rectangular"

    # --- Materials ---
    mat_unconf = tags.next()
    model.materials.append({
        "tag": mat_unconf, "type": "Concrete01",
        "name": "unconfined_concrete", "fc_ksi": wall.fc_ksi,
    })

    fcc, ecc, ecu = _mander_confinement(
        wall.fc_ksi, wall.fy_transverse_ksi, wall.rho_s, config,
    )
    mat_conf = tags.next()
    model.materials.append({
        "tag": mat_conf, "type": "Concrete01",
        "name": "confined_concrete", "fcc_ksi": fcc,
        "ecc": ecc, "ecu": ecu,
    })

    mat_steel = tags.next()
    model.materials.append({
        "tag": mat_steel, "type": "Steel02",
        "name": "reinforcing_steel", "fy_ksi": wall.fy_ksi,
    })

    # --- Section ---
    sec_tag = tags.next()
    model.sections.append({
        "tag": sec_tag, "type": "rectangular_rc",
        "width_in": wall.width_in,
        "depth_in": wall.thickness_in,
        "cover_in": wall.cover_in,
        "num_bars_face": wall.num_bars_face,
        "bar_area": wall.bar_area,
        "mat_confined": mat_conf, "mat_unconfined": mat_unconf,
        "mat_steel": mat_steel,
    })

    # --- Transform (PDelta) ---
    trans_tag = tags.next()
    model.materials.append({
        "tag": trans_tag, "type": "geomTransf", "transform": "PDelta",
    })

    # --- Mesh ---
    Lp = plastic_hinge_length(height, wall.fy_ksi, wall.bar_diameter)
    lengths, is_ph = _column_mesh_lengths(height, Lp)

    # --- Nodes and elements ---
    base_node = tags.next()
    model.nodes.append({"tag": base_node, "x": base_x, "y": base_y, "z": base_z})
    model.base_nodes.append(base_node)

    cum_y = base_y
    prev = base_node

    for el_len, ph in zip(lengths, is_ph):
        cum_y += el_len
        nnode = tags.next()
        model.nodes.append({"tag": nnode, "x": base_x, "y": cum_y, "z": base_z})

        etag = tags.next()
        model.elements.append({
            "tag": etag, "type": "dispBeamColumn",
            "nodes": [prev, nnode], "section": sec_tag,
            "transform": trans_tag, "integration": "Lobatto", "np": 5,
        })
        if ph:
            model.plastic_hinge_elements.append(etag)
        prev = nnode

    model.top_nodes.append(prev)

    # Slenderness check
    if wall.is_slender:
        model.warnings.append(
            f"SLENDER WALL: h/t = {wall.height_in / wall.thickness_in:.1f} > 25. "
            f"Second-order effects significant per ACI 318-19 §6.6.4."
        )

    return model


def build_pile_bent(
    config: PileBentConfig,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    tag_start: int = 1,
) -> SubstructureModel:
    """Build a pile bent with cap beam.

    Piles extend above ground (free height) with a cap beam connecting tops.
    Pile bases are output as base_nodes for connection to foundation springs.

    Args:
        config: PileBentConfig.
        base_x, base_y, base_z: Origin (first pile base).
        tag_start: Starting tag.

    Returns:
        SubstructureModel.
    """
    tags = TagAllocator(tag_start)
    model = SubstructureModel(substructure_type="pile_bent")

    spacing_in = config.spacing_ft * FT_TO_IN
    free_height_in = config.free_height_ft * FT_TO_IN
    cap = config.cap

    # --- Materials for piles (structural steel for HP/pipe) ---
    if config.pile_type in ("HP", "pipe"):
        mat_pile = tags.next()
        model.materials.append({
            "tag": mat_pile, "type": "Steel02",
            "name": "structural_steel_pile", "fy_ksi": 50.0,
        })
    else:
        # Precast concrete pile
        mat_pile = tags.next()
        model.materials.append({
            "tag": mat_pile, "type": "Concrete01",
            "name": "pile_concrete", "fc_ksi": 6.0,
        })

    # Pile section (elastic beam for simplicity — piles are capacity-protected)
    sec_pile = tags.next()
    if config.pile_type == "pipe":
        # Pipe pile section
        D = config.pile_diameter_in
        t = config.pile_wall_thickness_in
        A = math.pi * (D * t - t * t)
        I = math.pi / 64 * (D**4 - (D - 2 * t)**4)
        model.sections.append({
            "tag": sec_pile, "type": "pipe",
            "diameter_in": D, "wall_in": t, "A_in2": A, "I_in4": I,
        })
    else:
        # HP or precast: approximate as solid section
        D = config.pile_diameter_in
        A = D * D if config.pile_type == "HP" else math.pi * D**2 / 4
        I = D**4 / 12 if config.pile_type == "HP" else math.pi * D**4 / 64
        model.sections.append({
            "tag": sec_pile, "type": "solid",
            "diameter_in": D, "A_in2": A, "I_in4": I,
        })

    # Cap beam materials and section
    mat_cap_conc = tags.next()
    model.materials.append({
        "tag": mat_cap_conc, "type": "Concrete01",
        "name": "cap_concrete", "fc_ksi": cap.fc_ksi,
    })
    mat_cap_steel = tags.next()
    model.materials.append({
        "tag": mat_cap_steel, "type": "Steel02",
        "name": "cap_rebar", "fy_ksi": cap.fy_ksi,
    })

    sec_cap = tags.next()
    cap_bar_area = REBAR_AREAS.get(cap.bar_size, 1.0)
    model.sections.append({
        "tag": sec_cap, "type": "rectangular_rc",
        "width_in": cap.width_in, "depth_in": cap.depth_in,
        "cover_in": cap.cover_in,
        "num_bars_top": cap.num_bars_top, "num_bars_bot": cap.num_bars_bot,
        "bar_area": cap_bar_area,
    })

    # --- Transforms ---
    trans_pile = tags.next()
    model.materials.append({
        "tag": trans_pile, "type": "geomTransf", "transform": "PDelta",
    })
    trans_cap = tags.next()
    model.materials.append({
        "tag": trans_cap, "type": "geomTransf", "transform": "Linear",
    })

    # --- Build piles ---
    min_pile_elem = 4
    pile_el_len = free_height_in / min_pile_elem
    pile_top_nodes: list[int] = []

    for p_idx in range(config.pile_count):
        z_pile = base_z + p_idx * spacing_in

        # Base node (at ground level — connects to foundation)
        bnode = tags.next()
        model.nodes.append({"tag": bnode, "x": base_x, "y": base_y, "z": z_pile})
        model.base_nodes.append(bnode)

        prev = bnode
        cum_y = base_y

        for _ in range(min_pile_elem):
            cum_y += pile_el_len
            nnode = tags.next()
            model.nodes.append({"tag": nnode, "x": base_x, "y": cum_y, "z": z_pile})

            etag = tags.next()
            model.elements.append({
                "tag": etag, "type": "dispBeamColumn",
                "nodes": [prev, nnode], "section": sec_pile,
                "transform": trans_pile, "integration": "Lobatto", "np": 5,
            })
            prev = nnode

        pile_top_nodes.append(prev)

    # --- Cap beam ---
    cap_y = base_y + free_height_in
    n_cap_elem = 3

    for i in range(config.pile_count - 1):
        z_start = base_z + i * spacing_in
        z_end = base_z + (i + 1) * spacing_in
        prev_cap = pile_top_nodes[i]

        for j in range(1, n_cap_elem):
            frac = j / n_cap_elem
            z_int = z_start + frac * (z_end - z_start)
            int_node = tags.next()
            model.nodes.append({"tag": int_node, "x": base_x, "y": cap_y, "z": z_int})
            model.cap_nodes.append(int_node)

            etag = tags.next()
            model.elements.append({
                "tag": etag, "type": "dispBeamColumn",
                "nodes": [prev_cap, int_node], "section": sec_cap,
                "transform": trans_cap, "integration": "Lobatto", "np": 5,
            })
            prev_cap = int_node

        etag = tags.next()
        model.elements.append({
            "tag": etag, "type": "dispBeamColumn",
            "nodes": [prev_cap, pile_top_nodes[i + 1]], "section": sec_cap,
            "transform": trans_cap, "integration": "Lobatto", "np": 5,
        })

    # Cap nodes
    for ptn in pile_top_nodes:
        if ptn not in model.cap_nodes:
            model.cap_nodes.append(ptn)

    model.top_nodes = list(model.cap_nodes)

    return model


def build_integral_abutment(
    config: IntegralAbutmentConfig,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    tag_start: int = 1,
) -> SubstructureModel:
    """Build an integral abutment with backfill springs.

    Monolithic connection between superstructure and abutment.
    Passive pressure backfill springs model soil-structure interaction.

    Args:
        config: IntegralAbutmentConfig.
        base_x, base_y, base_z: Origin.
        tag_start: Starting tag.

    Returns:
        SubstructureModel with backfill springs.
    """
    tags = TagAllocator(tag_start)
    model = SubstructureModel(substructure_type="integral_abutment")

    height = config.backwall_height_in
    gamma_pci = config.backfill_gamma_pcf * PCF_TO_PCI

    # --- Base node (bottom of abutment) ---
    base_node = tags.next()
    model.nodes.append({"tag": base_node, "x": base_x, "y": base_y, "z": base_z})
    model.base_nodes.append(base_node)

    # --- Top node (top of backwall = connection to superstructure) ---
    top_node = tags.next()
    model.nodes.append({
        "tag": top_node, "x": base_x, "y": base_y + height, "z": base_z,
    })
    model.top_nodes.append(top_node)

    # --- Abutment wall element ---
    mat_conc = tags.next()
    model.materials.append({
        "tag": mat_conc, "type": "Concrete01",
        "name": "abutment_concrete", "fc_ksi": config.fc_ksi,
    })

    sec_tag = tags.next()
    model.sections.append({
        "tag": sec_tag, "type": "rectangular_rc",
        "width_in": config.seat_width_in,
        "depth_in": height,
    })

    trans_tag = tags.next()
    model.materials.append({
        "tag": trans_tag, "type": "geomTransf", "transform": "Linear",
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "dispBeamColumn",
        "nodes": [base_node, top_node], "section": sec_tag,
        "transform": trans_tag,
    })

    # --- Backfill springs ---
    spring_params = _backfill_spring_params(
        height_in=height,
        width_in=config.seat_width_in,
        gamma_pci=gamma_pci,
        phi_deg=config.backfill_phi_deg,
        num_springs=config.num_springs,
    )

    for i, sp in enumerate(spring_params):
        # Fixed backfill node
        sp_y = base_y + sp["depth_in"]
        fixed_node = tags.next()
        model.nodes.append({
            "tag": fixed_node, "x": base_x - 1.0, "y": sp_y, "z": base_z,
        })

        # Spring node on wall
        wall_node = tags.next()
        model.nodes.append({
            "tag": wall_node, "x": base_x, "y": sp_y, "z": base_z,
        })

        # Compression-only spring material (ENT)
        mat_sp = tags.next()
        model.materials.append({
            "tag": mat_sp, "type": "ENT",
            "name": f"backfill_spring_{i}",
            "k_kip_per_in": sp["Kini_kip_per_in"],
            "Fult_kip": sp["Fult_kip"],
        })

        # Zero-length spring element
        sp_elem = tags.next()
        model.springs.append({
            "tag": sp_elem, "type": "zeroLength",
            "nodes": [fixed_node, wall_node],
            "material": mat_sp, "direction": 1,
            "spring_type": "backfill_passive",
            "Fult_kip": sp["Fult_kip"],
            "y50_in": sp["y50_in"],
        })

    # Skew warning
    if config.skew_deg > 30:
        model.warnings.append(
            f"HIGH SKEW: {config.skew_deg}° — passive pressure reduction "
            f"and non-uniform loading effects should be considered."
        )

    return model


def build_seat_abutment(
    config: SeatAbutmentConfig,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    tag_start: int = 1,
) -> SubstructureModel:
    """Build a stub/seat abutment.

    Conventional abutment with discrete bearing seats. No backfill springs
    (joint separates backwall from bridge).

    Args:
        config: SeatAbutmentConfig.
        base_x, base_y, base_z: Origin.
        tag_start: Starting tag.

    Returns:
        SubstructureModel with bearing seat nodes.
    """
    tags = TagAllocator(tag_start)
    model = SubstructureModel(substructure_type="seat_abutment")

    height = config.backwall_height_in

    # --- Base node ---
    base_node = tags.next()
    model.nodes.append({"tag": base_node, "x": base_x, "y": base_y, "z": base_z})
    model.base_nodes.append(base_node)

    # --- Seat level node (top of stem) ---
    seat_y = base_y + height
    seat_node = tags.next()
    model.nodes.append({"tag": seat_node, "x": base_x, "y": seat_y, "z": base_z})

    # --- Bearing seat nodes at specified transverse locations ---
    for z_loc in config.bearing_locations_in:
        bn = tags.next()
        model.nodes.append({
            "tag": bn, "x": base_x, "y": seat_y, "z": base_z + z_loc,
        })
        model.top_nodes.append(bn)
        model.cap_nodes.append(bn)

    # Abutment stem element
    mat_conc = tags.next()
    model.materials.append({
        "tag": mat_conc, "type": "Concrete01",
        "name": "abutment_concrete", "fc_ksi": config.fc_ksi,
    })

    sec_tag = tags.next()
    model.sections.append({
        "tag": sec_tag, "type": "rectangular_rc",
        "width_in": config.seat_width_in,
        "depth_in": height,
    })

    trans_tag = tags.next()
    model.materials.append({
        "tag": trans_tag, "type": "geomTransf", "transform": "Linear",
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "dispBeamColumn",
        "nodes": [base_node, seat_node], "section": sec_tag,
        "transform": trans_tag,
    })

    return model


# ===================================================================
# TOP-LEVEL DISPATCHER
# ===================================================================

def create_substructure(
    sub_type: str,
    **kwargs: Any,
) -> SubstructureModel:
    """Create a substructure model from type string and parameters.

    This is the main entry point for the NLB pipeline.

    Args:
        sub_type: One of SubstructureType values.
        **kwargs: Parameters forwarded to the specific builder.

    Returns:
        SubstructureModel.

    Raises:
        ValueError: If sub_type is not recognized.
    """
    sub_type = sub_type.lower().replace(" ", "_").replace("-", "_")

    if sub_type == "single_column":
        col = kwargs.get("column", ColumnConfig(**{
            k: v for k, v in kwargs.items()
            if k in ColumnConfig.__dataclass_fields__
        }))
        return build_single_column(
            col=col,
            base_x=kwargs.get("base_x", 0.0),
            base_y=kwargs.get("base_y", 0.0),
            base_z=kwargs.get("base_z", 0.0),
            tag_start=kwargs.get("tag_start", 1),
        )

    elif sub_type == "multi_column_bent":
        col = kwargs.get("column", ColumnConfig(**{
            k: v for k, v in kwargs.items()
            if k in ColumnConfig.__dataclass_fields__
        }))
        cap = kwargs.get("cap", CapBeamConfig(**{
            k: v for k, v in kwargs.items()
            if k in CapBeamConfig.__dataclass_fields__
        }))
        return build_multi_column_bent(
            num_columns=kwargs.get("num_columns", 3),
            spacing_ft=kwargs.get("spacing_ft", 12.0),
            col=col,
            cap=cap,
            base_x=kwargs.get("base_x", 0.0),
            base_y=kwargs.get("base_y", 0.0),
            base_z=kwargs.get("base_z", 0.0),
            tag_start=kwargs.get("tag_start", 1),
        )

    elif sub_type == "wall_pier":
        wall = kwargs.get("wall", WallPierConfig(**{
            k: v for k, v in kwargs.items()
            if k in WallPierConfig.__dataclass_fields__
        }))
        return build_wall_pier(
            wall=wall,
            base_x=kwargs.get("base_x", 0.0),
            base_y=kwargs.get("base_y", 0.0),
            base_z=kwargs.get("base_z", 0.0),
            tag_start=kwargs.get("tag_start", 1),
        )

    elif sub_type == "pile_bent":
        config = kwargs.get("config", PileBentConfig(**{
            k: v for k, v in kwargs.items()
            if k in PileBentConfig.__dataclass_fields__
        }))
        return build_pile_bent(
            config=config,
            base_x=kwargs.get("base_x", 0.0),
            base_y=kwargs.get("base_y", 0.0),
            base_z=kwargs.get("base_z", 0.0),
            tag_start=kwargs.get("tag_start", 1),
        )

    elif sub_type == "integral_abutment":
        config = kwargs.get("config", IntegralAbutmentConfig(**{
            k: v for k, v in kwargs.items()
            if k in IntegralAbutmentConfig.__dataclass_fields__
        }))
        return build_integral_abutment(
            config=config,
            base_x=kwargs.get("base_x", 0.0),
            base_y=kwargs.get("base_y", 0.0),
            base_z=kwargs.get("base_z", 0.0),
            tag_start=kwargs.get("tag_start", 1),
        )

    elif sub_type == "seat_abutment":
        config = kwargs.get("config", SeatAbutmentConfig(**{
            k: v for k, v in kwargs.items()
            if k in SeatAbutmentConfig.__dataclass_fields__
        }))
        return build_seat_abutment(
            config=config,
            base_x=kwargs.get("base_x", 0.0),
            base_y=kwargs.get("base_y", 0.0),
            base_z=kwargs.get("base_z", 0.0),
            tag_start=kwargs.get("tag_start", 1),
        )

    else:
        raise ValueError(
            f"Unknown substructure type '{sub_type}'. "
            f"Supported: {[e.value for e in SubstructureType]}"
        )

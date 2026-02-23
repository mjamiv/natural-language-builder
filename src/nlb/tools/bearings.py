"""Bearing modeling tool for OpenSees FEA models.

Creates bearing elements connecting superstructure to substructure.
Supports elastomeric, pot, PTFE sliding, friction pendulum, integral,
and legacy rocker/roller bearings.

Key engineering:
    - Upper/lower bound analysis (temperature-dependent friction)
    - Compression-only behavior for all non-integral bearings
    - Velocity-dependent friction for PTFE and FP bearings
    - Bearing layout helper for multi-girder bridges

Units: kip-inch-second internally. Accepts ft input, converts internally.

References:
    - AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    - AASHTO LRFD §14.7: Elastomeric Bearings
    - AASHTO LRFD §14.7.2: PTFE Sliding Bearings
    - AASHTO Guide Specifications for Seismic Isolation Design, 4th Ed (2014)
    - Constantinou, M.C. et al. (2007). MCEER Report 07-0012:
      Performance of Seismic Isolation Hardware Under Service and
      Seismic Loading.
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
G_ACCEL = 386.4  # gravitational acceleration (in/s²)

# PTFE friction tables: (mu_slow, mu_fast) at reference temp ~68°F
# Source: AASHTO LRFD Table 14.7.2.5-1 / Constantinou (2007)
PTFE_FRICTION: dict[str, dict[str, float]] = {
    "unfilled": {
        "mu_slow": 0.03, "mu_fast": 0.05,
        "mu_slow_cold": 0.05, "mu_fast_cold": 0.08,  # at ~20°F
    },
    "glass_filled": {
        "mu_slow": 0.06, "mu_fast": 0.10,
        "mu_slow_cold": 0.09, "mu_fast_cold": 0.15,
    },
    "carbon_filled": {
        "mu_slow": 0.06, "mu_fast": 0.08,
        "mu_slow_cold": 0.09, "mu_fast_cold": 0.12,
    },
    "woven": {
        "mu_slow": 0.08, "mu_fast": 0.12,
        "mu_slow_cold": 0.12, "mu_fast_cold": 0.18,
    },
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BearingType(str, Enum):
    ELASTOMERIC = "elastomeric"
    POT_FIXED = "pot_fixed"
    POT_GUIDED = "pot_guided"
    PTFE_SLIDING = "ptfe_sliding"
    FRICTION_PENDULUM_SINGLE = "fp_single"
    FRICTION_PENDULUM_TRIPLE = "fp_triple"
    INTEGRAL = "integral"
    ROCKER_ROLLER = "rocker_roller"


# ---------------------------------------------------------------------------
# Data classes — inputs
# ---------------------------------------------------------------------------

@dataclass
class ElastomericConfig:
    """Steel-reinforced elastomeric bearing configuration.

    Attributes:
        length_in: Bearing length along bridge (inches).
        width_in: Bearing width transverse (inches).
        total_rubber_thickness_in: Sum of all rubber layers (inches).
        shear_modulus_ksi: Elastomer shear modulus G (ksi).
            AASHTO Table 14.7.6.2-1: 50 dur = 0.080-0.110 ksi,
            60 dur = 0.130-0.200 ksi.
        num_internal_layers: Number of internal rubber layers.
        layer_thickness_in: Individual rubber layer thickness (inches).
        steel_shim_thickness_in: Internal steel shim thickness (inches).
    """
    length_in: float = 14.0
    width_in: float = 9.0
    total_rubber_thickness_in: float = 2.5
    shear_modulus_ksi: float = 0.100
    num_internal_layers: int = 5
    layer_thickness_in: float = 0.50
    steel_shim_thickness_in: float = 0.105


@dataclass
class PotBearingConfig:
    """Pot bearing configuration.

    Attributes:
        vertical_capacity_kip: Vertical load capacity (kip).
        guide_direction: For guided: 1=longitudinal, 3=transverse, None=fixed.
        stiffness_kip_per_in: Horizontal stiffness for "fixed" direction.
    """
    vertical_capacity_kip: float = 500.0
    guide_direction: int | None = None  # None=fixed, 1=long, 3=trans
    stiffness_kip_per_in: float = 1.0e6  # essentially rigid


@dataclass
class PTFEConfig:
    """PTFE sliding bearing configuration.

    Attributes:
        vertical_capacity_kip: Vertical capacity (kip).
        ptfe_type: One of: "unfilled", "glass_filled", "carbon_filled", "woven".
        contact_pressure_ksi: Average contact pressure on PTFE (ksi).
    """
    vertical_capacity_kip: float = 500.0
    ptfe_type: str = "glass_filled"
    contact_pressure_ksi: float = 3.0


@dataclass
class FPSingleConfig:
    """Single friction pendulum bearing configuration.

    Attributes:
        radius_in: Radius of curvature R (inches).
        mu: Coefficient of friction at reference temperature.
        displacement_capacity_in: Maximum displacement (inches).
        vertical_capacity_kip: Vertical load capacity (kip).
    """
    radius_in: float = 40.0  # ~40" → T ≈ 2.0s
    mu: float = 0.06
    displacement_capacity_in: float = 8.0
    vertical_capacity_kip: float = 500.0


@dataclass
class FPTripleConfig:
    """Triple friction pendulum bearing configuration.

    Attributes:
        R1, R2, R3, R4: Radii of curvature (inches).
        mu1, mu2, mu3, mu4: Friction coefficients per surface.
        d1, d2, d3, d4: Displacement capacities per surface (inches).
        vertical_capacity_kip: Vertical load capacity (kip).
    """
    R1: float = 12.0
    R2: float = 88.0
    R3: float = 88.0
    R4: float = 12.0
    mu1: float = 0.012
    mu2: float = 0.052
    mu3: float = 0.052
    mu4: float = 0.012
    d1: float = 3.0
    d2: float = 12.0
    d3: float = 12.0
    d4: float = 3.0
    vertical_capacity_kip: float = 500.0


@dataclass
class RockerRollerConfig:
    """Rocker/roller (steel) bearing configuration.

    Attributes:
        vertical_capacity_kip: Vertical capacity (kip).
        rocker_radius_in: Rocker contact radius (inches).
        roller_diameter_in: Roller diameter (inches) — 0 if rocker only.
        steel_fy_ksi: Steel yield strength (ksi).
    """
    vertical_capacity_kip: float = 300.0
    rocker_radius_in: float = 6.0
    roller_diameter_in: float = 0.0
    steel_fy_ksi: float = 36.0


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class BearingModel:
    """Complete bearing model output.

    All tags are integers for direct OpenSees use.
    All coordinates in kip-inch-second.
    """
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)
    materials: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    top_nodes: list[int] = field(default_factory=list)
    bottom_nodes: list[int] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    cases: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    bearing_type: str = ""
    compression_only: bool = True

    def summary(self) -> str:
        return (
            f"BearingModel ({self.bearing_type}): "
            f"{len(self.nodes)} nodes, {len(self.elements)} elements, "
            f"{len(self.materials)} materials | "
            f"top={self.top_nodes}, bot={self.bottom_nodes}"
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


# ===================================================================
# ELASTOMERIC BEARING
# ===================================================================

def _elastomeric_stiffness(cfg: ElastomericConfig) -> dict:
    """Compute elastomeric bearing stiffness properties.

    Per AASHTO LRFD 14.7.5 and 14.7.6.

    Returns dict with Kh, Kv, shape_factor, rotation_capacity.
    """
    A = cfg.length_in * cfg.width_in  # plan area (in²)
    h_rt = cfg.total_rubber_thickness_in  # total rubber thickness
    G = cfg.shear_modulus_ksi

    # Shape factor S = loaded plan area / (perimeter × layer thickness)
    # Per AASHTO 14.7.5.1-1
    perimeter = 2 * (cfg.length_in + cfg.width_in)
    t_layer = cfg.layer_thickness_in
    S = A / (perimeter * t_layer) if (perimeter * t_layer) > 0 else 1.0

    # Horizontal stiffness: Kh = G × A / h_rt
    Kh = G * A / h_rt if h_rt > 0 else 1.0e6

    # Effective compressive modulus (shape factor dependent)
    # Ec ≈ 3G(1 + 2κS²) for steel-reinforced bearings
    # κ = material constant ≈ 0.93 for plain pads, ≈ 1.0 for reinforced
    kappa = 1.0
    Ec = 3.0 * G * (1.0 + 2.0 * kappa * S ** 2)

    # Vertical stiffness: Kv = Ec × A / h_rt
    Kv = Ec * A / h_rt if h_rt > 0 else 1.0e8

    # Rotation capacity: max rotation per AASHTO 14.7.6.3.5
    # θ_max = 0.5 × G × S / (fc × n)  — simplified check
    # More practical: θ_max ≈ 2 × h_rt / L for no-lift-off
    theta_max = 2.0 * h_rt / cfg.length_in if cfg.length_in > 0 else 0.01

    return {
        "Kh_kip_per_in": Kh,
        "Kv_kip_per_in": Kv,
        "Ec_ksi": Ec,
        "shape_factor": S,
        "plan_area_in2": A,
        "rotation_capacity_rad": theta_max,
    }


def build_elastomeric(
    cfg: ElastomericConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build elastomeric bearing model.

    Creates zeroLength element with Elastic material for shear
    and ENT (compression-only) for vertical.
    """
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="elastomeric", compression_only=True)

    props = _elastomeric_stiffness(cfg)
    model.properties = props

    # --- Nodes ---
    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # --- Materials ---
    # Horizontal shear (elastic)
    mat_h = tags.next()
    model.materials.append({
        "tag": mat_h, "type": "Elastic",
        "name": "elastomeric_shear",
        "k_kip_per_in": props["Kh_kip_per_in"],
    })

    # Vertical (compression-only)
    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "elastomeric_vertical",
        "k_kip_per_in": props["Kv_kip_per_in"],
    })

    # --- Element ---
    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "zeroLength",
        "nodes": [bot_node, top_node],
        "materials": [mat_h, mat_v, mat_h],
        "directions": [1, 2, 3],
    })

    # Temperature bounds: G varies ±15% with temperature
    # Per AASHTO 14.7.5.2: λ_a × G for thermal effects
    model.cases = {
        "upper_bound": {
            "temperature": "low",
            "G_factor": 1.15,
            "Kh_kip_per_in": props["Kh_kip_per_in"] * 1.15,
            "note": "Low temp → stiffer elastomer → higher forces",
        },
        "lower_bound": {
            "temperature": "high",
            "G_factor": 0.85,
            "Kh_kip_per_in": props["Kh_kip_per_in"] * 0.85,
            "note": "High temp → softer elastomer → larger displacements",
        },
    }

    return model


# ===================================================================
# POT BEARINGS
# ===================================================================

def build_pot_fixed(
    cfg: PotBearingConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build fixed pot bearing — rigid in both horizontal directions."""
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="pot_fixed", compression_only=True)

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # Fixed horizontal: very high stiffness
    mat_h = tags.next()
    model.materials.append({
        "tag": mat_h, "type": "Elastic",
        "name": "pot_fixed_horizontal",
        "k_kip_per_in": cfg.stiffness_kip_per_in,
    })

    # Vertical: compression only
    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "pot_vertical",
        "k_kip_per_in": cfg.stiffness_kip_per_in,
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "zeroLength",
        "nodes": [bot_node, top_node],
        "materials": [mat_h, mat_v, mat_h],
        "directions": [1, 2, 3],
    })

    model.properties = {
        "Kh_kip_per_in": cfg.stiffness_kip_per_in,
        "Kv_kip_per_in": cfg.stiffness_kip_per_in,
        "vertical_capacity_kip": cfg.vertical_capacity_kip,
    }

    # No temperature cases for pot bearings
    model.cases = {"upper_bound": {}, "lower_bound": {}}

    return model


def build_pot_guided(
    cfg: PotBearingConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build guided pot bearing — free in guided direction, fixed in other."""
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="pot_guided", compression_only=True)

    guide_dir = cfg.guide_direction or 1  # default: free longitudinal

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # Fixed direction: very stiff
    mat_fixed = tags.next()
    model.materials.append({
        "tag": mat_fixed, "type": "Elastic",
        "name": "pot_guided_fixed",
        "k_kip_per_in": cfg.stiffness_kip_per_in,
    })

    # Guided direction: very low stiffness (essentially free)
    mat_free = tags.next()
    model.materials.append({
        "tag": mat_free, "type": "Elastic",
        "name": "pot_guided_free",
        "k_kip_per_in": 0.001,  # near zero
    })

    # Vertical: compression only
    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "pot_vertical",
        "k_kip_per_in": cfg.stiffness_kip_per_in,
    })

    # Assign materials based on guide direction
    if guide_dir == 1:
        mats = [mat_free, mat_v, mat_fixed]  # free long, fixed trans
    else:
        mats = [mat_fixed, mat_v, mat_free]  # fixed long, free trans

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "zeroLength",
        "nodes": [bot_node, top_node],
        "materials": mats,
        "directions": [1, 2, 3],
    })

    model.properties = {
        "guide_direction": guide_dir,
        "Kh_fixed_kip_per_in": cfg.stiffness_kip_per_in,
        "Kh_free_kip_per_in": 0.001,
        "vertical_capacity_kip": cfg.vertical_capacity_kip,
    }

    model.cases = {"upper_bound": {}, "lower_bound": {}}

    return model


# ===================================================================
# PTFE SLIDING BEARING
# ===================================================================

def _ptfe_friction(ptfe_type: str, temperature: str = "reference") -> dict:
    """Look up PTFE friction coefficients.

    Args:
        ptfe_type: Key into PTFE_FRICTION table.
        temperature: "reference" (~68°F), "cold" (~20°F), "hot" (~100°F).

    Returns:
        Dict with mu_slow, mu_fast.
    """
    if ptfe_type not in PTFE_FRICTION:
        ptfe_type = "glass_filled"  # default

    table = PTFE_FRICTION[ptfe_type]

    if temperature == "cold":
        return {
            "mu_slow": table["mu_slow_cold"],
            "mu_fast": table["mu_fast_cold"],
        }
    elif temperature == "hot":
        # Hot = lower friction (interpolate 20% reduction from reference)
        return {
            "mu_slow": table["mu_slow"] * 0.80,
            "mu_fast": table["mu_fast"] * 0.80,
        }
    else:
        return {
            "mu_slow": table["mu_slow"],
            "mu_fast": table["mu_fast"],
        }


def build_ptfe_sliding(
    cfg: PTFEConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build PTFE sliding bearing with velocity-dependent friction.

    Generates upper and lower bound cases for temperature effects.
    Uses flatSliderBearing element concept modeled with zeroLength
    and friction material.
    """
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="ptfe_sliding", compression_only=True)

    # Reference friction
    mu_ref = _ptfe_friction(cfg.ptfe_type, "reference")
    mu_cold = _ptfe_friction(cfg.ptfe_type, "cold")
    mu_hot = _ptfe_friction(cfg.ptfe_type, "hot")

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # Friction material (reference temperature)
    mat_fric = tags.next()
    model.materials.append({
        "tag": mat_fric, "type": "Elastic",
        "name": "ptfe_friction",
        "mu_slow": mu_ref["mu_slow"],
        "mu_fast": mu_ref["mu_fast"],
        "note": "Simplified: use flatSliderBearing for full velocity-dependent model",
    })

    # Vertical: compression only
    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "ptfe_vertical",
        "k_kip_per_in": 1.0e6,
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "zeroLength",
        "nodes": [bot_node, top_node],
        "materials": [mat_fric, mat_v, mat_fric],
        "directions": [1, 2, 3],
        "note": "For full model use flatSliderBearing element",
    })

    model.properties = {
        "ptfe_type": cfg.ptfe_type,
        "mu_slow": mu_ref["mu_slow"],
        "mu_fast": mu_ref["mu_fast"],
        "vertical_capacity_kip": cfg.vertical_capacity_kip,
        "contact_pressure_ksi": cfg.contact_pressure_ksi,
    }

    # Upper/lower bound: temperature effects
    model.cases = {
        "upper_bound": {
            "temperature": "cold",
            "mu_slow": mu_cold["mu_slow"],
            "mu_fast": mu_cold["mu_fast"],
            "note": "Low temp → high friction → higher forces on substructure",
        },
        "lower_bound": {
            "temperature": "hot",
            "mu_slow": mu_hot["mu_slow"],
            "mu_fast": mu_hot["mu_fast"],
            "note": "High temp → low friction → larger displacements",
        },
    }

    return model


# ===================================================================
# SINGLE FRICTION PENDULUM
# ===================================================================

def build_fp_single(
    cfg: FPSingleConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build single friction pendulum bearing.

    Self-centering period T = 2π√(R/g).
    Uses SingleFPBearing element concept.
    """
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="fp_single", compression_only=True)

    # Self-centering period
    T = 2.0 * math.pi * math.sqrt(cfg.radius_in / G_ACCEL)

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # Friction material
    mat_fric = tags.next()
    model.materials.append({
        "tag": mat_fric, "type": "frictionModel",
        "name": "fp_friction",
        "mu": cfg.mu,
    })

    # Vertical: compression only
    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "fp_vertical",
        "k_kip_per_in": 1.0e6,
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "SingleFPBearing",
        "nodes": [bot_node, top_node],
        "R": cfg.radius_in,
        "mu": cfg.mu,
        "displacement_capacity_in": cfg.displacement_capacity_in,
    })

    model.properties = {
        "R_in": cfg.radius_in,
        "mu": cfg.mu,
        "period_sec": T,
        "displacement_capacity_in": cfg.displacement_capacity_in,
        "vertical_capacity_kip": cfg.vertical_capacity_kip,
    }

    # Upper/lower bound: ±20% friction variation
    model.cases = {
        "upper_bound": {
            "temperature": "cold",
            "mu": cfg.mu * 1.20,
            "note": "Low temp → higher friction → higher substructure forces",
        },
        "lower_bound": {
            "temperature": "hot",
            "mu": cfg.mu * 0.80,
            "note": "High temp → lower friction → larger displacements",
        },
    }

    return model


# ===================================================================
# TRIPLE FRICTION PENDULUM
# ===================================================================

def build_fp_triple(
    cfg: FPTripleConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build triple friction pendulum bearing.

    Multi-stage behavior with 4 friction surfaces and 4 radii.
    Uses TripleFrictionPendulum element concept.
    """
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="fp_triple", compression_only=True)

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # Friction materials per surface
    fric_tags = []
    for i, (mu, R) in enumerate(zip(
        [cfg.mu1, cfg.mu2, cfg.mu3, cfg.mu4],
        [cfg.R1, cfg.R2, cfg.R3, cfg.R4],
    )):
        mat_tag = tags.next()
        model.materials.append({
            "tag": mat_tag, "type": "frictionModel",
            "name": f"fp_triple_surface_{i+1}",
            "mu": mu,
            "R": R,
        })
        fric_tags.append(mat_tag)

    # Vertical: compression only
    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "fp_triple_vertical",
        "k_kip_per_in": 1.0e6,
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "TripleFrictionPendulum",
        "nodes": [bot_node, top_node],
        "R1": cfg.R1, "R2": cfg.R2, "R3": cfg.R3, "R4": cfg.R4,
        "mu1": cfg.mu1, "mu2": cfg.mu2, "mu3": cfg.mu3, "mu4": cfg.mu4,
        "d1": cfg.d1, "d2": cfg.d2, "d3": cfg.d3, "d4": cfg.d4,
        "friction_materials": fric_tags,
    })

    # Effective R for period estimate (outer surfaces dominate)
    R_eff = cfg.R2 + cfg.R3  # simplified
    T_eff = 2.0 * math.pi * math.sqrt(R_eff / G_ACCEL)

    total_capacity = cfg.d1 + cfg.d2 + cfg.d3 + cfg.d4

    model.properties = {
        "R1": cfg.R1, "R2": cfg.R2, "R3": cfg.R3, "R4": cfg.R4,
        "mu1": cfg.mu1, "mu2": cfg.mu2, "mu3": cfg.mu3, "mu4": cfg.mu4,
        "d1": cfg.d1, "d2": cfg.d2, "d3": cfg.d3, "d4": cfg.d4,
        "R_effective_in": R_eff,
        "period_effective_sec": T_eff,
        "total_displacement_capacity_in": total_capacity,
        "vertical_capacity_kip": cfg.vertical_capacity_kip,
    }

    # Upper/lower bound: ±20% all friction coefficients
    model.cases = {
        "upper_bound": {
            "temperature": "cold",
            "mu1": cfg.mu1 * 1.20, "mu2": cfg.mu2 * 1.20,
            "mu3": cfg.mu3 * 1.20, "mu4": cfg.mu4 * 1.20,
            "note": "Low temp → higher friction → higher substructure forces",
        },
        "lower_bound": {
            "temperature": "hot",
            "mu1": cfg.mu1 * 0.80, "mu2": cfg.mu2 * 0.80,
            "mu3": cfg.mu3 * 0.80, "mu4": cfg.mu4 * 0.80,
            "note": "High temp → lower friction → larger displacements",
        },
    }

    return model


# ===================================================================
# INTEGRAL (MONOLITHIC)
# ===================================================================

def build_integral(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build integral (monolithic) connection.

    Uses equalDOF constraint — no relative displacement.
    No bearing element, just a constraint between super and sub nodes.
    """
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="integral", compression_only=False)

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # equalDOF constraint: all DOFs tied
    model.constraints.append({
        "type": "equalDOF",
        "master": bot_node,
        "slave": top_node,
        "dofs": [1, 2, 3, 4, 5, 6],
    })

    model.properties = {
        "connection": "monolithic",
        "relative_displacement": 0.0,
    }

    # No temperature cases for integral
    model.cases = {"upper_bound": {}, "lower_bound": {}}

    return model


# ===================================================================
# ROCKER / ROLLER (STEEL — VINTAGE)
# ===================================================================

def build_rocker_roller(
    cfg: RockerRollerConfig,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
) -> BearingModel:
    """Build rocker/roller bearing (vintage steel).

    Uses ENT (compression-only) vertical + near-free horizontal.
    Flags uplift vulnerability.
    """
    tags = TagAllocator(tag_start)
    model = BearingModel(bearing_type="rocker_roller", compression_only=True)

    bot_node = tags.next()
    model.nodes.append({"tag": bot_node, "x": x, "y": y, "z": z})
    model.bottom_nodes.append(bot_node)

    top_node = tags.next()
    model.nodes.append({"tag": top_node, "x": x, "y": y, "z": z})
    model.top_nodes.append(top_node)

    # Horizontal: near-free (roller) or with some resistance (rocker)
    if cfg.roller_diameter_in > 0:
        # Roller: very free
        mat_h = tags.next()
        model.materials.append({
            "tag": mat_h, "type": "Elastic",
            "name": "roller_horizontal",
            "k_kip_per_in": 0.001,
        })
    else:
        # Rocker: some rotational resistance via contact
        mat_h = tags.next()
        model.materials.append({
            "tag": mat_h, "type": "Elastic",
            "name": "rocker_horizontal",
            "k_kip_per_in": 1.0,  # minimal
        })

    # Vertical: compression only (ENT)
    # Hertz contact capacity: P = π/4 × fy × D × w
    # (simplified line contact for rocker)
    D = cfg.rocker_radius_in * 2
    hertz_capacity = math.pi / 4 * cfg.steel_fy_ksi * D  # per inch of width
    Kv = 1.0e5  # stiff in compression

    mat_v = tags.next()
    model.materials.append({
        "tag": mat_v, "type": "ENT",
        "name": "rocker_vertical",
        "k_kip_per_in": Kv,
    })

    elem_tag = tags.next()
    model.elements.append({
        "tag": elem_tag, "type": "zeroLength",
        "nodes": [bot_node, top_node],
        "materials": [mat_h, mat_v, mat_h],
        "directions": [1, 2, 3],
    })

    model.properties = {
        "rocker_radius_in": cfg.rocker_radius_in,
        "roller_diameter_in": cfg.roller_diameter_in,
        "hertz_line_capacity_kip_per_in": hertz_capacity,
        "vertical_capacity_kip": cfg.vertical_capacity_kip,
        "Kv_kip_per_in": Kv,
    }

    model.warnings.append(
        "VINTAGE BEARING: Rocker/roller bearings are vulnerable to uplift "
        "and unseating. Consider retrofit per FHWA-HRT-06-032."
    )

    model.cases = {"upper_bound": {}, "lower_bound": {}}

    return model


# ===================================================================
# BEARING LAYOUT HELPER
# ===================================================================

def layout_bearings(
    num_girders: int,
    girder_spacing_ft: float,
    bearing_type: str,
    support_type: str = "fixed",
    x: float = 0.0,
    y: float = 0.0,
    z_center: float = 0.0,
    tag_start: int = 1,
    **kwargs: Any,
) -> list[BearingModel]:
    """Generate bearings for a multi-girder bridge at one support line.

    Places one bearing per girder line at regular transverse spacing.

    Args:
        num_girders: Number of girders.
        girder_spacing_ft: Girder spacing (ft).
        bearing_type: BearingType value string.
        support_type: "fixed" or "expansion" (affects pot bearing choice).
        x: Longitudinal coordinate (inches).
        y: Vertical coordinate (inches).
        z_center: Transverse center coordinate (inches).
        tag_start: Starting tag.
        **kwargs: Forwarded to bearing config constructors.

    Returns:
        List of BearingModel, one per girder line.
    """
    spacing_in = girder_spacing_ft * FT_TO_IN
    total_width = (num_girders - 1) * spacing_in
    z_start = z_center - total_width / 2.0

    bearings: list[BearingModel] = []
    current_tag = tag_start

    for g_idx in range(num_girders):
        z_loc = z_start + g_idx * spacing_in

        bm = create_bearing(
            bearing_type=bearing_type,
            x=x, y=y, z=z_loc,
            tag_start=current_tag,
            support_type=support_type,
            **kwargs,
        )
        bearings.append(bm)

        # Advance tags past what was used
        max_tag = 0
        for n in bm.nodes:
            max_tag = max(max_tag, n["tag"])
        for e in bm.elements:
            max_tag = max(max_tag, e["tag"])
        for m in bm.materials:
            max_tag = max(max_tag, m["tag"])
        current_tag = max_tag + 1

    return bearings


# ===================================================================
# TOP-LEVEL DISPATCHER
# ===================================================================

def create_bearing(
    bearing_type: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    tag_start: int = 1,
    **kwargs: Any,
) -> BearingModel:
    """Create a bearing model from type string and parameters.

    Main entry point for the NLB pipeline.

    Args:
        bearing_type: One of BearingType values.
        x, y, z: Bearing location (inches).
        tag_start: Starting tag.
        **kwargs: Parameters forwarded to specific builder.

    Returns:
        BearingModel.

    Raises:
        ValueError: If bearing_type is not recognized.
    """
    bt = bearing_type.lower().replace(" ", "_").replace("-", "_")

    if bt == "elastomeric":
        cfg = kwargs.get("config", ElastomericConfig(**{
            k: v for k, v in kwargs.items()
            if k in ElastomericConfig.__dataclass_fields__
        }))
        return build_elastomeric(cfg, x, y, z, tag_start)

    elif bt == "pot_fixed":
        cfg = kwargs.get("config", PotBearingConfig(**{
            k: v for k, v in kwargs.items()
            if k in PotBearingConfig.__dataclass_fields__
        }))
        return build_pot_fixed(cfg, x, y, z, tag_start)

    elif bt == "pot_guided":
        cfg = kwargs.get("config", PotBearingConfig(**{
            k: v for k, v in kwargs.items()
            if k in PotBearingConfig.__dataclass_fields__
        }))
        if cfg.guide_direction is None:
            cfg.guide_direction = 1
        return build_pot_guided(cfg, x, y, z, tag_start)

    elif bt == "ptfe_sliding":
        cfg = kwargs.get("config", PTFEConfig(**{
            k: v for k, v in kwargs.items()
            if k in PTFEConfig.__dataclass_fields__
        }))
        return build_ptfe_sliding(cfg, x, y, z, tag_start)

    elif bt in ("fp_single", "friction_pendulum_single"):
        cfg = kwargs.get("config", FPSingleConfig(**{
            k: v for k, v in kwargs.items()
            if k in FPSingleConfig.__dataclass_fields__
        }))
        return build_fp_single(cfg, x, y, z, tag_start)

    elif bt in ("fp_triple", "friction_pendulum_triple"):
        cfg = kwargs.get("config", FPTripleConfig(**{
            k: v for k, v in kwargs.items()
            if k in FPTripleConfig.__dataclass_fields__
        }))
        return build_fp_triple(cfg, x, y, z, tag_start)

    elif bt == "integral":
        return build_integral(x, y, z, tag_start)

    elif bt in ("rocker_roller", "rocker", "roller"):
        cfg = kwargs.get("config", RockerRollerConfig(**{
            k: v for k, v in kwargs.items()
            if k in RockerRollerConfig.__dataclass_fields__
        }))
        return build_rocker_roller(cfg, x, y, z, tag_start)

    else:
        raise ValueError(
            f"Unknown bearing type '{bearing_type}'. "
            f"Supported: {[e.value for e in BearingType]}"
        )

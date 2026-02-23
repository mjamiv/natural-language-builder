"""
OpenSees Abstraction Layer for Natural Language Builder.

Provides bridge engineering materials, fiber sections, elements, and analysis
runners built on OpenSeesPy. All internal units: kip-inch-second (KIS).

Modules:
    materials  — Concrete, steel, soil springs, bearing materials
    sections   — Fiber section builders (I-shapes, RC, box girders, PT)
    elements   — Element type mappings and geometric transforms
    analysis   — Analysis sequence runners with convergence handling
"""

from nlb.opensees.materials import (
    confined_concrete,
    unconfined_concrete,
    concrete_defaults,
    mander_confinement,
    reinforcing_steel,
    structural_steel,
    prestressing_strand,
    STEEL_DEFAULTS,
    py_spring,
    tz_spring,
    qz_spring,
    api_py_curves,
    compute_tult,
    compute_qult,
    elastomeric_shear,
    friction_model,
    compression_only,
)

from nlb.opensees.sections import (
    steel_i_section,
    composite_section,
    circular_rc_section,
    rectangular_rc_section,
    box_girder_section,
    prestressed_i_section,
)

from nlb.opensees.elements import (
    beam_column,
    truss_element,
    zero_length,
    shell_element,
    geometric_transform,
)

from nlb.opensees.analysis import (
    gravity_analysis,
    pushover_analysis,
    response_spectrum,
    time_history,
    staged_construction,
    convergence_handler,
)

__all__ = [
    # Materials
    "confined_concrete", "unconfined_concrete", "concrete_defaults",
    "mander_confinement", "reinforcing_steel", "structural_steel",
    "prestressing_strand", "STEEL_DEFAULTS",
    "py_spring", "tz_spring", "qz_spring",
    "api_py_curves", "compute_tult", "compute_qult",
    "elastomeric_shear", "friction_model", "compression_only",
    # Sections
    "steel_i_section", "composite_section", "circular_rc_section",
    "rectangular_rc_section", "box_girder_section", "prestressed_i_section",
    # Elements
    "beam_column", "truss_element", "zero_length", "shell_element",
    "geometric_transform",
    # Analysis
    "gravity_analysis", "pushover_analysis", "response_spectrum",
    "time_history", "staged_construction", "convergence_handler",
]

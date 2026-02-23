"""
OpenSees Element Type Mappings for Bridge Engineering.

All internal units: kip-inch-second (KIS).

This module provides convenience wrappers around OpenSeesPy element commands
with bridge-engineering-appropriate defaults and documentation.

References:
    - AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    - OpenSees Command Manual: https://opensees.berkeley.edu/wiki/
    - Scott, M.H. & Fenves, G.L. (2006). "Plastic Hinge Integration Methods
      for Force-Based Beam-Column Elements." ASCE J. Structural Engineering.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import openseespy.opensees as ops


# ============================================================================
# GEOMETRIC TRANSFORMS
# ============================================================================

def geometric_transform(tag: int, transform_type: str,
                        *vecxz: float) -> int:
    """Define a geometric transformation for beam-column elements.

    The transformation maps element local coordinates to global coordinates
    and determines how geometric nonlinearity is handled.

    Args:
        tag:             Transformation tag.
        transform_type:  One of:
                         - "Linear"        — small displacement, no P-delta
                         - "PDelta"        — includes P-delta effects
                         - "Corotational"  — large displacement (co-rotational)
        *vecxz:          Components of the vector in the local x-z plane
                         (used to define the local coordinate system).
                         For 2D: not needed (can be omitted).
                         For 3D: typically (0, 0, 1) for vertical elements or
                                 (0, 1, 0) for horizontal elements.

    Returns:
        Transformation tag.

    Reference:
        AASHTO LRFD C4.5.3.2.2b: P-Delta effects.
        OpenSees geomTransf command documentation.

    Example:
        >>> # 2D PDelta transform
        >>> geometric_transform(1, 'PDelta')
        1
        >>> # 3D Linear transform for horizontal beam
        >>> geometric_transform(2, 'Linear', 0.0, 0.0, 1.0)
        2
    """
    valid_types = ('Linear', 'PDelta', 'Corotational')
    if transform_type not in valid_types:
        raise ValueError(
            f"transform_type must be one of {valid_types}, got '{transform_type}'"
        )

    if vecxz:
        ops.geomTransf(transform_type, tag, *vecxz)
    else:
        ops.geomTransf(transform_type, tag)
    return tag


# ============================================================================
# BEAM-COLUMN ELEMENTS
# ============================================================================

def beam_column(tag: int, nodes: Tuple[int, int], section: int,
                transform: int, integration: str = 'Lobatto',
                np: int = 5) -> int:
    """Create a displacement-based beam-column element.

    Uses dispBeamColumn for displacement-based formulation, which is
    recommended for most bridge analysis applications. The element uses
    distributed plasticity with fiber sections at integration points.

    Args:
        tag:          Element tag.
        nodes:        Tuple of (iNode, jNode) — end node tags.
        section:      Section tag (fiber section from sections module).
        transform:    Geometric transformation tag.
        integration:  Integration scheme. Options:
                      - 'Lobatto'       — Gauss-Lobatto (default, recommended)
                      - 'Legendre'      — Gauss-Legendre
                      - 'Radau'         — Gauss-Radau
                      - 'NewtonCotes'   — Newton-Cotes
                      - 'HingeRadau'    — Modified two-point Gauss-Radau
        np:           Number of integration points. Default: 5.
                      Per Scott & Fenves (2006), 5 is generally sufficient.
                      Use 3 minimum, 7 for high-curvature demands.

    Returns:
        Element tag.

    Reference:
        AASHTO LRFD 4.5.3.2: Approximate Methods of Analysis.
        Scott & Fenves (2006), ASCE J. Structural Eng.
    """
    # Create beam integration rule with same tag as element
    ops.beamIntegration(integration, tag, section, np)
    ops.element('dispBeamColumn', tag, *nodes, transform, tag)
    return tag


# ============================================================================
# TRUSS ELEMENTS
# ============================================================================

def truss_element(tag: int, nodes: Tuple[int, int], area: float,
                  material: int) -> int:
    """Create a co-rotational truss element.

    Uses corotTruss for large-displacement truss analysis, suitable for
    cable elements, bracing, and truss bridges. The co-rotational
    formulation accounts for geometric nonlinearity.

    Args:
        tag:      Element tag.
        nodes:    Tuple of (iNode, jNode).
        area:     Cross-sectional area (in²).
        material: Uniaxial material tag.

    Returns:
        Element tag.

    Reference:
        AASHTO LRFD 4.6.2.7: Truss Bridges.
    """
    ops.element('corotTruss', tag, *nodes, area, material)
    return tag


# ============================================================================
# ZERO-LENGTH ELEMENTS
# ============================================================================

def zero_length(tag: int, nodes: Tuple[int, int],
                materials: List[int], directions: List[int]) -> int:
    """Create a zero-length element for bearings, springs, or releases.

    Zero-length elements connect two nodes at the same location with
    specified material behavior in each DOF. Used extensively for:
    - Elastomeric bearing pads (shear + axial)
    - Soil springs (p-y, t-z, q-z)
    - Moment releases (pins)
    - Friction elements

    Args:
        tag:         Element tag.
        nodes:       Tuple of (iNode, jNode) — must be at same coordinates.
        materials:   List of material tags, one per direction.
        directions:  List of DOF directions (1=x, 2=y, 3=z, 4=rx, 5=ry, 6=rz).
                     Must match length of materials.

    Returns:
        Element tag.

    Raises:
        ValueError: If materials and directions have different lengths.

    Reference:
        AASHTO LRFD 14.7: Bearings.
        OpenSees zeroLength command documentation.

    Example:
        >>> # Bearing with shear spring in x and y, rigid in z
        >>> zero_length(1, (100, 200), [10, 11, 12], [1, 2, 3])
        1
    """
    if len(materials) != len(directions):
        raise ValueError(
            f"materials ({len(materials)}) and directions ({len(directions)}) "
            f"must have the same length."
        )
    ops.element('zeroLength', tag, *nodes,
                '-mat', *materials, '-dir', *directions)
    return tag


# ============================================================================
# SHELL ELEMENTS
# ============================================================================

def shell_element(tag: int, nodes: Tuple[int, int, int, int],
                  thickness: float, material: int) -> int:
    """Create a ShellMITC4 element for deck or wall modeling.

    MITC4 (Mixed Interpolation of Tensorial Components) formulation avoids
    shear locking and is suitable for both thin and moderately thick shells.

    Args:
        tag:        Element tag.
        nodes:      Tuple of 4 node tags (counter-clockwise ordering).
        thickness:  Shell thickness (inches).
        material:   nDMaterial tag (e.g., ElasticIsotropic or PlateFromPlaneStress).

    Returns:
        Element tag.

    Note:
        The material must be an nDMaterial (2D), not a uniaxialMaterial.
        For bridge decks, typically use:
            ops.nDMaterial('ElasticIsotropic', matTag, E, nu)
            ops.section('PlateFiber', secTag, matTag, thickness)
        Then pass secTag to this function.

    Reference:
        AASHTO LRFD 4.6.3.3: Refined Methods (FEM).
        Dvorkin, E.N. & Bathe, K.J. (1984). "A Continuum Mechanics Based
        Four-Node Shell Element." Engineering Computations, 1.
    """
    # ShellMITC4 uses a section tag, not raw material + thickness
    # The caller should create a PlateFiber section first.
    # For convenience, we accept the section tag as 'material' parameter.
    ops.element('ShellMITC4', tag, *nodes, material)
    return tag

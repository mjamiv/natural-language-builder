"""
OpenSees Material Library for Bridge Engineering.

All internal units: kip-inch-second (KIS).
    Stress: ksi (kip/in²)
    Strain: in/in (dimensionless)
    Force:  kip
    Length: inch
    Time:   second

References:
    - AASHTO LRFD Bridge Design Specifications, 9th Edition (2020)
    - ACI 318-19: Building Code Requirements for Structural Concrete
    - Mander, J.B., Priestley, M.J.N., Park, R. (1988). "Theoretical Stress-Strain
      Model for Confined Concrete." ASCE J. Structural Engineering, 114(8).
    - API RP 2GEO: Geotechnical and Foundation Design Considerations (2014)
    - FHWA-NHI-10-016: Drilled Shafts (2010)
    - Reese, L.C. & O'Neill, M.W. (1988). Drilled Shafts: Construction Procedures
      and Design Methods. FHWA-HI-88-042.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import openseespy.opensees as ops


# ============================================================================
# CONCRETE
# ============================================================================

@dataclass
class ConcreteProperties:
    """Computed concrete material properties.

    Attributes:
        fc:    Compressive strength (ksi, positive value)
        Ec:    Elastic modulus (ksi)
        fr:    Modulus of rupture (ksi)
        eps_0: Strain at peak stress (in/in, positive value)
        eps_cu: Ultimate crushing strain (in/in, positive value)
    """
    fc: float
    Ec: float
    fr: float
    eps_0: float
    eps_cu: float


def concrete_defaults(fc_ksi: float) -> ConcreteProperties:
    """Auto-compute concrete properties from f'c.

    Per AASHTO LRFD 5.4.2.4 and ACI 318-19 Table 19.2.2.1.

    Args:
        fc_ksi: Specified compressive strength f'c in ksi (positive).

    Returns:
        ConcreteProperties with Ec, fr, eps_0, eps_cu.

    Example:
        >>> props = concrete_defaults(4.0)  # 4 ksi concrete
        >>> round(props.Ec, 0)  # ~3644 ksi
        3644.0
    """
    fc = abs(fc_ksi)
    # Unit weight of normal-weight concrete: 0.150 kcf = 150 pcf
    wc = 0.150  # kcf

    # AASHTO LRFD Eq. 5.4.2.4-1: Ec = 33000 * K1 * wc^1.5 * sqrt(f'c)
    # K1 = 1.0 (correction factor), wc in kcf, f'c in ksi
    # Note: 33000 * (0.150)^1.5 = 33000 * 0.05809 = 1917 (standard approximation)
    Ec = 33000.0 * 1.0 * (wc ** 1.5) * math.sqrt(fc)  # ksi

    # AASHTO LRFD Eq. 5.4.2.6-1: fr = 0.24 * sqrt(f'c) (ksi)
    fr = 0.24 * math.sqrt(fc)

    # ACI 318 / Hognestad: eps_0 = 2*f'c / Ec
    eps_0 = 2.0 * fc / Ec

    # Ultimate crushing strain per AASHTO 5.6.3.3.2: 0.003
    eps_cu = 0.003

    return ConcreteProperties(fc=fc, Ec=Ec, fr=fr, eps_0=eps_0, eps_cu=eps_cu)


def unconfined_concrete(tag: int, fc: float, eps_0: Optional[float] = None,
                        eps_cu: Optional[float] = None) -> int:
    """Define unconfined (cover) concrete using Concrete01.

    OpenSees Concrete01: zero tensile strength, linear-to-peak then linear
    degradation to crushing.

    Args:
        tag:    Material tag.
        fc:     Compressive strength f'c (ksi, positive). Internally negated
                for OpenSees convention.
        eps_0:  Strain at peak stress (positive). Default: computed from fc.
        eps_cu: Ultimate crushing strain (positive). Default: 0.003.

    Returns:
        Material tag.

    Reference:
        AASHTO LRFD 5.6.3.3.2 — strain limits for concrete.
    """
    props = concrete_defaults(fc)
    e0 = eps_0 if eps_0 is not None else props.eps_0
    ecu = eps_cu if eps_cu is not None else props.eps_cu

    # Concrete01: fpc, epsc0, fpcu, epsU
    # fpc = peak compressive stress (negative in OpenSees)
    # epsc0 = strain at peak (negative)
    # fpcu = residual stress at epsU (assume 0 for unconfined)
    # epsU = ultimate strain (negative)
    ops.uniaxialMaterial('Concrete01', tag, -fc, -e0, 0.0, -ecu)
    return tag


def confined_concrete(tag: int, fc: float, fcc: float, ecc: float,
                      ecu: float) -> int:
    """Define confined concrete using Concrete01.

    Mander model: confined concrete has enhanced strength (fcc > fc) and
    enhanced ductility (ecu >> 0.003).

    Args:
        tag:  Material tag.
        fc:   Unconfined compressive strength (ksi, positive).
        fcc:  Confined compressive strength (ksi, positive).
        ecc:  Strain at confined peak stress (positive).
        ecu:  Ultimate confined crushing strain (positive).

    Returns:
        Material tag.

    Reference:
        Mander et al. (1988), AASHTO SGS Guide Spec 8.4.4.
    """
    # Residual strength: typically 0.2*fcc for confined concrete
    fcu_residual = 0.2 * fcc
    ops.uniaxialMaterial('Concrete01', tag, -fcc, -ecc, -fcu_residual, -ecu)
    return tag


def mander_confinement(fc: float, fy_transverse: float, rho_s: float,
                       config: str = "circular") -> Tuple[float, float, float]:
    """Compute confined concrete parameters per Mander et al. (1988).

    Args:
        fc:             Unconfined f'c (ksi, positive).
        fy_transverse:  Yield strength of transverse reinforcement (ksi).
        rho_s:          Volumetric ratio of transverse steel.
        config:         "circular" for spiral/hoop, "rectangular" for ties.

    Returns:
        Tuple of (fcc, ecc, ecu):
            fcc — confined compressive strength (ksi)
            ecc — strain at confined peak
            ecu — ultimate confined strain

    Reference:
        Mander, Priestley & Park (1988).
        AASHTO SGS Guide Spec Section 8.4.4.
    """
    # Effective confinement pressure
    # ke = confinement effectiveness coefficient
    if config == "circular":
        ke = 0.95  # spirals
    else:
        ke = 0.75  # rectangular ties (approximate)

    # Lateral confining pressure: fl = 0.5 * ke * rho_s * fy_transverse
    fl = 0.5 * ke * rho_s * fy_transverse

    # Confined strength ratio
    ratio = fl / fc
    # Mander equation: fcc = fc * (-1.254 + 2.254*sqrt(1 + 7.94*fl/fc) - 2*fl/fc)
    fcc = fc * (-1.254 + 2.254 * math.sqrt(1.0 + 7.94 * ratio) - 2.0 * ratio)

    # Strain at peak confined stress
    eps_co = 0.002  # unconfined peak strain (standard assumption)
    ecc = eps_co * (1.0 + 5.0 * (fcc / fc - 1.0))

    # Ultimate confined strain (Priestley et al. energy balance)
    # ecu = 0.004 + 1.4 * rho_s * fy_transverse * eps_su / fcc
    # eps_su = 0.09 for Grade 60 steel (assumed)
    eps_su = 0.09
    ecu = 0.004 + 1.4 * rho_s * fy_transverse * eps_su / fcc

    return (fcc, ecc, ecu)


# ============================================================================
# STEEL
# ============================================================================

@dataclass
class SteelDefaults:
    """Common steel material property sets.

    All values in ksi.
    """
    fy: float       # Yield strength
    fu: float       # Ultimate strength
    Es: float       # Elastic modulus
    b: float        # Strain-hardening ratio (Esh/Es)
    R0: float = 20.0    # Steel02 transition parameter
    cR1: float = 0.925   # Steel02 transition parameter
    cR2: float = 0.15    # Steel02 transition parameter


# Standard steel defaults — AASHTO Table 6.4.1-1 and ASTM specs
STEEL_DEFAULTS: Dict[str, SteelDefaults] = {
    "A615_Gr60": SteelDefaults(fy=60.0, fu=90.0, Es=29000.0, b=0.01),
    "A706_Gr60": SteelDefaults(fy=60.0, fu=80.0, Es=29000.0, b=0.01),
    "A992_Gr50": SteelDefaults(fy=50.0, fu=65.0, Es=29000.0, b=0.01),
    "HPS_70W": SteelDefaults(fy=70.0, fu=85.0, Es=29000.0, b=0.005),
    "270ksi_strand": SteelDefaults(fy=243.0, fu=270.0, Es=28500.0, b=0.008,
                                    R0=10.0, cR1=0.9, cR2=0.15),
}


def reinforcing_steel(tag: int, fy: float = 60.0, fu: float = 90.0,
                      Es: float = 29000.0, b: float = 0.01,
                      R0: float = 20.0, cR1: float = 0.925,
                      cR2: float = 0.15) -> int:
    """Define reinforcing steel using Steel02 (Giuffré-Menegotto-Pinto).

    Suitable for A615 Gr60 and A706 Gr60 reinforcement. Steel02 captures
    the Bauschinger effect under cyclic loading, critical for seismic analysis.

    Args:
        tag:  Material tag.
        fy:   Yield strength (ksi). Default: 60 ksi (Gr60).
        fu:   Ultimate strength (ksi). Default: 90 ksi.
        Es:   Elastic modulus (ksi). Default: 29000 ksi.
        b:    Strain-hardening ratio (b = Esh/Es). Default: 0.01.
        R0:   Initial value of curvature parameter. Default: 20.
        cR1:  Curvature degradation parameter. Default: 0.925.
        cR2:  Curvature degradation parameter. Default: 0.15.

    Returns:
        Material tag.

    Reference:
        AASHTO LRFD 5.4.3: Reinforcing Steel.
        Giuffré, A. & Pinto, P. (1970). Menegotto, M. & Pinto, P. (1973).
    """
    ops.uniaxialMaterial('Steel02', tag, fy, Es, b, R0, cR1, cR2)
    return tag


def structural_steel(tag: int, fy: float = 50.0, Es: float = 29000.0,
                     b: float = 0.01) -> int:
    """Define structural steel for W-shapes using Steel02.

    Default: ASTM A992 Grade 50 for rolled W-shapes.

    Args:
        tag: Material tag.
        fy:  Yield strength (ksi). Default: 50 ksi (A992).
        Es:  Elastic modulus (ksi). Default: 29000 ksi.
        b:   Strain-hardening ratio. Default: 0.01.

    Returns:
        Material tag.

    Reference:
        AASHTO LRFD 6.4.1: Structural Steel.
        ASTM A992/A992M — Standard Specification for Structural Steel Shapes.
    """
    ops.uniaxialMaterial('Steel02', tag, fy, Es, b, 20.0, 0.925, 0.15)
    return tag


def prestressing_strand(tag: int, fpu: float = 270.0,
                        Eps: float = 28500.0) -> int:
    """Define prestressing strand using Steel02.

    Calibrated for 7-wire, low-relaxation strand per ASTM A416.
    fpy = 0.9 * fpu for low-relaxation strand.

    Args:
        tag:  Material tag.
        fpu:  Ultimate tensile strength (ksi). Default: 270 ksi.
        Eps:  Elastic modulus (ksi). Default: 28500 ksi.

    Returns:
        Material tag.

    Reference:
        AASHTO LRFD 5.4.4: Prestressing Steel.
        ASTM A416/A416M: Standard Specification for Low-Relaxation,
        Seven-Wire Steel Strand for Prestressed Concrete.
    """
    fpy = 0.9 * fpu  # Yield strength for low-relaxation strand
    b = 0.008  # Low strain-hardening ratio for strand
    # R0=10 gives better fit for strand stress-strain curve
    ops.uniaxialMaterial('Steel02', tag, fpy, Eps, b, 10.0, 0.9, 0.15)
    return tag


# ============================================================================
# SOIL SPRINGS
# ============================================================================

@dataclass
class SoilLayer:
    """Soil layer definition for spring parameter computation.

    Attributes:
        depth_top:    Depth to top of layer (inches from mudline).
        depth_bot:    Depth to bottom of layer (inches).
        soil_type:    "sand" or "clay".
        gamma:        Effective unit weight (kip/in³). Typical sand: 0.0000694 (120 pcf).
        phi:          Friction angle (degrees). For sand.
        su:           Undrained shear strength (ksi). For clay.
        eps_50:       Strain at 50% of ultimate resistance. For clay.
        k_py:         Initial modulus of subgrade reaction (kip/in³). For sand.
    """
    depth_top: float
    depth_bot: float
    soil_type: str  # "sand" or "clay"
    gamma: float = 0.0000694  # ~120 pcf in kip/in³
    phi: float = 35.0
    su: float = 0.0
    eps_50: float = 0.01
    k_py: float = 0.0


def py_spring(tag: int, soil_type: int, pu: float, y50: float,
              cd: float = 0.0) -> int:
    """Define lateral soil spring using PySimple1.

    Models the p-y behavior for laterally loaded piles.

    Args:
        tag:        Material tag.
        soil_type:  1 = soft clay (Matlock 1970), 2 = sand (API).
        pu:         Ultimate capacity of the p-y spring (kip/in).
        y50:        Displacement at 50% of pu (inches).
        cd:         Drag resistance ratio (0 to 1). Default: 0.

    Returns:
        Material tag.

    Reference:
        API RP 2GEO Section 8.
        Boulanger, R.W. et al. (1999). "Seismic Soil-Pile-Structure
        Interaction Experiments and Analyses." ASCE JGGE.
    """
    ops.uniaxialMaterial('PySimple1', tag, soil_type, pu, y50, cd)
    return tag


def tz_spring(tag: int, soil_type: int, tult: float, z50: float) -> int:
    """Define skin friction spring using TzSimple1.

    Models the t-z behavior for axially loaded piles (shaft friction).

    Args:
        tag:        Material tag.
        soil_type:  1 = Reese & O'Neill t-z for driven piles,
                    2 = Mosher (1984) t-z for drilled shafts.
        tult:       Ultimate skin friction capacity (kip/in).
        z50:        Displacement at 50% of tult (inches).

    Returns:
        Material tag.

    Reference:
        API RP 2GEO Section 9.
        FHWA-NHI-10-016 Section 9.3: Axial Capacity.
    """
    ops.uniaxialMaterial('TzSimple1', tag, soil_type, tult, z50)
    return tag


def qz_spring(tag: int, qult: float, z50: float, soil_type: int = 2) -> int:
    """Define tip bearing spring using QzSimple1.

    Models the q-z behavior for pile tip resistance.

    Args:
        tag:        Material tag.
        qult:       Ultimate tip bearing capacity (kip).
        z50:        Displacement at 50% of qult (inches).
        soil_type:  1 = Reese & O'Neill backbone, 2 = Vijayvergiya (1977).

    Returns:
        Material tag.

    Reference:
        API RP 2GEO Section 9.
        FHWA-NHI-10-016: Drilled Shafts Chapter 13.
    """
    ops.uniaxialMaterial('QzSimple1', tag, soil_type, qult, z50)
    return tag


def api_py_curves(diameter: float, depth: float,
                  soil_layers: List[SoilLayer]) -> List[Dict[str, float]]:
    """Generate p-y curve parameters per API RP 2GEO / AASHTO.

    Computes pu and y50 at a given depth for the soil profile.

    Args:
        diameter:     Pile diameter (inches).
        depth:        Depth below mudline (inches).
        soil_layers:  List of SoilLayer definitions.

    Returns:
        List of dicts with keys: 'depth', 'pu', 'y50', 'soil_type_code'.

    Reference:
        API RP 2GEO (2014) Section 8.6-8.7.
        AASHTO LRFD Section 10.7.3.
    """
    results = []
    for layer in soil_layers:
        if depth < layer.depth_top or depth > layer.depth_bot:
            continue

        if layer.soil_type == "sand":
            pu, y50 = _api_py_sand(diameter, depth, layer)
            soil_code = 2
        else:  # clay
            pu, y50 = _api_py_clay(diameter, depth, layer)
            soil_code = 1

        results.append({
            'depth': depth,
            'pu': pu,
            'y50': y50,
            'soil_type_code': soil_code,
        })

    return results


def _api_py_sand(diameter: float, depth: float,
                 layer: SoilLayer) -> Tuple[float, float]:
    """Compute p-y parameters for sand per API RP 2GEO Section 8.7.

    Args:
        diameter: Pile diameter (inches).
        depth:    Depth below mudline (inches).
        layer:    SoilLayer with sand properties.

    Returns:
        Tuple of (pu, y50) in kip/in and inches.
    """
    phi_rad = math.radians(layer.phi)
    gamma = layer.gamma  # kip/in³

    # Overburden pressure at depth
    sigma_v = gamma * depth  # ksi

    # API coefficients (simplified)
    # C1, C2, C3 depend on phi — use approximate relations
    c1 = max(0.0, (0.115 * 10 ** (0.0405 * layer.phi)))
    c2 = max(0.0, (0.571 * 10 ** (0.022 * layer.phi)))
    c3 = max(0.0, (0.646 * 10 ** (0.0555 * layer.phi)))

    # Shallow vs deep mechanism
    pu_shallow = (c1 * depth + c2 * diameter) * sigma_v
    pu_deep = c3 * sigma_v * diameter
    pu = min(pu_shallow, pu_deep) if depth > 0 else 0.001

    pu = max(pu, 0.001)  # Floor

    # Initial modulus: k * depth (k in kip/in³ from API table)
    k = layer.k_py if layer.k_py > 0 else _api_k_sand(layer.phi)

    # A factor (static vs cyclic) — use static (A >= 0.9)
    A = max(0.9, 3.0 - 0.8 * depth / diameter) if diameter > 0 else 0.9

    # y50 for sand: approximate as A*pu / (k*depth) but bounded
    if k * depth > 0:
        y50 = A * pu / (k * depth)
    else:
        y50 = 0.01 * diameter  # fallback

    y50 = max(y50, 0.001)
    return (pu, y50)


def _api_py_clay(diameter: float, depth: float,
                 layer: SoilLayer) -> Tuple[float, float]:
    """Compute p-y parameters for soft clay per Matlock (1970).

    Args:
        diameter: Pile diameter (inches).
        depth:    Depth below mudline (inches).
        layer:    SoilLayer with clay properties (su, eps_50).

    Returns:
        Tuple of (pu, y50) in kip/in and inches.

    Reference:
        Matlock, H. (1970). "Correlations for Design of Laterally Loaded
        Piles in Soft Clay." OTC 1204.
    """
    su = layer.su  # ksi
    gamma = layer.gamma  # kip/in³
    J = 0.5  # Matlock's J factor (0.25-0.5, typical 0.5)

    # Ultimate soil resistance
    # Shallow: Np = 3 + gamma*z/su + J*z/D
    # Deep: Np = 9
    if su > 0 and diameter > 0:
        Np_shallow = 3.0 + (gamma * depth) / su + J * depth / diameter
    else:
        Np_shallow = 3.0
    Np = min(Np_shallow, 9.0)

    pu = Np * su * diameter  # kip/in (force per unit length)
    pu = max(pu, 0.001)

    # y50 = 2.5 * eps_50 * D
    y50 = 2.5 * layer.eps_50 * diameter
    y50 = max(y50, 0.001)

    return (pu, y50)


def _api_k_sand(phi: float) -> float:
    """Approximate initial modulus of subgrade reaction for sand (kip/in³).

    From API RP 2GEO Table 8.7-1 (interpolated).

    Args:
        phi: Friction angle in degrees.

    Returns:
        k in kip/in³.
    """
    # Approximate linear interpolation of API table
    # phi: 25→5, 30→10, 35→25, 40→50 (lb/in³ then /1000 for kip/in³)
    # Actually the API table gives k in lb/in³: 25→20, 30→42, 35→70, 40→140
    # Convert to kip/in³
    if phi <= 25:
        k_pci = 20.0
    elif phi <= 30:
        k_pci = 20.0 + (42.0 - 20.0) * (phi - 25.0) / 5.0
    elif phi <= 35:
        k_pci = 42.0 + (70.0 - 42.0) * (phi - 30.0) / 5.0
    elif phi <= 40:
        k_pci = 70.0 + (140.0 - 70.0) * (phi - 35.0) / 5.0
    else:
        k_pci = 140.0

    return k_pci / 1000.0  # Convert lb/in³ to kip/in³


def compute_tult(diameter: float, depth: float,
                 soil_layer: SoilLayer) -> float:
    """Compute ultimate unit skin friction (tult) for axially loaded piles.

    For clay: alpha method (FHWA-NHI-10-016 Section 9.3.2).
    For sand: beta method (FHWA-NHI-10-016 Section 9.3.3).

    Args:
        diameter:    Pile diameter (inches).
        depth:       Depth to midpoint of layer (inches).
        soil_layer:  SoilLayer definition.

    Returns:
        Ultimate unit skin friction tult (kip/in) — force per unit length
        of pile. Multiply by tributary length for total spring capacity.

    Reference:
        FHWA-NHI-10-016: Drilled Shafts, Chapter 9.
        AASHTO LRFD Section 10.8.3.5.
    """
    perimeter = math.pi * diameter

    if soil_layer.soil_type == "clay":
        # Alpha method: f_s = alpha * su
        su = soil_layer.su  # ksi
        # Alpha from O'Neill & Reese (1999):
        # su/pa <= 1.5: alpha = 0.55
        # su/pa > 1.5: alpha = 0.55 - 0.1*(su/pa - 1.5), min 0.45
        pa = 0.0145  # atmospheric pressure in ksi (~14.7 psi)
        su_over_pa = su / pa if pa > 0 else 0
        if su_over_pa <= 1.5:
            alpha = 0.55
        else:
            alpha = max(0.45, 0.55 - 0.1 * (su_over_pa - 1.5))

        f_s = alpha * su  # ksi
    else:
        # Beta method: f_s = beta * sigma_v'
        sigma_v = soil_layer.gamma * depth  # ksi
        phi_rad = math.radians(soil_layer.phi)

        # Beta = K * tan(delta), where K ~ K0 = 1-sin(phi) for drilled shafts
        # delta ~ phi for rough interface
        K0 = 1.0 - math.sin(phi_rad)
        beta = K0 * math.tan(phi_rad)
        # FHWA limits beta: 0.25 to 1.2 for drilled shafts
        beta = max(0.25, min(1.2, beta))

        f_s = beta * sigma_v  # ksi

    # tult = f_s * perimeter (force per unit length of pile)
    tult = f_s * perimeter
    return max(tult, 0.001)


def compute_qult(diameter: float, depth: float,
                 soil_layer: SoilLayer) -> float:
    """Compute ultimate tip bearing capacity.

    For clay: q_p = Nc * su (Nc = 9 for deep foundations).
    For sand: q_p = Nq * sigma_v' (Reese & O'Neill / FHWA method).

    Args:
        diameter:    Pile diameter (inches).
        depth:       Depth to pile tip (inches).
        soil_layer:  SoilLayer at tip.

    Returns:
        Ultimate tip resistance qult (kip) — total force at tip.

    Reference:
        Reese & O'Neill (1988). FHWA-HI-88-042.
        FHWA-NHI-10-016: Drilled Shafts, Chapter 13.
        AASHTO LRFD 10.8.3.5.
    """
    area = math.pi * (diameter / 2.0) ** 2  # in²

    if soil_layer.soil_type == "clay":
        # Bearing capacity: qp = Nc * su
        # Nc = 6[1 + 0.2(Z/D)] <= 9 for drilled shafts
        Nc = min(9.0, 6.0 * (1.0 + 0.2 * depth / diameter)) if diameter > 0 else 9.0
        qp = Nc * soil_layer.su  # ksi
    else:
        # Sand: q_p = 0.6 * Nq * sigma_v' (FHWA drilled shaft method)
        sigma_v = soil_layer.gamma * depth  # ksi
        phi_rad = math.radians(soil_layer.phi)
        # Nq from Vesic (1977), simplified
        Nq = math.exp(math.pi * math.tan(phi_rad)) * \
             (math.tan(math.radians(45 + soil_layer.phi / 2))) ** 2
        # FHWA reduction for drilled shafts
        qp = 0.6 * Nq * sigma_v
        # Limit to 50 tsf ≈ 0.694 ksi (AASHTO/FHWA recommendation)
        qp = min(qp, 0.694)

    qult = qp * area  # kip
    return max(qult, 0.001)


# ============================================================================
# BEARING MATERIALS
# ============================================================================

def elastomeric_shear(tag: int, G: float, A: float, h: float) -> int:
    """Define elastic material for elastomeric bearing pad shear stiffness.

    Stiffness: k = G * A / h (kip/in).

    Args:
        tag: Material tag.
        G:   Shear modulus of elastomer (ksi). Typical: 0.080-0.175 ksi
             (AASHTO Table 14.7.6.2-1).
        A:   Plan area of elastomer (in²).
        h:   Total elastomer thickness (in), sum of all layers.

    Returns:
        Material tag.

    Reference:
        AASHTO LRFD 14.7.6.2: Design of Elastomeric Bearings.
    """
    k = G * A / h  # kip/in
    ops.uniaxialMaterial('Elastic', tag, k)
    return tag


def friction_model(tag: int, mu_slow: float = 0.06, mu_fast: float = 0.10,
                   rate: float = 0.001) -> int:
    """Define velocity-dependent friction material for sliding bearings.

    Models the transition from static (slow) to kinetic (fast) friction,
    typical of PTFE/stainless steel sliding surfaces.

    Note: OpenSees uses ``frictionModel`` for friction pendulum elements.
    For simpler modeling, this uses a VelDepMultiLinear or equivalent.
    Here we use the Elastic material as a simplified placeholder with
    the slow friction coefficient, since VelDependent friction models
    are element-level in OpenSees (not uniaxial materials).

    For full friction pendulum modeling, use the SingleFPBearing element
    or FlatSliderBearing element directly.

    Args:
        tag:      Material tag.
        mu_slow:  Coefficient of friction at low velocity. Default: 0.06.
        mu_fast:  Coefficient of friction at high velocity. Default: 0.10.
        rate:     Rate parameter for velocity transition (in/sec). Default: 0.001.

    Returns:
        Material tag (friction coefficient stored as stiffness proxy for
        zero-length spring modeling: F = mu * N, where N is applied separately).

    Reference:
        AASHTO LRFD 14.7.2: PTFE Sliding Bearings.
        Constantinou, M.C. et al. (1990). "Teflon Bearings in Base Isolation."
    """
    # For zero-length springs: use Flat Slider or manually apply
    # friction * normal_force. Store mu_slow as the effective friction
    # coefficient in an Elastic material for simplified analysis.
    # Proper implementation uses frictionModel + bearing element.
    ops.uniaxialMaterial('Elastic', tag, mu_slow)
    return tag


def compression_only(tag: int, k: float) -> int:
    """Define elastic-no-tension (ENT) material.

    For bearings that only transmit compression (e.g., concrete on concrete,
    rocker bearings, expansion bearings under uplift).

    Args:
        tag: Material tag.
        k:   Compressive stiffness (kip/in).

    Returns:
        Material tag.

    Reference:
        AASHTO LRFD 14.8: Anchorage and Uplift.
    """
    ops.uniaxialMaterial('ENT', tag, k)
    return tag

"""Foundation modeling tool for OpenSees FEA models.

Creates nonlinear foundation models from natural language descriptions of
foundation type and site conditions. Supports drilled shafts, driven pile
groups, spread footings, and pile bents.

Soil spring formulations per:
- AASHTO LRFD Bridge Design Specifications, 10th Edition
- FHWA Drilled Shaft Manual (FHWA-NHI-10-016)
- API RP 2GEO (offshore — widely used for p-y curves)
- Matlock (1970) — soft clay p-y curves
- Reese et al. (1975) — stiff clay p-y curves

Units: kip-inch-second internally. Accepts ft input, converts internally.
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
KSF_TO_KSI = 1.0 / 144.0  # ksf → ksi
PCF_TO_PCI = 1.0 / 1728.0  # pcf → pci (lb/ft³ → lb/in³)
KCF_TO_KCI = 1.0 / 1728.0  # kcf → kci
GAMMA_WATER_PCI = 62.4 / 1728.0  # lb/in³

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

class SoilType(str, Enum):
    SOFT_CLAY = "soft_clay"
    STIFF_CLAY = "stiff_clay"
    SAND = "sand"
    ROCK = "rock"


class FoundationType(str, Enum):
    DRILLED_SHAFT = "drilled_shaft"
    DRIVEN_PILE_GROUP = "driven_pile_group"
    SPREAD_FOOTING = "spread_footing"
    PILE_BENT = "pile_bent"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SoilLayer:
    """Single soil layer in a profile.

    Attributes:
        soil_type: Classification (soft_clay, stiff_clay, sand, rock).
        top_depth_ft: Depth to top of layer from ground surface (ft).
        thickness_ft: Layer thickness (ft).
        su_ksf: Undrained shear strength (ksf) — clays.
        phi_deg: Friction angle (degrees) — sands.
        gamma_pcf: Total unit weight (pcf).
        N_spt: SPT blow count (blows/ft).
        eps50: Strain at 50% strength — clays (default from su).
        k_py_pci: Initial modulus of subgrade reaction for p-y (pci) — sand.
        qu_ksf: Unconfined compressive strength (ksf) — rock.
    """
    soil_type: str
    top_depth_ft: float
    thickness_ft: float
    su_ksf: float = 0.0
    phi_deg: float = 0.0
    gamma_pcf: float = 120.0
    N_spt: int = 0
    eps50: float | None = None
    k_py_pci: float | None = None
    qu_ksf: float = 0.0

    @property
    def bot_depth_ft(self) -> float:
        return self.top_depth_ft + self.thickness_ft

    @property
    def top_depth_in(self) -> float:
        return self.top_depth_ft * FT_TO_IN

    @property
    def bot_depth_in(self) -> float:
        return self.bot_depth_ft * FT_TO_IN

    @property
    def su_ksi(self) -> float:
        return self.su_ksf * KSF_TO_KSI

    @property
    def gamma_pci(self) -> float:
        return self.gamma_pcf * PCF_TO_PCI

    @property
    def qu_ksi(self) -> float:
        return self.qu_ksf * KSF_TO_KSI

    def get_eps50(self) -> float:
        """Return eps50; estimate from su if not provided (FHWA Table)."""
        if self.eps50 is not None:
            return self.eps50
        # Matlock recommended values based on su
        su = self.su_ksf
        if su <= 0.5:
            return 0.02
        elif su <= 1.0:
            return 0.01
        elif su <= 2.0:
            return 0.007
        elif su <= 4.0:
            return 0.005
        else:
            return 0.004

    def get_k_py(self) -> float:
        """Initial modulus of subgrade reaction (pci) for sand p-y.

        Per API RP 2GEO Table for submerged sand.
        """
        if self.k_py_pci is not None:
            return self.k_py_pci
        # Approximate from friction angle (pci)
        phi = self.phi_deg
        if phi <= 25:
            return 20.0
        elif phi <= 30:
            return 35.0
        elif phi <= 35:
            return 55.0
        elif phi <= 40:
            return 90.0
        else:
            return 115.0


@dataclass
class SiteProfile:
    """Complete site geotechnical profile."""
    layers: list[SoilLayer]
    gwt_depth_ft: float = 10.0  # groundwater table depth from surface
    scour: dict | None = None   # {water_crossing: bool, depth_ft: float}

    @property
    def gwt_depth_in(self) -> float:
        return self.gwt_depth_ft * FT_TO_IN

    def layer_at_depth(self, depth_ft: float) -> SoilLayer:
        """Return the soil layer at a given depth."""
        for layer in self.layers:
            if layer.top_depth_ft <= depth_ft < layer.bot_depth_ft:
                return layer
        # Below last layer — return last
        return self.layers[-1]

    def effective_vertical_stress(self, depth_ft: float) -> float:
        """Compute effective vertical stress at depth (ksi).

        Integrates gamma through layers, subtracts pore pressure below GWT.
        """
        sigma_v = 0.0
        prev_depth = 0.0
        for layer in sorted(self.layers, key=lambda l: l.top_depth_ft):
            top = max(layer.top_depth_ft, prev_depth)
            bot = min(layer.bot_depth_ft, depth_ft)
            if top >= bot:
                continue
            dz = (bot - top) * FT_TO_IN
            sigma_v += layer.gamma_pci * dz
            prev_depth = bot
            if bot >= depth_ft:
                break

        # Pore pressure below GWT
        if depth_ft > self.gwt_depth_ft:
            u = GAMMA_WATER_PCI * (depth_ft - self.gwt_depth_ft) * FT_TO_IN
            sigma_v -= u

        return max(sigma_v, 0.0)


@dataclass
class FoundationModel:
    """Complete foundation model output.

    All tags are integers suitable for direct use in OpenSees commands.
    All coordinates and forces are in kip-inch-second units.
    """
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)
    springs: list[dict] = field(default_factory=list)
    materials: list[dict] = field(default_factory=list)
    boundary_conditions: list[dict] = field(default_factory=list)
    top_node: int = 0
    base_node: int = 0
    capacity: dict = field(default_factory=dict)
    cases: dict = field(default_factory=dict)
    opensees_commands: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"FoundationModel: {len(self.nodes)} nodes, "
            f"{len(self.elements)} elements, {len(self.springs)} springs, "
            f"{len(self.materials)} materials | "
            f"top_node={self.top_node}, base_node={self.base_node}"
        )


# ===================================================================
# SOIL SPRING COMPUTATIONS
# ===================================================================

class PYCurve:
    """p-y curve computation for lateral soil resistance.

    References:
        - Matlock (1970): Soft clay
        - Reese et al. (1975): Stiff clay
        - API RP 2GEO: Sand
        - FHWA-NHI-10-016 Ch. 9: Drilled shafts
    """

    @staticmethod
    def soft_clay_matlock(
        depth_in: float,
        su_ksi: float,
        eps50: float,
        diameter_in: float,
        gamma_pci: float,
        sigma_v_ksi: float | None = None,
    ) -> dict:
        """Matlock (1970) soft clay p-y curve.

        pu increases from 3*su*D at surface to 9*su*D at depth.
        Transition depth: J*su*D / (gamma*D + J*su) where J ≈ 0.5 (soft) or 0.25.

        Args:
            depth_in: Depth below ground (in).
            su_ksi: Undrained shear strength (ksi).
            eps50: Strain at 50% strength.
            diameter_in: Shaft/pile diameter (in).
            gamma_pci: Effective unit weight (pci).
            sigma_v_ksi: Effective overburden (ksi), computed if None.

        Returns:
            dict with pu (kip/in), y50 (in), curve points [(y, p), ...].
        """
        D = diameter_in
        J = 0.5  # Matlock recommended for soft clay

        if sigma_v_ksi is None:
            sigma_v_ksi = gamma_pci * depth_in

        # Ultimate resistance
        pu_shallow = (3.0 + sigma_v_ksi / su_ksi + J * depth_in / D) * su_ksi * D
        pu_deep = 9.0 * su_ksi * D
        pu = min(pu_shallow, pu_deep)

        y50 = 2.5 * eps50 * D

        # Curve: p/pu = 0.5 * (y/y50)^(1/3) up to p = pu at y = 8*y50
        points = []
        for ratio in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]:
            y = ratio * y50
            if ratio == 0:
                p = 0.0
            elif ratio <= 8.0:
                p = 0.5 * (ratio ** (1.0 / 3.0)) * pu
            else:
                p = pu  # Residual
            points.append((y, min(p, pu)))

        return {"pu": pu, "y50": y50, "points": points}

    @staticmethod
    def stiff_clay_reese(
        depth_in: float,
        su_ksi: float,
        eps50: float,
        diameter_in: float,
        gamma_pci: float,
        sigma_v_ksi: float | None = None,
    ) -> dict:
        """Reese et al. (1975) stiff clay p-y curve (no free water).

        Args:
            depth_in: Depth below ground (in).
            su_ksi: Undrained shear strength (ksi).
            eps50: Strain at 50% strength.
            diameter_in: Shaft/pile diameter (in).
            gamma_pci: Effective unit weight (pci).

        Returns:
            dict with pu, y50, curve points.
        """
        D = diameter_in

        if sigma_v_ksi is None:
            sigma_v_ksi = gamma_pci * depth_in

        # Ultimate resistance — Reese formulation
        # Shallow wedge failure
        ca = su_ksi  # adhesion ≈ su for stiff clay
        pu_wedge = (2.0 * ca * D
                    + sigma_v_ksi * D
                    + 2.83 * su_ksi * depth_in)
        # Deep flow-around failure
        pu_flow = 11.0 * su_ksi * D
        pu = min(pu_wedge, pu_flow)

        # Prevent zero
        if pu <= 0:
            pu = su_ksi * D

        y50 = eps50 * D

        # Parabolic curve: p = 0.5*pu*(y/y50)^0.25 up to 16*y50
        points = []
        for ratio in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]:
            y = ratio * y50
            if ratio == 0:
                p = 0.0
            elif ratio <= 16.0:
                p = 0.5 * pu * (ratio ** 0.25)
            else:
                p = pu
            points.append((y, min(p, pu)))

        return {"pu": pu, "y50": y50, "points": points}

    @staticmethod
    def sand_api(
        depth_in: float,
        phi_deg: float,
        diameter_in: float,
        gamma_pci: float,
        k_py_pci: float,
    ) -> dict:
        """API RP 2GEO sand p-y curve.

        pu from passive pressure coefficients; initial modulus k*z.

        Args:
            depth_in: Depth below ground (in).
            phi_deg: Friction angle (degrees).
            diameter_in: Shaft/pile diameter (in).
            gamma_pci: Effective unit weight (pci).
            k_py_pci: Initial modulus of subgrade reaction (pci).

        Returns:
            dict with pu, k_initial, curve points.
        """
        D = diameter_in
        phi = math.radians(phi_deg)

        # Coefficients per API
        beta_angle = math.pi / 4.0 + phi / 2.0
        Kp = math.tan(beta_angle) ** 2
        Ka = math.tan(math.pi / 4.0 - phi / 2.0) ** 2
        K0 = 1.0 - math.sin(phi)

        alpha_a = phi / 2.0

        # Shallow failure (wedge)
        C1 = (Kp - Ka) * math.tan(phi) + K0 * math.tan(phi) * math.tan(beta_angle)
        C2 = Kp - Ka
        C3 = Kp ** 2 * (Kp + K0 * math.tan(phi)) - Ka

        sigma_v = gamma_pci * depth_in

        # Shallow
        pu_shallow = (C1 * depth_in + C2 * D) * sigma_v
        # Deep
        pu_deep = C3 * sigma_v * D

        pu = min(pu_shallow, pu_deep)
        pu = max(pu, 0.001)  # avoid zero

        # Hyperbolic tangent curve: p = A*pu*tanh(k*z*y/(A*pu))
        # A = 0.9 for cyclic, 3.0-0.8*z/D for static (use static)
        A = max(0.9, 3.0 - 0.8 * depth_in / D)
        k_init = k_py_pci * depth_in  # kip/in/in → increases with depth

        points = []
        for y in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            if y == 0 or A * pu == 0:
                p = 0.0
            else:
                arg = k_init * y / (A * pu) if A * pu > 0 else 0.0
                p = A * pu * math.tanh(arg)
            points.append((y, p))

        return {"pu": pu, "k_initial": k_init, "points": points}

    @staticmethod
    def rock(
        depth_in: float,
        qu_ksi: float,
        diameter_in: float,
    ) -> dict:
        """Simplified rock p-y curve.

        pu = qu * D (unconfined compressive strength × diameter).
        Linear-elastic up to yield, then constant.

        Args:
            depth_in: Depth below ground (in).
            qu_ksi: Unconfined compressive strength (ksi).
            diameter_in: Shaft/pile diameter (in).

        Returns:
            dict with pu, curve points.
        """
        D = diameter_in
        pu = qu_ksi * D

        # Initial stiffness: 100*qu*D / D = 100*qu per unit length
        k_rock = 100.0 * qu_ksi
        y_yield = pu / k_rock if k_rock > 0 else 0.01

        points = [
            (0.0, 0.0),
            (y_yield * 0.5, pu * 0.5),
            (y_yield, pu),
            (y_yield * 5.0, pu),
            (y_yield * 20.0, pu),
        ]

        return {"pu": pu, "y_yield": y_yield, "points": points}


class TZCurve:
    """t-z curve computation for skin friction resistance.

    References:
        - API RP 2GEO
        - AASHTO 10th Ed. Section 10.8
        - FHWA-NHI-10-016 Ch. 13
    """

    @staticmethod
    def clay_alpha(
        su_ksi: float,
        diameter_in: float,
        spacing_in: float,
    ) -> dict:
        """Alpha method for clay skin friction.

        tult = alpha * su (per unit length = alpha * su * pi * D * dz)
        alpha from API: alpha = 0.5*(su/sigma_v')^-0.5 for su/sigma_v' <= 1
                        alpha = 0.5*(su/sigma_v')^-0.25 for su/sigma_v' > 1
        Simplified: alpha ≈ min(1.0, 0.55 - 0.1*(su_ksf - 1.5)) per FHWA

        Args:
            su_ksi: Undrained shear strength (ksi).
            diameter_in: Shaft diameter (in).
            spacing_in: Spring spacing (in).

        Returns:
            dict with tult (kip/in), z_peak (in), curve points [(z, t)].
        """
        su_ksf = su_ksi / KSF_TO_KSI

        # Alpha per AASHTO/FHWA simplified
        if su_ksf <= 1.5:
            alpha = 0.55
        else:
            alpha = max(0.25, 0.55 - 0.1 * (su_ksf - 1.5))
        alpha = min(1.0, alpha)

        # Unit skin friction
        fs_ksi = alpha * su_ksi
        perimeter = math.pi * diameter_in
        tult = fs_ksi * perimeter * spacing_in  # kip (force per spring)

        # t-z curve shape per API
        z_peak = 0.01 * diameter_in  # ~1% of diameter

        points = [
            (0.0, 0.0),
            (0.16 * z_peak, 0.30 * tult),
            (0.31 * z_peak, 0.50 * tult),
            (0.57 * z_peak, 0.75 * tult),
            (0.80 * z_peak, 0.90 * tult),
            (1.00 * z_peak, 1.00 * tult),
            (2.00 * z_peak, 0.90 * tult),  # Post-peak softening
            (5.00 * z_peak, 0.70 * tult),
        ]

        return {"tult": tult, "z_peak": z_peak, "alpha": alpha, "points": points}

    @staticmethod
    def sand_beta(
        sigma_v_ksi: float,
        phi_deg: float,
        diameter_in: float,
        spacing_in: float,
        depth_in: float,
    ) -> dict:
        """Beta method for sand skin friction.

        tult = beta * sigma_v' (per unit area)
        beta from AASHTO Table 10.8.3.5.2b-1 (simplified).

        Args:
            sigma_v_ksi: Effective vertical stress (ksi).
            phi_deg: Friction angle (degrees).
            diameter_in: Shaft diameter (in).
            spacing_in: Spring spacing (in).
            depth_in: Depth below ground (in).

        Returns:
            dict with tult (kip/in), z_peak (in), curve points.
        """
        # Beta from AASHTO — increases with friction angle, decreases with depth
        depth_ft = depth_in / FT_TO_IN
        N_corr = depth_ft  # simplified depth correction

        if phi_deg <= 25:
            beta_base = 0.25
        elif phi_deg <= 30:
            beta_base = 0.35
        elif phi_deg <= 35:
            beta_base = 0.50
        elif phi_deg <= 40:
            beta_base = 0.70
        else:
            beta_base = 0.80

        # Depth reduction (beta decreases with depth, AASHTO)
        if depth_ft > 60:
            beta = beta_base * 0.7
        elif depth_ft > 30:
            beta = beta_base * 0.85
        else:
            beta = beta_base

        fs_ksi = beta * sigma_v_ksi
        perimeter = math.pi * diameter_in
        tult = fs_ksi * perimeter * spacing_in

        z_peak = 0.01 * diameter_in

        points = [
            (0.0, 0.0),
            (0.20 * z_peak, 0.35 * tult),
            (0.40 * z_peak, 0.60 * tult),
            (0.60 * z_peak, 0.80 * tult),
            (0.80 * z_peak, 0.95 * tult),
            (1.00 * z_peak, 1.00 * tult),
            (3.00 * z_peak, 1.00 * tult),  # No softening in sand
        ]

        return {"tult": tult, "z_peak": z_peak, "beta": beta, "points": points}


class QZCurve:
    """Q-z curve computation for end bearing resistance.

    References:
        - Reese & O'Neill (1988): Drilled shafts
        - AASHTO 10th Ed. Section 10.8.3.5
    """

    @staticmethod
    def clay(su_ksi: float, diameter_in: float) -> dict:
        """Clay end bearing Q-z curve.

        qult = Nc * su, Nc ≈ 9 per AASHTO/Reese & O'Neill.

        Args:
            su_ksi: Undrained shear strength at tip (ksi).
            diameter_in: Shaft diameter (in).

        Returns:
            dict with Qult (kip), z_peak (in), curve points [(z, Q)].
        """
        Nc = 9.0
        area = math.pi * diameter_in ** 2 / 4.0
        qult = Nc * su_ksi
        Qult = qult * area  # Total end bearing (kip)

        z_peak = 0.05 * diameter_in  # 5% of diameter per Reese & O'Neill

        points = [
            (0.0, 0.0),
            (0.002 * diameter_in, 0.15 * Qult),
            (0.013 * diameter_in, 0.50 * Qult),
            (0.042 * diameter_in, 0.85 * Qult),
            (z_peak, Qult),
            (0.10 * diameter_in, Qult),
        ]

        return {"Qult": Qult, "z_peak": z_peak, "Nc": Nc, "points": points}

    @staticmethod
    def sand(sigma_v_ksi: float, phi_deg: float, diameter_in: float) -> dict:
        """Sand end bearing Q-z curve.

        qult = Nq * sigma_v', Nq from AASHTO tables.

        Args:
            sigma_v_ksi: Effective vertical stress at tip (ksi).
            phi_deg: Friction angle at tip (degrees).
            diameter_in: Shaft diameter (in).

        Returns:
            dict with Qult (kip), z_peak (in), Nq, curve points.
        """
        # Nq from AASHTO (simplified Meyerhof/Berezantzev)
        if phi_deg <= 25:
            Nq = 8.0
        elif phi_deg <= 28:
            Nq = 12.0
        elif phi_deg <= 30:
            Nq = 20.0
        elif phi_deg <= 33:
            Nq = 30.0
        elif phi_deg <= 36:
            Nq = 50.0
        elif phi_deg <= 40:
            Nq = 80.0
        else:
            Nq = 120.0

        area = math.pi * diameter_in ** 2 / 4.0

        # Limit qult to 100 ksf ≈ 0.694 ksi per AASHTO
        qult = min(Nq * sigma_v_ksi, 0.694)
        Qult = qult * area

        z_peak = 0.05 * diameter_in

        points = [
            (0.0, 0.0),
            (0.002 * diameter_in, 0.10 * Qult),
            (0.010 * diameter_in, 0.40 * Qult),
            (0.030 * diameter_in, 0.75 * Qult),
            (z_peak, Qult),
            (0.10 * diameter_in, Qult),
        ]

        return {"Qult": Qult, "z_peak": z_peak, "Nq": Nq, "points": points}

    @staticmethod
    def rock(qu_ksi: float, diameter_in: float) -> dict:
        """Rock end bearing Q-z curve.

        qult = 2.5 * qu per FHWA-NHI-10-016.

        Args:
            qu_ksi: Unconfined compressive strength (ksi).
            diameter_in: Shaft diameter (in).

        Returns:
            dict with Qult (kip), z_peak (in), curve points.
        """
        area = math.pi * diameter_in ** 2 / 4.0
        qult = 2.5 * qu_ksi
        Qult = qult * area

        z_peak = 0.01 * diameter_in  # Rock — very stiff

        points = [
            (0.0, 0.0),
            (z_peak * 0.25, 0.40 * Qult),
            (z_peak * 0.50, 0.70 * Qult),
            (z_peak, Qult),
            (z_peak * 5.0, Qult),
        ]

        return {"Qult": Qult, "z_peak": z_peak, "points": points}


# ===================================================================
# TAG ALLOCATOR — ensures unique OpenSees tags
# ===================================================================

class TagAllocator:
    """Simple sequential tag allocator for OpenSees objects."""

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
# SECTION BUILDER — shaft/pile cross section
# ===================================================================

def build_circular_section(
    tag_alloc: TagAllocator,
    diameter_in: float,
    fc_ksi: float,
    fy_ksi: float = 60.0,
    n_bars: int = 12,
    bar_size: str = "#10",
    cover_in: float = 3.0,
) -> tuple[list[dict], int]:
    """Build OpenSees fiber section for circular RC section.

    Uses Concrete01 (confined/unconfined) and Steel02 materials.

    Returns:
        (materials_list, section_tag)
    """
    materials = []

    # Concrete — Mander confined model (simplified)
    # Unconfined concrete
    fc = fc_ksi
    ec0 = 0.002
    fcu = 0.0  # spalling
    ecu = 0.005

    unconf_tag = tag_alloc.next()
    materials.append({
        "tag": unconf_tag,
        "type": "Concrete01",
        "params": [-fc, -ec0, -fcu, -ecu],
        "description": f"Unconfined concrete f'c={fc} ksi",
    })

    # Confined concrete (simplified Mander — ~1.3x f'c for typical confinement)
    fcc = 1.3 * fc
    ecc0 = 0.004
    fccu = 0.2 * fcc
    eccu = 0.015

    conf_tag = tag_alloc.next()
    materials.append({
        "tag": conf_tag,
        "type": "Concrete01",
        "params": [-fcc, -ecc0, -fccu, -eccu],
        "description": f"Confined concrete f'cc={fcc:.1f} ksi",
    })

    # Steel reinforcement
    bar_area = REBAR_AREAS.get(bar_size, 1.0)
    Es = 29000.0  # ksi

    steel_tag = tag_alloc.next()
    materials.append({
        "tag": steel_tag,
        "type": "Steel02",
        "params": [fy_ksi, Es, 0.01],  # Fy, E, b (strain hardening)
        "description": f"Steel {bar_size} Fy={fy_ksi} ksi",
    })

    # Fiber section
    section_tag = tag_alloc.next()
    radius = diameter_in / 2.0
    core_radius = radius - cover_in
    bar_radius = core_radius - REBAR_DIAMETERS.get(bar_size, 1.0) / 2.0

    materials.append({
        "tag": section_tag,
        "type": "FiberSection",
        "params": {
            "core": {"material": conf_tag, "radius": core_radius, "nfr": 8, "nft": 16},
            "cover": {"material": unconf_tag, "outer_radius": radius, "inner_radius": core_radius, "nfr": 2, "nft": 16},
            "steel": {"material": steel_tag, "n_bars": n_bars, "bar_area": bar_area, "radius": bar_radius},
        },
        "description": f"Circular RC section D={diameter_in} in, {n_bars}-{bar_size}",
    })

    return materials, section_tag


# ===================================================================
# DRILLED SHAFT BUILDER
# ===================================================================

def _spring_depths(
    length_in: float,
    spacing_in: float = 12.0,
    scour_depth_in: float = 0.0,
) -> list[float]:
    """Generate spring depths along shaft, skipping scour zone.

    Args:
        length_in: Total embedded length (in).
        spacing_in: Spring spacing (in), default 12 (1 ft).
        scour_depth_in: Depth of scour below ground (in). Springs above
            this depth are omitted.

    Returns:
        List of depths (in) from ground surface where springs are placed.
    """
    depths = []
    d = spacing_in / 2.0  # First spring at half-spacing
    while d < length_in:
        if d >= scour_depth_in:
            depths.append(d)
        d += spacing_in
    return depths


def _build_py_spring(
    layer: SoilLayer,
    depth_in: float,
    diameter_in: float,
    spacing_in: float,
    site: SiteProfile,
    multiplier: float = 1.0,
) -> dict:
    """Build a single p-y spring at given depth.

    Selects formulation based on soil type. Returns curve data with
    multiplier applied for upper/lower bound analysis.
    """
    sigma_v = site.effective_vertical_stress(depth_in / FT_TO_IN)

    if layer.soil_type in (SoilType.SOFT_CLAY, "soft_clay"):
        curve = PYCurve.soft_clay_matlock(
            depth_in=depth_in,
            su_ksi=layer.su_ksi,
            eps50=layer.get_eps50(),
            diameter_in=diameter_in,
            gamma_pci=layer.gamma_pci,
            sigma_v_ksi=sigma_v,
        )
    elif layer.soil_type in (SoilType.STIFF_CLAY, "stiff_clay"):
        curve = PYCurve.stiff_clay_reese(
            depth_in=depth_in,
            su_ksi=layer.su_ksi,
            eps50=layer.get_eps50(),
            diameter_in=diameter_in,
            gamma_pci=layer.gamma_pci,
            sigma_v_ksi=sigma_v,
        )
    elif layer.soil_type in (SoilType.SAND, "sand"):
        curve = PYCurve.sand_api(
            depth_in=depth_in,
            phi_deg=layer.phi_deg,
            diameter_in=diameter_in,
            gamma_pci=layer.gamma_pci,
            k_py_pci=layer.get_k_py(),
        )
    elif layer.soil_type in (SoilType.ROCK, "rock"):
        curve = PYCurve.rock(
            depth_in=depth_in,
            qu_ksi=layer.qu_ksi,
            diameter_in=diameter_in,
        )
    else:
        raise ValueError(f"Unknown soil type: {layer.soil_type}")

    # Apply multiplier (upper bound = 2x, lower bound = 0.5x)
    scaled_points = [(y, p * multiplier) for y, p in curve["points"]]
    curve["points"] = scaled_points
    curve["pu"] = curve.get("pu", 0) * multiplier
    curve["multiplier"] = multiplier
    curve["formulation"] = layer.soil_type

    return curve


def _build_tz_spring(
    layer: SoilLayer,
    depth_in: float,
    diameter_in: float,
    spacing_in: float,
    site: SiteProfile,
    multiplier: float = 1.0,
) -> dict:
    """Build a single t-z spring at given depth."""
    if layer.soil_type in (SoilType.SOFT_CLAY, SoilType.STIFF_CLAY, "soft_clay", "stiff_clay"):
        curve = TZCurve.clay_alpha(
            su_ksi=layer.su_ksi,
            diameter_in=diameter_in,
            spacing_in=spacing_in,
        )
    elif layer.soil_type in (SoilType.SAND, "sand"):
        sigma_v = site.effective_vertical_stress(depth_in / FT_TO_IN)
        curve = TZCurve.sand_beta(
            sigma_v_ksi=sigma_v,
            phi_deg=layer.phi_deg,
            diameter_in=diameter_in,
            spacing_in=spacing_in,
            depth_in=depth_in,
        )
    elif layer.soil_type in (SoilType.ROCK, "rock"):
        # Rock skin friction — use alpha method with high su
        su_equiv = layer.qu_ksi / 2.0  # su ≈ qu/2
        curve = TZCurve.clay_alpha(
            su_ksi=su_equiv,
            diameter_in=diameter_in,
            spacing_in=spacing_in,
        )
    else:
        raise ValueError(f"Unknown soil type for t-z: {layer.soil_type}")

    # Apply multiplier
    scaled_points = [(z, t * multiplier) for z, t in curve["points"]]
    curve["points"] = scaled_points
    curve["tult"] = curve.get("tult", 0) * multiplier
    curve["multiplier"] = multiplier

    return curve


def build_drilled_shaft(
    params: dict,
    site: SiteProfile,
    tag_alloc: TagAllocator | None = None,
    spring_spacing_ft: float = 1.0,
) -> FoundationModel:
    """Build complete drilled shaft foundation model.

    Args:
        params: {
            diameter_ft: float,        # Shaft diameter (ft)
            length_ft: float,          # Embedded length (ft)
            above_grade_ft: float,     # Extension above ground (ft), default 0
            fc_ksi: float,             # Concrete strength (ksi), default 4.0
            fy_ksi: float,             # Rebar yield strength (ksi), default 60
            n_bars: int,               # Number of longitudinal bars
            bar_size: str,             # e.g. "#10"
            cover_in: float,           # Clear cover (in), default 3
        }
        site: SiteProfile with soil layers and GWT.
        tag_alloc: Tag allocator (created if None).
        spring_spacing_ft: Spring spacing in ft (default 1.0).

    Returns:
        FoundationModel with upper_bound and lower_bound cases.
    """
    if tag_alloc is None:
        tag_alloc = TagAllocator(start=1000)

    # Extract params
    diameter_ft = params.get("diameter_ft", 5.0)
    length_ft = params.get("length_ft", 60.0)
    above_grade_ft = params.get("above_grade_ft", 0.0)
    fc_ksi = params.get("fc_ksi", 4.0)
    fy_ksi = params.get("fy_ksi", 60.0)
    n_bars = params.get("n_bars", 20)
    bar_size = params.get("bar_size", "#10")
    cover_in = params.get("cover_in", 3.0)

    diameter_in = diameter_ft * FT_TO_IN
    length_in = length_ft * FT_TO_IN
    above_grade_in = above_grade_ft * FT_TO_IN
    spacing_in = spring_spacing_ft * FT_TO_IN

    # Scour depth
    scour_depth_in = 0.0
    if site.scour and site.scour.get("water_crossing"):
        scour_depth_in = site.scour.get("depth_ft", 0.0) * FT_TO_IN

    # Build section
    section_mats, section_tag = build_circular_section(
        tag_alloc, diameter_in, fc_ksi, fy_ksi, n_bars, bar_size, cover_in,
    )

    model = FoundationModel()
    model.materials.extend(section_mats)

    # ---------------------------------------------------------------
    # Nodes: from top of shaft to tip
    # Top node at y = above_grade_in (above ground is positive y)
    # Ground surface at y = 0
    # Tip at y = -length_in
    # ---------------------------------------------------------------
    node_depths = []

    # Above ground nodes (if any)
    if above_grade_in > 0:
        node_depths.append(above_grade_in)  # top
        n_above = max(1, int(above_grade_in / spacing_in))
        for i in range(1, n_above):
            node_depths.append(above_grade_in - i * spacing_in)

    node_depths.append(0.0)  # ground surface

    # Below ground nodes at spring locations + tip
    spring_depths = _spring_depths(length_in, spacing_in, scour_depth_in)
    for d in spring_depths:
        node_depths.append(-d)
    node_depths.append(-length_in)  # tip

    # Deduplicate and sort descending (top to bottom)
    node_depths = sorted(set(node_depths), reverse=True)

    # Create node dicts
    node_tags = tag_alloc.next(len(node_depths))
    if isinstance(node_tags, int):
        node_tags = [node_tags]

    depth_to_node: dict[float, int] = {}
    for i, depth in enumerate(node_depths):
        tag = node_tags[i]
        model.nodes.append({"tag": tag, "x": 0.0, "y": depth, "z": 0.0})
        depth_to_node[depth] = tag

    model.top_node = node_tags[0]
    model.base_node = node_tags[-1]

    # ---------------------------------------------------------------
    # Elements: dispBeamColumn between consecutive nodes
    # ---------------------------------------------------------------
    n_ip = 5  # integration points
    for i in range(len(node_tags) - 1):
        elem_tag = tag_alloc.next()
        model.elements.append({
            "tag": elem_tag,
            "type": "dispBeamColumn",
            "nodes": [node_tags[i], node_tags[i + 1]],
            "n_ip": n_ip,
            "section": section_tag,
            "transform": "Linear",
        })

    # ---------------------------------------------------------------
    # Springs: p-y (lateral) + t-z (axial skin) at each spring depth
    # Q-z (end bearing) at tip
    # ---------------------------------------------------------------
    # Build for both upper and lower bound
    for case_name, multiplier in [("upper_bound", 2.0), ("lower_bound", 0.5)]:
        case_springs = []
        case_materials = []

        for spring_depth in spring_depths:
            depth_ft = spring_depth / FT_TO_IN
            layer = site.layer_at_depth(depth_ft)
            shaft_node = depth_to_node.get(-spring_depth)
            if shaft_node is None:
                continue

            # p-y spring (lateral, y-direction)
            py_data = _build_py_spring(
                layer, spring_depth, diameter_in, spacing_in, site, multiplier,
            )
            py_mat_tag = tag_alloc.next()
            case_materials.append({
                "tag": py_mat_tag,
                "type": "PySimple1",
                "params": {
                    "soilType": 1 if "clay" in layer.soil_type else 2,
                    "pult": py_data["pu"] * spacing_in,
                    "y50": py_data.get("y50", 0.1),
                    "Cd": 0.0,
                },
                "depth_in": spring_depth,
                "case": case_name,
                "description": f"p-y {layer.soil_type} @ {depth_ft:.1f} ft ({case_name})",
            })

            py_spring_tag = tag_alloc.next()
            case_springs.append({
                "tag": py_spring_tag,
                "node": shaft_node,
                "direction": "lateral",
                "material": py_mat_tag,
                "type": "PySimple1",
                "depth_ft": depth_ft,
                "case": case_name,
                "curve_data": py_data,
            })

            # t-z spring (axial skin friction)
            tz_data = _build_tz_spring(
                layer, spring_depth, diameter_in, spacing_in, site, multiplier,
            )
            tz_mat_tag = tag_alloc.next()
            case_materials.append({
                "tag": tz_mat_tag,
                "type": "TzSimple1",
                "params": {
                    "soilType": 1 if "clay" in layer.soil_type else 2,
                    "tult": tz_data["tult"],
                    "z50": tz_data["z_peak"] * 0.5,
                    "c": 0.0,
                },
                "depth_in": spring_depth,
                "case": case_name,
                "description": f"t-z {layer.soil_type} @ {depth_ft:.1f} ft ({case_name})",
            })

            tz_spring_tag = tag_alloc.next()
            case_springs.append({
                "tag": tz_spring_tag,
                "node": shaft_node,
                "direction": "axial",
                "material": tz_mat_tag,
                "type": "TzSimple1",
                "depth_ft": depth_ft,
                "case": case_name,
                "curve_data": tz_data,
            })

        # Q-z spring at tip
        tip_layer = site.layer_at_depth(length_ft)
        sigma_v_tip = site.effective_vertical_stress(length_ft)

        if tip_layer.soil_type in ("soft_clay", "stiff_clay"):
            qz_data = QZCurve.clay(tip_layer.su_ksi, diameter_in)
        elif tip_layer.soil_type == "sand":
            qz_data = QZCurve.sand(sigma_v_tip, tip_layer.phi_deg, diameter_in)
        elif tip_layer.soil_type == "rock":
            qz_data = QZCurve.rock(tip_layer.qu_ksi, diameter_in)
        else:
            qz_data = QZCurve.clay(tip_layer.su_ksi, diameter_in)

        # Apply multiplier to Q-z
        qz_data["Qult"] = qz_data["Qult"] * multiplier
        qz_data["points"] = [(z, Q * multiplier) for z, Q in qz_data["points"]]

        qz_mat_tag = tag_alloc.next()
        case_materials.append({
            "tag": qz_mat_tag,
            "type": "QzSimple1",
            "params": {
                "qzType": 1 if "clay" in tip_layer.soil_type else 2,
                "Qult": qz_data["Qult"],
                "z50": qz_data["z_peak"] * 0.5,
                "suction": 0.0,
            },
            "case": case_name,
            "description": f"Q-z {tip_layer.soil_type} at tip ({case_name})",
        })

        qz_spring_tag = tag_alloc.next()
        case_springs.append({
            "tag": qz_spring_tag,
            "node": model.base_node,
            "direction": "axial_tip",
            "material": qz_mat_tag,
            "type": "QzSimple1",
            "depth_ft": length_ft,
            "case": case_name,
            "curve_data": qz_data,
        })

        # Store case
        model.cases[case_name] = {
            "multiplier": multiplier,
            "springs": case_springs,
            "materials": case_materials,
        }

        # Add to main model (upper_bound as default)
        if case_name == "upper_bound":
            model.springs.extend(case_springs)
            model.materials.extend(case_materials)

    # Boundary condition: fixed at tip (or springs handle it)
    model.boundary_conditions.append({
        "node": model.base_node,
        "fixity": [0, 0, 0, 0, 0, 0],  # Springs provide restraint
        "description": "Tip node — restrained by Q-z and soil springs",
    })

    # Capacity estimate
    total_skin = sum(
        s["curve_data"]["tult"]
        for s in model.cases["lower_bound"]["springs"]
        if s["direction"] == "axial"
    )
    tip_bearing = model.cases["lower_bound"]["springs"][-1]["curve_data"]["Qult"]
    lateral_capacity = sum(
        s["curve_data"]["pu"] * spacing_in
        for s in model.cases["lower_bound"]["springs"]
        if s["direction"] == "lateral"
    ) * 0.5  # Rough estimate at working loads

    model.capacity = {
        "axial_kips": total_skin + tip_bearing,
        "lateral_kips": lateral_capacity,
        "moment_kip_ft": lateral_capacity * length_ft * 0.3,  # Approximate
        "skin_friction_kips": total_skin,
        "end_bearing_kips": tip_bearing,
        "note": "Lower bound values (φ-factored capacity requires AASHTO resistance factors)",
    }

    return model


# ===================================================================
# SPREAD FOOTING BUILDER
# ===================================================================

def build_spread_footing(
    params: dict,
    site: SiteProfile,
    tag_alloc: TagAllocator | None = None,
) -> FoundationModel:
    """Build spread footing with Winkler spring model.

    Uses compression-only (ENT material) zeroLength springs on a grid.

    Args:
        params: {
            length_ft: float,          # Footing length in plan (ft)
            width_ft: float,           # Footing width in plan (ft)
            depth_ft: float,           # Embedment depth (ft)
            thickness_ft: float,       # Footing thickness (ft), default 3
            subgrade_modulus_kcf: float, # kcf (default 100)
            n_springs_l: int,          # Springs along length (default 5)
            n_springs_w: int,          # Springs along width (default 5)
        }
        site: SiteProfile.
        tag_alloc: Tag allocator.

    Returns:
        FoundationModel with translational + rotational stiffness.
    """
    if tag_alloc is None:
        tag_alloc = TagAllocator(start=2000)

    L_ft = params.get("length_ft", 10.0)
    W_ft = params.get("width_ft", 10.0)
    D_ft = params.get("depth_ft", 4.0)
    thickness_ft = params.get("thickness_ft", 3.0)
    ks_kcf = params.get("subgrade_modulus_kcf", 100.0)
    n_L = params.get("n_springs_l", 5)
    n_W = params.get("n_springs_w", 5)

    L_in = L_ft * FT_TO_IN
    W_in = W_ft * FT_TO_IN
    D_in = D_ft * FT_TO_IN
    thickness_in = thickness_ft * FT_TO_IN

    # Subgrade modulus: kcf → kci
    ks_kci = ks_kcf * KCF_TO_KCI

    # Tributary area per spring
    trib_L = L_in / n_L
    trib_W = W_in / n_W
    trib_area = trib_L * trib_W

    # Spring stiffness (kip/in per spring)
    k_spring = ks_kci * trib_area

    model = FoundationModel()

    # Top node — center of footing at ground surface
    top_tag = tag_alloc.next()
    model.nodes.append({"tag": top_tag, "x": 0.0, "y": 0.0, "z": 0.0})
    model.top_node = top_tag

    # Spring grid at base of footing (y = -D_in - thickness_in ... or just -D_in)
    y_base = -D_in
    spring_nodes = []
    for i in range(n_L):
        for j in range(n_W):
            x = -L_in / 2.0 + trib_L * (i + 0.5)
            z = -W_in / 2.0 + trib_W * (j + 0.5)
            n_tag = tag_alloc.next()
            model.nodes.append({"tag": n_tag, "x": x, "y": y_base, "z": z})
            spring_nodes.append(n_tag)

    model.base_node = spring_nodes[0]

    # For upper/lower bound
    for case_name, mult in [("upper_bound", 2.0), ("lower_bound", 0.5)]:
        case_springs = []
        case_materials = []
        k_eff = k_spring * mult

        for s_node in spring_nodes:
            mat_tag = tag_alloc.next()
            case_materials.append({
                "tag": mat_tag,
                "type": "ENT",
                "params": [k_eff],
                "case": case_name,
                "description": f"Compression-only spring k={k_eff:.1f} kip/in ({case_name})",
            })

            spring_tag = tag_alloc.next()
            case_springs.append({
                "tag": spring_tag,
                "node": s_node,
                "direction": "vertical",
                "material": mat_tag,
                "type": "zeroLength",
                "case": case_name,
            })

        model.cases[case_name] = {
            "multiplier": mult,
            "springs": case_springs,
            "materials": case_materials,
        }

        if case_name == "upper_bound":
            model.springs.extend(case_springs)
            model.materials.extend(case_materials)

    # Rigid links from top_node to each spring node
    for s_node in spring_nodes:
        elem_tag = tag_alloc.next()
        model.elements.append({
            "tag": elem_tag,
            "type": "rigidLink",
            "nodes": [top_tag, s_node],
            "linkType": "beam",
        })

    # Capacity
    bearing_ksf = params.get("allowable_bearing_ksf", 4.0)
    area_ft2 = L_ft * W_ft
    model.capacity = {
        "axial_kips": bearing_ksf * area_ft2,  # ksf × ft² = kip
        "vertical_stiffness_kip_in": k_spring * n_L * n_W,
        "rotational_stiffness_kip_in_rad": (ks_kci * L_in * W_in ** 3 / 12.0),
        "note": "Winkler model — compression only springs",
    }

    return model


# ===================================================================
# DRIVEN PILE GROUP BUILDER
# ===================================================================

# AASHTO 10.7.2.4 — p-multipliers for group effects
P_MULTIPLIERS_AASHTO: dict[str, dict[float, list[float]]] = {
    # spacing/diameter → [lead_row, 2nd_row, 3rd_row, trailing_rows]
    "3D": [0.80, 0.40, 0.30, 0.30],
    "4D": [0.85, 0.55, 0.45, 0.40],
    "5D": [0.90, 0.65, 0.55, 0.50],
    "6D": [0.95, 0.75, 0.65, 0.60],
    "8D": [1.00, 0.90, 0.80, 0.75],
}


def _get_p_multipliers(spacing_over_D: float, n_rows: int) -> list[float]:
    """Get p-multipliers for pile group per AASHTO 10.7.2.4.

    Interpolates between tabulated values.

    Args:
        spacing_over_D: Center-to-center spacing / pile diameter.
        n_rows: Number of rows in loading direction.

    Returns:
        List of p-multiplier per row [lead, 2nd, 3rd, ... trailing].
    """
    # Breakpoints
    breakpoints = [3.0, 4.0, 5.0, 6.0, 8.0]
    values_table = [
        [0.80, 0.40, 0.30, 0.30],
        [0.85, 0.55, 0.45, 0.40],
        [0.90, 0.65, 0.55, 0.50],
        [0.95, 0.75, 0.65, 0.60],
        [1.00, 0.90, 0.80, 0.75],
    ]

    s = max(3.0, min(8.0, spacing_over_D))

    # Find bracketing indices
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= s <= breakpoints[i + 1]:
            frac = (s - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            interp = [
                values_table[i][j] + frac * (values_table[i + 1][j] - values_table[i][j])
                for j in range(4)
            ]
            break
    else:
        interp = values_table[-1]

    # Build per-row list
    result = []
    for row_idx in range(n_rows):
        if row_idx == 0:
            result.append(interp[0])
        elif row_idx == 1:
            result.append(interp[1])
        elif row_idx == 2:
            result.append(interp[2])
        else:
            result.append(interp[3])

    return result


# Pile section properties (simplified)
PILE_SECTIONS: dict[str, dict] = {
    "HP10x42": {"area": 12.4, "Ix": 210, "depth": 9.70, "width": 10.075},
    "HP12x53": {"area": 15.5, "Ix": 393, "depth": 11.78, "width": 12.045},
    "HP14x73": {"area": 21.4, "Ix": 729, "depth": 13.61, "width": 14.585},
    "HP14x89": {"area": 26.1, "Ix": 904, "depth": 13.83, "width": 14.695},
    "HP14x117": {"area": 34.4, "Ix": 1220, "depth": 14.21, "width": 14.885},
    "PIPE12": {"area": 14.6, "Ix": 279, "depth": 12.75, "width": 12.75},
    "PIPE16": {"area": 19.4, "Ix": 672, "depth": 16.0, "width": 16.0},
    "PIPE24": {"area": 29.5, "Ix": 2350, "depth": 24.0, "width": 24.0},
    "CONC12": {"area": 144.0, "Ix": 1728, "depth": 12.0, "width": 12.0},
    "CONC16": {"area": 256.0, "Ix": 5461, "depth": 16.0, "width": 16.0},
    "CONC18": {"area": 324.0, "Ix": 8748, "depth": 18.0, "width": 18.0},
}


def build_driven_pile_group(
    params: dict,
    site: SiteProfile,
    tag_alloc: TagAllocator | None = None,
    spring_spacing_ft: float = 1.0,
) -> FoundationModel:
    """Build driven pile group foundation model.

    Individual piles modeled with p-y/t-z/Q-z springs (like shafts).
    Pile cap modeled as rigid body with rigidLink constraints.
    Group effects via AASHTO p-multipliers.

    Args:
        params: {
            pile_type: str,            # "HP14x73", "PIPE16", "CONC18", etc.
            n_rows: int,               # Rows in loading direction
            n_cols: int,               # Columns perpendicular
            spacing_ft: float,         # Center-to-center spacing (ft)
            length_ft: float,          # Embedded pile length (ft)
            cap_length_ft: float,      # Pile cap L (ft)
            cap_width_ft: float,       # Pile cap W (ft)
            cap_thickness_ft: float,   # Pile cap thickness (ft)
        }
        site: SiteProfile.
        tag_alloc: Tag allocator.
        spring_spacing_ft: Spring spacing (ft).

    Returns:
        FoundationModel with grouped piles and cap.
    """
    if tag_alloc is None:
        tag_alloc = TagAllocator(start=3000)

    pile_type = params.get("pile_type", "HP14x73")
    n_rows = params.get("n_rows", 3)
    n_cols = params.get("n_cols", 3)
    spacing_ft = params.get("spacing_ft", 5.0)
    length_ft = params.get("length_ft", 50.0)
    cap_thick_ft = params.get("cap_thickness_ft", 4.0)

    pile_props = PILE_SECTIONS.get(pile_type, PILE_SECTIONS["HP14x73"])
    pile_diameter_in = pile_props["depth"]
    spacing_in = spacing_ft * FT_TO_IN
    spacing_over_D = spacing_in / pile_diameter_in

    # p-multipliers
    p_mults = _get_p_multipliers(spacing_over_D, n_rows)

    model = FoundationModel()

    # Cap center node (top of model)
    cap_node = tag_alloc.next()
    cap_y = cap_thick_ft * FT_TO_IN  # top of cap above ground
    model.nodes.append({"tag": cap_node, "x": 0.0, "y": cap_y, "z": 0.0})
    model.top_node = cap_node

    # Build each pile
    pile_top_nodes = []
    all_pile_springs = {"upper_bound": [], "lower_bound": []}
    all_pile_materials = {"upper_bound": [], "lower_bound": []}

    for row in range(n_rows):
        p_mult_row = p_mults[row] if row < len(p_mults) else p_mults[-1]

        for col in range(n_cols):
            # Pile head position
            x_offset = (row - (n_rows - 1) / 2.0) * spacing_in
            z_offset = (col - (n_cols - 1) / 2.0) * spacing_in

            # Build single pile as a mini drilled shaft
            pile_params = {
                "diameter_ft": pile_diameter_in / FT_TO_IN,
                "length_ft": length_ft,
                "above_grade_ft": 0.0,
                "fc_ksi": 4.0,
                "fy_ksi": 60.0,
                "n_bars": 0,
                "bar_size": "#8",
                "cover_in": 2.0,
            }

            pile_model = build_drilled_shaft(
                pile_params, site, tag_alloc, spring_spacing_ft,
            )

            # Offset all nodes
            for node in pile_model.nodes:
                node["x"] += x_offset
                node["z"] += z_offset

            model.nodes.extend(pile_model.nodes)
            model.elements.extend(pile_model.elements)
            pile_top_nodes.append(pile_model.top_node)

            # Apply p-multiplier to lateral springs (group effect)
            for case_name in ("upper_bound", "lower_bound"):
                case = pile_model.cases[case_name]
                for spring in case["springs"]:
                    if spring["direction"] == "lateral":
                        # Scale by group p-multiplier
                        spring["p_multiplier"] = p_mult_row
                        if "curve_data" in spring and "pu" in spring["curve_data"]:
                            spring["curve_data"]["pu"] *= p_mult_row
                            spring["curve_data"]["points"] = [
                                (y, p * p_mult_row)
                                for y, p in spring["curve_data"]["points"]
                            ]
                    all_pile_springs[case_name].append(spring)
                all_pile_materials[case_name].extend(case["materials"])

    # Rigid links from cap to pile heads
    for pile_top in pile_top_nodes:
        link_tag = tag_alloc.next()
        model.elements.append({
            "tag": link_tag,
            "type": "rigidLink",
            "nodes": [cap_node, pile_top],
            "linkType": "beam",
        })

    # Store cases
    for case_name in ("upper_bound", "lower_bound"):
        mult = 2.0 if case_name == "upper_bound" else 0.5
        model.cases[case_name] = {
            "multiplier": mult,
            "springs": all_pile_springs[case_name],
            "materials": all_pile_materials[case_name],
            "p_multipliers": p_mults,
        }

    model.springs = all_pile_springs["upper_bound"]
    model.materials.extend(all_pile_materials["upper_bound"])
    model.base_node = model.nodes[-1]["tag"]

    n_piles = n_rows * n_cols
    model.capacity = {
        "n_piles": n_piles,
        "group_axial_kips": n_piles * 200.0,  # Placeholder — need driving analysis
        "p_multipliers": p_mults,
        "note": "Axial capacity requires wave equation / PDA verification",
    }

    return model


# ===================================================================
# PILE BENT BUILDER
# ===================================================================

def build_pile_bent(
    params: dict,
    site: SiteProfile,
    tag_alloc: TagAllocator | None = None,
    spring_spacing_ft: float = 1.0,
) -> FoundationModel:
    """Build pile bent (trestle) foundation model.

    Piles extend through soil with p-y springs and above ground.
    Simple pile bent: no cap, piles connect directly to superstructure.
    Capped pile bent: cap beam ties pile heads.

    Args:
        params: {
            pile_type: str,            # "HP14x73", "PIPE16", etc.
            n_piles: int,              # Number of piles in bent
            spacing_ft: float,         # Center-to-center spacing (ft)
            embedded_length_ft: float, # Below ground (ft)
            exposed_length_ft: float,  # Above ground to cap/deck (ft)
            batter_deg: float,         # Batter angle (degrees), 0 = vertical
            has_cap: bool,             # Whether to include cap beam
        }
        site: SiteProfile.
        tag_alloc: Tag allocator.
        spring_spacing_ft: Spring spacing (ft).

    Returns:
        FoundationModel.
    """
    if tag_alloc is None:
        tag_alloc = TagAllocator(start=5000)

    pile_type = params.get("pile_type", "HP14x73")
    n_piles = params.get("n_piles", 4)
    spacing_ft = params.get("spacing_ft", 6.0)
    embedded_ft = params.get("embedded_length_ft", 40.0)
    exposed_ft = params.get("exposed_length_ft", 15.0)
    has_cap = params.get("has_cap", True)

    model = FoundationModel()

    # Build each pile with exposed length above ground
    pile_models = []
    pile_top_nodes = []

    for i in range(n_piles):
        pile_props = PILE_SECTIONS.get(pile_type, PILE_SECTIONS["HP14x73"])
        pile_diameter_in = pile_props["depth"]
        x_offset = (i - (n_piles - 1) / 2.0) * spacing_ft * FT_TO_IN

        pile_params = {
            "diameter_ft": pile_diameter_in / FT_TO_IN,
            "length_ft": embedded_ft,
            "above_grade_ft": exposed_ft,
            "fc_ksi": 4.0,
            "fy_ksi": 50.0,  # HP piles typically Fy=50
            "n_bars": 0,
            "bar_size": "#8",
            "cover_in": 2.0,
        }

        pile_model = build_drilled_shaft(
            pile_params, site, tag_alloc, spring_spacing_ft,
        )

        # Offset x
        for node in pile_model.nodes:
            node["x"] += x_offset

        model.nodes.extend(pile_model.nodes)
        model.elements.extend(pile_model.elements)
        pile_top_nodes.append(pile_model.top_node)
        pile_models.append(pile_model)

        for case_name in ("upper_bound", "lower_bound"):
            if case_name not in model.cases:
                model.cases[case_name] = {"springs": [], "materials": []}
            model.cases[case_name]["springs"].extend(
                pile_model.cases[case_name]["springs"]
            )
            model.cases[case_name]["materials"].extend(
                pile_model.cases[case_name]["materials"]
            )

    # Cap beam (if applicable)
    if has_cap and len(pile_top_nodes) > 1:
        cap_node = tag_alloc.next()
        cap_y = exposed_ft * FT_TO_IN
        model.nodes.append({"tag": cap_node, "x": 0.0, "y": cap_y, "z": 0.0})
        model.top_node = cap_node

        for pt_node in pile_top_nodes:
            link_tag = tag_alloc.next()
            model.elements.append({
                "tag": link_tag,
                "type": "rigidLink",
                "nodes": [cap_node, pt_node],
                "linkType": "beam",
            })
    else:
        model.top_node = pile_top_nodes[0] if pile_top_nodes else 0

    model.base_node = model.nodes[-1]["tag"] if model.nodes else 0

    # Collect springs from upper bound
    for pm in pile_models:
        model.springs.extend(pm.cases.get("upper_bound", {}).get("springs", []))
        model.materials.extend(pm.cases.get("upper_bound", {}).get("materials", []))

    model.capacity = {
        "n_piles": n_piles,
        "exposed_length_ft": exposed_ft,
        "note": "Pile bent — capacity governed by individual pile + buckling",
    }

    return model


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def create_foundation(
    foundation_type: str,
    params: dict,
    site_profile: dict | SiteProfile,
) -> FoundationModel:
    """Create complete foundation model with OpenSees commands.

    This is the main entry point for the foundation tool. It makes
    engineering decisions internally based on foundation type and site data.

    Args:
        foundation_type: One of "drilled_shaft", "driven_pile_group",
            "spread_footing", "pile_bent".
        params: Foundation-specific parameters (see individual builders).
        site_profile: Either a SiteProfile object or a dict with:
            {
                "layers": [
                    {"soil_type": "soft_clay", "top_depth_ft": 0,
                     "thickness_ft": 20, "su_ksf": 1.0, "gamma_pcf": 110},
                    ...
                ],
                "gwt_depth_ft": 10,
                "scour": {"water_crossing": true, "depth_ft": 6}
            }

    Returns:
        FoundationModel with nodes, elements, springs, materials,
        and upper/lower bound cases.

    Raises:
        ValueError: If foundation_type is not supported.

    References:
        AASHTO LRFD Bridge Design Specifications, 10th Edition
        FHWA-NHI-10-016: Drilled Shafts (2010)
        API RP 2GEO: Geotechnical and Foundation Design Considerations
    """
    # Convert dict to SiteProfile if needed
    if isinstance(site_profile, dict):
        layers = [
            SoilLayer(**layer_data) for layer_data in site_profile.get("layers", [])
        ]
        site = SiteProfile(
            layers=layers,
            gwt_depth_ft=site_profile.get("gwt_depth_ft", 10.0),
            scour=site_profile.get("scour"),
        )
    else:
        site = site_profile

    # Validate
    ftype = foundation_type.lower().replace(" ", "_")
    spring_spacing_ft = params.pop("spring_spacing_ft", 1.0) if "spring_spacing_ft" in params else 1.0

    if ftype in ("drilled_shaft", "drilled shaft", "shaft"):
        return build_drilled_shaft(params, site, spring_spacing_ft=spring_spacing_ft)
    elif ftype in ("driven_pile_group", "pile_group", "pile group", "driven pile group"):
        return build_driven_pile_group(params, site, spring_spacing_ft=spring_spacing_ft)
    elif ftype in ("spread_footing", "spread footing", "footing"):
        return build_spread_footing(params, site)
    elif ftype in ("pile_bent", "pile bent", "trestle"):
        return build_pile_bent(params, site, spring_spacing_ft=spring_spacing_ft)
    else:
        raise ValueError(
            f"Unknown foundation type: '{foundation_type}'. "
            f"Supported: drilled_shaft, driven_pile_group, spread_footing, pile_bent"
        )

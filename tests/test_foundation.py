"""Tests for foundation modeling tool.

Validates soil spring computations, model assembly, scour effects,
upper/lower bound cases, and pile group effects.
"""

import math
import pytest

from nlb.tools.foundation import (
    PYCurve,
    TZCurve,
    QZCurve,
    SoilLayer,
    SiteProfile,
    FoundationModel,
    TagAllocator,
    build_drilled_shaft,
    build_spread_footing,
    build_driven_pile_group,
    build_pile_bent,
    create_foundation,
    _spring_depths,
    _get_p_multipliers,
    FT_TO_IN,
    KSF_TO_KSI,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def soft_clay_site():
    """Simple soft clay profile — typical Gulf Coast."""
    return SiteProfile(
        layers=[
            SoilLayer(
                soil_type="soft_clay",
                top_depth_ft=0.0,
                thickness_ft=40.0,
                su_ksf=0.75,
                gamma_pcf=110.0,
            ),
            SoilLayer(
                soil_type="stiff_clay",
                top_depth_ft=40.0,
                thickness_ft=20.0,
                su_ksf=3.0,
                gamma_pcf=125.0,
            ),
        ],
        gwt_depth_ft=5.0,
    )


@pytest.fixture
def sand_site():
    """Medium dense sand profile."""
    return SiteProfile(
        layers=[
            SoilLayer(
                soil_type="sand",
                top_depth_ft=0.0,
                thickness_ft=30.0,
                phi_deg=32.0,
                gamma_pcf=120.0,
                N_spt=20,
            ),
            SoilLayer(
                soil_type="sand",
                top_depth_ft=30.0,
                thickness_ft=30.0,
                phi_deg=36.0,
                gamma_pcf=125.0,
                N_spt=40,
            ),
        ],
        gwt_depth_ft=10.0,
    )


@pytest.fixture
def rock_site():
    """Clay over rock profile."""
    return SiteProfile(
        layers=[
            SoilLayer(
                soil_type="stiff_clay",
                top_depth_ft=0.0,
                thickness_ft=15.0,
                su_ksf=2.0,
                gamma_pcf=120.0,
            ),
            SoilLayer(
                soil_type="rock",
                top_depth_ft=15.0,
                thickness_ft=50.0,
                qu_ksf=50.0,
                gamma_pcf=150.0,
            ),
        ],
        gwt_depth_ft=8.0,
    )


@pytest.fixture
def scour_site():
    """Soft clay with scour condition (water crossing)."""
    return SiteProfile(
        layers=[
            SoilLayer(
                soil_type="soft_clay",
                top_depth_ft=0.0,
                thickness_ft=50.0,
                su_ksf=1.0,
                gamma_pcf=110.0,
            ),
        ],
        gwt_depth_ft=0.0,
        scour={"water_crossing": True, "depth_ft": 10.0},
    )


@pytest.fixture
def shaft_params():
    """Typical 5-ft diameter drilled shaft."""
    return {
        "diameter_ft": 5.0,
        "length_ft": 60.0,
        "fc_ksi": 4.0,
        "fy_ksi": 60.0,
        "n_bars": 20,
        "bar_size": "#10",
        "cover_in": 3.0,
    }


# ===========================================================================
# p-y curve tests
# ===========================================================================

class TestPYCurves:
    """Test lateral soil resistance (p-y) curve generation."""

    def test_soft_clay_matlock_pu_shallow(self):
        """Shallow soft clay pu should be between 3*su*D and 9*su*D."""
        result = PYCurve.soft_clay_matlock(
            depth_in=12.0,    # 1 ft deep
            su_ksi=0.75 * KSF_TO_KSI,
            eps50=0.02,
            diameter_in=60.0,  # 5 ft
            gamma_pci=110.0 / 1728.0,
        )
        su = 0.75 * KSF_TO_KSI
        D = 60.0
        assert result["pu"] >= 3.0 * su * D * 0.9  # Allow 10% tolerance
        assert result["pu"] <= 9.0 * su * D * 1.1

    def test_soft_clay_matlock_pu_deep(self):
        """Deep soft clay pu should approach 9*su*D."""
        su_ksi = 0.75 * KSF_TO_KSI
        D = 60.0
        result = PYCurve.soft_clay_matlock(
            depth_in=600.0,    # 50 ft deep — well past transition
            su_ksi=su_ksi,
            eps50=0.02,
            diameter_in=D,
            gamma_pci=110.0 / 1728.0,
        )
        expected_deep = 9.0 * su_ksi * D
        assert abs(result["pu"] - expected_deep) / expected_deep < 0.01

    def test_soft_clay_y50(self):
        """y50 = 2.5 * eps50 * D for Matlock."""
        eps50 = 0.02
        D = 60.0
        result = PYCurve.soft_clay_matlock(
            depth_in=120.0,
            su_ksi=0.75 * KSF_TO_KSI,
            eps50=eps50,
            diameter_in=D,
            gamma_pci=110.0 / 1728.0,
        )
        expected_y50 = 2.5 * eps50 * D
        assert abs(result["y50"] - expected_y50) < 0.001

    def test_soft_clay_curve_shape(self):
        """Curve should be monotonically increasing up to pu."""
        result = PYCurve.soft_clay_matlock(
            depth_in=120.0,
            su_ksi=0.75 * KSF_TO_KSI,
            eps50=0.02,
            diameter_in=60.0,
            gamma_pci=110.0 / 1728.0,
        )
        points = result["points"]
        for i in range(1, len(points)):
            assert points[i][1] >= points[i - 1][1], \
                f"p-y curve not monotonic at point {i}"

    def test_stiff_clay_reese_pu(self):
        """Stiff clay pu should be reasonable for given su."""
        su_ksi = 3.0 * KSF_TO_KSI
        D = 48.0  # 4 ft
        result = PYCurve.stiff_clay_reese(
            depth_in=120.0,  # 10 ft
            su_ksi=su_ksi,
            eps50=0.005,
            diameter_in=D,
            gamma_pci=125.0 / 1728.0,
        )
        # Deep flow-around: 11*su*D
        pu_max = 11.0 * su_ksi * D
        assert result["pu"] > 0
        assert result["pu"] <= pu_max * 1.01

    def test_stiff_clay_stiffer_than_soft(self):
        """Stiff clay should have higher pu than soft clay at same depth."""
        D = 60.0
        depth = 120.0
        gamma = 120.0 / 1728.0

        soft = PYCurve.soft_clay_matlock(
            depth_in=depth, su_ksi=0.5 * KSF_TO_KSI, eps50=0.02,
            diameter_in=D, gamma_pci=gamma,
        )
        stiff = PYCurve.stiff_clay_reese(
            depth_in=depth, su_ksi=3.0 * KSF_TO_KSI, eps50=0.005,
            diameter_in=D, gamma_pci=gamma,
        )
        assert stiff["pu"] > soft["pu"]

    def test_sand_api_pu_increases_with_depth(self):
        """Sand p-y: pu should increase with depth (more overburden)."""
        D = 60.0
        phi = 32.0
        gamma = 120.0 / 1728.0
        k = 55.0

        shallow = PYCurve.sand_api(
            depth_in=24.0, phi_deg=phi, diameter_in=D,
            gamma_pci=gamma, k_py_pci=k,
        )
        deep = PYCurve.sand_api(
            depth_in=240.0, phi_deg=phi, diameter_in=D,
            gamma_pci=gamma, k_py_pci=k,
        )
        assert deep["pu"] > shallow["pu"]

    def test_sand_api_curve_has_points(self):
        """Sand p-y curve should have reasonable number of points."""
        result = PYCurve.sand_api(
            depth_in=120.0, phi_deg=35.0, diameter_in=60.0,
            gamma_pci=120.0 / 1728.0, k_py_pci=55.0,
        )
        assert len(result["points"]) >= 8

    def test_rock_py_pu(self):
        """Rock pu = qu * D."""
        qu_ksi = 50.0 * KSF_TO_KSI
        D = 60.0
        result = PYCurve.rock(depth_in=200.0, qu_ksi=qu_ksi, diameter_in=D)
        expected = qu_ksi * D
        assert abs(result["pu"] - expected) < 0.001

    def test_rock_stiffer_than_soil(self):
        """Rock should be stiffer (higher pu) than clay."""
        D = 60.0
        clay = PYCurve.soft_clay_matlock(
            depth_in=200.0, su_ksi=1.0 * KSF_TO_KSI, eps50=0.01,
            diameter_in=D, gamma_pci=120.0 / 1728.0,
        )
        rock = PYCurve.rock(depth_in=200.0, qu_ksi=50.0 * KSF_TO_KSI, diameter_in=D)
        assert rock["pu"] > clay["pu"]


# ===========================================================================
# t-z curve tests
# ===========================================================================

class TestTZCurves:
    """Test skin friction (t-z) curve generation."""

    def test_clay_alpha_tult_positive(self):
        """Clay t-z tult should be positive."""
        result = TZCurve.clay_alpha(
            su_ksi=1.0 * KSF_TO_KSI,
            diameter_in=60.0,
            spacing_in=12.0,
        )
        assert result["tult"] > 0

    def test_clay_alpha_range(self):
        """Alpha should be between 0.25 and 1.0."""
        result = TZCurve.clay_alpha(
            su_ksi=1.0 * KSF_TO_KSI,
            diameter_in=60.0,
            spacing_in=12.0,
        )
        assert 0.25 <= result["alpha"] <= 1.0

    def test_clay_alpha_decreases_with_su(self):
        """Higher su should give lower alpha (FHWA correlation)."""
        low_su = TZCurve.clay_alpha(su_ksi=0.5 * KSF_TO_KSI, diameter_in=60.0, spacing_in=12.0)
        high_su = TZCurve.clay_alpha(su_ksi=4.0 * KSF_TO_KSI, diameter_in=60.0, spacing_in=12.0)
        assert high_su["alpha"] <= low_su["alpha"]

    def test_sand_beta_tult_positive(self):
        """Sand t-z tult should be positive."""
        result = TZCurve.sand_beta(
            sigma_v_ksi=0.05,
            phi_deg=32.0,
            diameter_in=60.0,
            spacing_in=12.0,
            depth_in=120.0,
        )
        assert result["tult"] > 0

    def test_sand_beta_increases_with_sigma_v(self):
        """Higher overburden → higher skin friction."""
        shallow = TZCurve.sand_beta(
            sigma_v_ksi=0.01, phi_deg=32.0, diameter_in=60.0,
            spacing_in=12.0, depth_in=24.0,
        )
        deep = TZCurve.sand_beta(
            sigma_v_ksi=0.10, phi_deg=32.0, diameter_in=60.0,
            spacing_in=12.0, depth_in=240.0,
        )
        assert deep["tult"] > shallow["tult"]


# ===========================================================================
# Q-z curve tests
# ===========================================================================

class TestQZCurves:
    """Test end bearing (Q-z) curve generation."""

    def test_clay_qz_Nc(self):
        """Clay Nc should be ~9."""
        result = QZCurve.clay(su_ksi=1.0 * KSF_TO_KSI, diameter_in=60.0)
        assert result["Nc"] == 9.0

    def test_clay_qz_Qult(self):
        """Qult = Nc * su * Area."""
        su = 1.0 * KSF_TO_KSI
        D = 60.0
        result = QZCurve.clay(su_ksi=su, diameter_in=D)
        area = math.pi * D ** 2 / 4.0
        expected = 9.0 * su * area
        assert abs(result["Qult"] - expected) < 0.01

    def test_sand_qz_Qult_positive(self):
        """Sand end bearing should be positive."""
        result = QZCurve.sand(sigma_v_ksi=0.10, phi_deg=35.0, diameter_in=60.0)
        assert result["Qult"] > 0

    def test_sand_qz_limited(self):
        """Sand Qult limited to ~100 ksf (0.694 ksi)."""
        # Very high sigma_v should hit the cap
        result = QZCurve.sand(sigma_v_ksi=1.0, phi_deg=40.0, diameter_in=60.0)
        area = math.pi * 60.0 ** 2 / 4.0
        max_qult = 0.694 * area
        assert result["Qult"] <= max_qult * 1.01

    def test_rock_qz_formula(self):
        """Rock Qult = 2.5 * qu * Area."""
        qu = 50.0 * KSF_TO_KSI
        D = 48.0
        result = QZCurve.rock(qu_ksi=qu, diameter_in=D)
        area = math.pi * D ** 2 / 4.0
        expected = 2.5 * qu * area
        assert abs(result["Qult"] - expected) < 0.01

    def test_rock_stiffer_qz(self):
        """Rock z_peak should be smaller (stiffer) than clay."""
        D = 60.0
        clay = QZCurve.clay(su_ksi=1.0 * KSF_TO_KSI, diameter_in=D)
        rock = QZCurve.rock(qu_ksi=50.0 * KSF_TO_KSI, diameter_in=D)
        assert rock["z_peak"] < clay["z_peak"]


# ===========================================================================
# Drilled shaft model tests
# ===========================================================================

class TestDrilledShaft:
    """Test drilled shaft model assembly."""

    def test_basic_shaft_creates_model(self, shaft_params, soft_clay_site):
        """Should create a valid FoundationModel."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        assert isinstance(model, FoundationModel)
        assert len(model.nodes) > 0
        assert len(model.elements) > 0
        assert model.top_node > 0
        assert model.base_node > 0

    def test_shaft_node_count(self, shaft_params, soft_clay_site):
        """Node count should match: ground surface + spring depths + tip."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        length_ft = shaft_params["length_ft"]
        # ~1 node per ft of embedment + ground surface + tip
        expected_min = length_ft  # At least one per spring depth
        assert len(model.nodes) >= expected_min * 0.8

    def test_shaft_element_count(self, shaft_params, soft_clay_site):
        """Elements = nodes - 1 (beam column elements between nodes)."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        beam_elements = [e for e in model.elements if e["type"] == "dispBeamColumn"]
        assert len(beam_elements) == len(model.nodes) - 1

    def test_shaft_has_both_bounds(self, shaft_params, soft_clay_site):
        """Model should have upper_bound and lower_bound cases."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        assert "upper_bound" in model.cases
        assert "lower_bound" in model.cases

    def test_upper_bound_multiplier(self, shaft_params, soft_clay_site):
        """Upper bound springs should have 2x multiplier."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        ub = model.cases["upper_bound"]
        assert ub["multiplier"] == 2.0

    def test_lower_bound_multiplier(self, shaft_params, soft_clay_site):
        """Lower bound springs should have 0.5x multiplier."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        lb = model.cases["lower_bound"]
        assert lb["multiplier"] == 0.5

    def test_upper_vs_lower_spring_stiffness(self, shaft_params, soft_clay_site):
        """Upper bound pu should be 4x lower bound pu (2.0/0.5)."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        ub_lateral = [
            s for s in model.cases["upper_bound"]["springs"]
            if s["direction"] == "lateral"
        ]
        lb_lateral = [
            s for s in model.cases["lower_bound"]["springs"]
            if s["direction"] == "lateral"
        ]
        # Same number of springs
        assert len(ub_lateral) == len(lb_lateral)

        # Upper should be 4x lower (2.0 / 0.5 = 4)
        if ub_lateral and lb_lateral:
            ub_pu = ub_lateral[0]["curve_data"]["pu"]
            lb_pu = lb_lateral[0]["curve_data"]["pu"]
            ratio = ub_pu / lb_pu if lb_pu > 0 else 0
            assert abs(ratio - 4.0) < 0.01

    def test_shaft_has_capacity(self, shaft_params, soft_clay_site):
        """Model should report capacity estimates."""
        model = build_drilled_shaft(shaft_params, soft_clay_site)
        assert "axial_kips" in model.capacity
        assert "lateral_kips" in model.capacity
        assert model.capacity["axial_kips"] > 0

    def test_shaft_in_sand(self, shaft_params, sand_site):
        """Shaft in sand should use API p-y formulation."""
        model = build_drilled_shaft(shaft_params, sand_site)
        assert len(model.springs) > 0
        # Check that sand formulation was used
        lateral_springs = [s for s in model.springs if s["direction"] == "lateral"]
        assert any("sand" in s["curve_data"].get("formulation", "") for s in lateral_springs)

    def test_shaft_in_rock(self, shaft_params, rock_site):
        """Shaft in rock profile should have springs in both clay and rock."""
        shaft_params["length_ft"] = 30.0  # Extends into rock
        model = build_drilled_shaft(shaft_params, rock_site)
        assert len(model.springs) > 0


# ===========================================================================
# Scour tests
# ===========================================================================

class TestScour:
    """Test scour effects on foundation models."""

    def test_scour_removes_shallow_springs(self, shaft_params, scour_site):
        """Springs above scour depth should be removed."""
        model = build_drilled_shaft(shaft_params, scour_site)
        scour_depth_ft = scour_site.scour["depth_ft"]

        lateral_springs = [
            s for s in model.springs if s["direction"] == "lateral"
        ]
        for spring in lateral_springs:
            assert spring["depth_ft"] >= scour_depth_ft, \
                f"Spring at {spring['depth_ft']} ft is above scour depth {scour_depth_ft} ft"

    def test_scour_fewer_springs(self, shaft_params, soft_clay_site, scour_site):
        """Scoured model should have fewer springs than non-scoured."""
        model_no_scour = build_drilled_shaft(shaft_params, soft_clay_site)
        model_scour = build_drilled_shaft(shaft_params, scour_site)

        springs_no_scour = len([s for s in model_no_scour.springs if s["direction"] == "lateral"])
        springs_scour = len([s for s in model_scour.springs if s["direction"] == "lateral"])

        assert springs_scour < springs_no_scour

    def test_scour_via_create_foundation(self, shaft_params):
        """Scour should work through the main create_foundation API."""
        site_dict = {
            "layers": [
                {"soil_type": "soft_clay", "top_depth_ft": 0, "thickness_ft": 50,
                 "su_ksf": 1.0, "gamma_pcf": 110},
            ],
            "gwt_depth_ft": 0,
            "scour": {"water_crossing": True, "depth_ft": 8.0},
        }
        model = create_foundation("drilled_shaft", shaft_params, site_dict)
        lateral = [s for s in model.springs if s["direction"] == "lateral"]
        for s in lateral:
            assert s["depth_ft"] >= 8.0

    def test_spring_depths_with_scour(self):
        """_spring_depths should skip depths above scour."""
        depths = _spring_depths(
            length_in=600.0,
            spacing_in=12.0,
            scour_depth_in=120.0,  # 10 ft scour
        )
        for d in depths:
            assert d >= 120.0


# ===========================================================================
# Spread footing tests
# ===========================================================================

class TestSpreadFooting:
    """Test spread footing model."""

    def test_footing_creates_model(self, soft_clay_site):
        """Should create a valid model."""
        params = {"length_ft": 12.0, "width_ft": 12.0, "depth_ft": 4.0}
        model = build_spread_footing(params, soft_clay_site)
        assert isinstance(model, FoundationModel)
        assert model.top_node > 0

    def test_footing_spring_count(self, soft_clay_site):
        """Spring count should be n_L × n_W."""
        params = {
            "length_ft": 10.0, "width_ft": 10.0, "depth_ft": 4.0,
            "n_springs_l": 4, "n_springs_w": 4,
        }
        model = build_spread_footing(params, soft_clay_site)
        assert len(model.springs) == 16  # 4×4

    def test_footing_has_rigid_links(self, soft_clay_site):
        """Should have rigid links from center to springs."""
        params = {"length_ft": 10.0, "width_ft": 10.0, "depth_ft": 4.0}
        model = build_spread_footing(params, soft_clay_site)
        links = [e for e in model.elements if e["type"] == "rigidLink"]
        assert len(links) > 0

    def test_footing_capacity(self, soft_clay_site):
        """Should report bearing capacity."""
        params = {
            "length_ft": 10.0, "width_ft": 10.0, "depth_ft": 4.0,
            "allowable_bearing_ksf": 4.0,
        }
        model = build_spread_footing(params, soft_clay_site)
        assert model.capacity["axial_kips"] == 4.0 * 10.0 * 10.0  # 400 kips


# ===========================================================================
# Pile group tests
# ===========================================================================

class TestPileGroup:
    """Test driven pile group model."""

    def test_pile_group_creates_model(self, sand_site):
        """Should create a valid pile group model."""
        params = {
            "pile_type": "HP14x73",
            "n_rows": 3,
            "n_cols": 3,
            "spacing_ft": 5.0,
            "length_ft": 50.0,
        }
        model = build_driven_pile_group(params, sand_site)
        assert isinstance(model, FoundationModel)
        assert model.top_node > 0

    def test_pile_group_has_cap_links(self, sand_site):
        """Should have rigid links from cap to each pile head."""
        params = {
            "pile_type": "HP14x73",
            "n_rows": 2,
            "n_cols": 3,
            "spacing_ft": 5.0,
            "length_ft": 40.0,
        }
        model = build_driven_pile_group(params, sand_site)
        links = [e for e in model.elements if e["type"] == "rigidLink"]
        assert len(links) == 6  # 2×3 = 6 piles

    def test_p_multipliers_3D_spacing(self):
        """At 3D spacing: lead=0.80, 2nd=0.40, 3rd=0.30."""
        mults = _get_p_multipliers(3.0, 3)
        assert abs(mults[0] - 0.80) < 0.01
        assert abs(mults[1] - 0.40) < 0.01
        assert abs(mults[2] - 0.30) < 0.01

    def test_p_multipliers_5D_spacing(self):
        """At 5D spacing: lead=0.90, 2nd=0.65, 3rd=0.55."""
        mults = _get_p_multipliers(5.0, 3)
        assert abs(mults[0] - 0.90) < 0.01
        assert abs(mults[1] - 0.65) < 0.01
        assert abs(mults[2] - 0.55) < 0.01

    def test_p_multipliers_8D_all_near_1(self):
        """At 8D spacing, multipliers should be close to 1.0."""
        mults = _get_p_multipliers(8.0, 4)
        assert all(m >= 0.75 for m in mults)

    def test_p_multipliers_interpolation(self):
        """Interpolation between 3D and 5D should give intermediate values."""
        mults_4D = _get_p_multipliers(4.0, 2)
        mults_3D = _get_p_multipliers(3.0, 2)
        mults_5D = _get_p_multipliers(5.0, 2)
        # 4D lead row should be between 3D and 5D
        assert mults_3D[0] < mults_4D[0] < mults_5D[0]

    def test_pile_group_applies_p_multipliers(self, sand_site):
        """Group effect should reduce lateral spring stiffness."""
        params = {
            "pile_type": "HP14x73",
            "n_rows": 3,
            "n_cols": 1,
            "spacing_ft": 3.5,  # ~3D spacing for HP14
            "length_ft": 40.0,
        }
        model = build_driven_pile_group(params, sand_site)
        assert "p_multipliers" in model.cases["upper_bound"]


# ===========================================================================
# Pile bent tests
# ===========================================================================

class TestPileBent:
    """Test pile bent (trestle) model."""

    def test_pile_bent_creates_model(self, sand_site):
        """Should create a valid pile bent model."""
        params = {
            "pile_type": "HP14x73",
            "n_piles": 4,
            "spacing_ft": 6.0,
            "embedded_length_ft": 40.0,
            "exposed_length_ft": 15.0,
            "has_cap": True,
        }
        model = build_pile_bent(params, sand_site)
        assert isinstance(model, FoundationModel)

    def test_pile_bent_has_exposed_nodes(self, sand_site):
        """Should have nodes above ground (positive y)."""
        params = {
            "pile_type": "HP14x73",
            "n_piles": 3,
            "spacing_ft": 6.0,
            "embedded_length_ft": 30.0,
            "exposed_length_ft": 20.0,
            "has_cap": True,
        }
        model = build_pile_bent(params, sand_site)
        above_ground = [n for n in model.nodes if n["y"] > 0]
        assert len(above_ground) > 0

    def test_pile_bent_cap_links(self, sand_site):
        """Capped bent should have rigid links to cap."""
        params = {
            "pile_type": "HP14x73",
            "n_piles": 4,
            "spacing_ft": 6.0,
            "embedded_length_ft": 30.0,
            "exposed_length_ft": 15.0,
            "has_cap": True,
        }
        model = build_pile_bent(params, sand_site)
        links = [e for e in model.elements if e["type"] == "rigidLink"]
        assert len(links) == 4  # One per pile


# ===========================================================================
# Main API tests
# ===========================================================================

class TestCreateFoundation:
    """Test the main create_foundation entry point."""

    def test_drilled_shaft_from_dict(self):
        """Should accept dict site profile and string foundation type."""
        site = {
            "layers": [
                {"soil_type": "soft_clay", "top_depth_ft": 0,
                 "thickness_ft": 60, "su_ksf": 1.0, "gamma_pcf": 110},
            ],
            "gwt_depth_ft": 5,
        }
        params = {"diameter_ft": 5.0, "length_ft": 50.0}
        model = create_foundation("drilled_shaft", params, site)
        assert isinstance(model, FoundationModel)
        assert len(model.nodes) > 0

    def test_spread_footing_from_dict(self):
        """Should create spread footing from dict inputs."""
        site = {
            "layers": [
                {"soil_type": "sand", "top_depth_ft": 0,
                 "thickness_ft": 30, "phi_deg": 30, "gamma_pcf": 120},
            ],
        }
        params = {"length_ft": 10, "width_ft": 8, "depth_ft": 4}
        model = create_foundation("spread_footing", params, site)
        assert isinstance(model, FoundationModel)

    def test_unknown_type_raises(self):
        """Unknown foundation type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown foundation type"):
            create_foundation("magic_foundation", {}, {"layers": []})

    def test_alternative_type_names(self):
        """Should accept common aliases."""
        site = {
            "layers": [
                {"soil_type": "sand", "top_depth_ft": 0,
                 "thickness_ft": 60, "phi_deg": 32, "gamma_pcf": 120},
            ],
        }
        # "shaft" alias
        model = create_foundation("shaft", {"diameter_ft": 4, "length_ft": 40}, site)
        assert isinstance(model, FoundationModel)

    def test_model_summary(self):
        """FoundationModel.summary() should return a string."""
        site = {
            "layers": [
                {"soil_type": "soft_clay", "top_depth_ft": 0,
                 "thickness_ft": 60, "su_ksf": 1.0, "gamma_pcf": 110},
            ],
        }
        model = create_foundation("drilled_shaft", {"diameter_ft": 5, "length_ft": 50}, site)
        summary = model.summary()
        assert "nodes" in summary
        assert "elements" in summary


# ===========================================================================
# SiteProfile tests
# ===========================================================================

class TestSiteProfile:
    """Test site profile utilities."""

    def test_effective_stress_increases_with_depth(self, soft_clay_site):
        """Effective vertical stress should increase with depth."""
        s5 = soft_clay_site.effective_vertical_stress(5.0)
        s20 = soft_clay_site.effective_vertical_stress(20.0)
        s40 = soft_clay_site.effective_vertical_stress(40.0)
        assert s5 < s20 < s40

    def test_layer_at_depth(self, soft_clay_site):
        """Should return correct layer for given depth."""
        layer = soft_clay_site.layer_at_depth(10.0)
        assert layer.soil_type == "soft_clay"
        layer = soft_clay_site.layer_at_depth(45.0)
        assert layer.soil_type == "stiff_clay"

    def test_gwt_reduces_effective_stress(self):
        """Below GWT, buoyancy should reduce effective stress."""
        site_dry = SiteProfile(
            layers=[SoilLayer("sand", 0, 30, gamma_pcf=120, phi_deg=30)],
            gwt_depth_ft=100.0,  # Very deep — effectively dry
        )
        site_wet = SiteProfile(
            layers=[SoilLayer("sand", 0, 30, gamma_pcf=120, phi_deg=30)],
            gwt_depth_ft=0.0,  # At surface
        )
        s_dry = site_dry.effective_vertical_stress(20.0)
        s_wet = site_wet.effective_vertical_stress(20.0)
        assert s_wet < s_dry


# ===========================================================================
# TagAllocator tests
# ===========================================================================

class TestTagAllocator:
    """Test tag allocation."""

    def test_sequential(self):
        alloc = TagAllocator(start=100)
        assert alloc.next() == 100
        assert alloc.next() == 101

    def test_batch(self):
        alloc = TagAllocator(start=1)
        tags = alloc.next(5)
        assert tags == [1, 2, 3, 4, 5]
        assert alloc.next() == 6

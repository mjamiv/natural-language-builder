"""Tests for substructure and bearing modeling tools.

Validates model assembly, engineering calculations, and output integrity
for all supported substructure and bearing types.
"""

import math
import pytest

from nlb.tools.substructure import (
    ColumnConfig,
    CapBeamConfig,
    WallPierConfig,
    PileBentConfig,
    IntegralAbutmentConfig,
    SeatAbutmentConfig,
    SubstructureModel,
    TagAllocator,
    build_single_column,
    build_multi_column_bent,
    build_wall_pier,
    build_pile_bent,
    build_integral_abutment,
    build_seat_abutment,
    create_substructure,
    plastic_hinge_length,
    _column_mesh_lengths,
    _mander_confinement,
    _backfill_spring_params,
    FT_TO_IN,
    REBAR_AREAS,
    REBAR_DIAMETERS,
)

from nlb.tools.bearings import (
    ElastomericConfig,
    PotBearingConfig,
    PTFEConfig,
    FPSingleConfig,
    FPTripleConfig,
    RockerRollerConfig,
    BearingModel,
    build_elastomeric,
    build_pot_fixed,
    build_pot_guided,
    build_ptfe_sliding,
    build_fp_single,
    build_fp_triple,
    build_integral,
    build_rocker_roller,
    create_bearing,
    layout_bearings,
    _elastomeric_stiffness,
    _ptfe_friction,
    PTFE_FRICTION,
    G_ACCEL,
    FT_TO_IN as B_FT_TO_IN,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def default_column():
    """Standard 48" diameter circular column, 20' tall."""
    return ColumnConfig(
        shape="circular",
        diameter_in=48.0,
        height_ft=20.0,
        fc_ksi=4.0,
        fy_ksi=60.0,
        num_bars=16,
        bar_size="#8",
        cover_in=2.0,
        rho_s=0.01,
    )


@pytest.fixture
def rect_column():
    """36x48 rectangular column, 25' tall."""
    return ColumnConfig(
        shape="rectangular",
        width_in=36.0,
        depth_in=48.0,
        height_ft=25.0,
        fc_ksi=5.0,
        fy_ksi=60.0,
        num_bars=20,
        bar_size="#9",
        cover_in=2.0,
        rho_s=0.008,
    )


@pytest.fixture
def default_cap():
    """Standard cap beam."""
    return CapBeamConfig(
        width_in=60.0,
        depth_in=48.0,
        fc_ksi=4.0,
        fy_ksi=60.0,
        num_bars_top=8,
        num_bars_bot=8,
        bar_size="#9",
        cover_in=2.0,
    )


@pytest.fixture
def default_wall():
    """Standard 20' wide, 4' thick wall pier, 25' tall."""
    return WallPierConfig(
        height_ft=25.0,
        width_in=240.0,
        thickness_in=48.0,
        fc_ksi=4.0,
        fy_ksi=60.0,
        num_bars_face=20,
        bar_size="#9",
        cover_in=2.0,
        rho_s=0.006,
    )


@pytest.fixture
def slender_wall():
    """Slender wall: h/t > 25."""
    return WallPierConfig(
        height_ft=60.0,   # 720 inches
        width_in=240.0,
        thickness_in=24.0,  # h/t = 720/24 = 30 > 25
        fc_ksi=4.0,
        fy_ksi=60.0,
    )


@pytest.fixture
def default_elastomeric():
    """Standard steel-reinforced elastomeric bearing."""
    return ElastomericConfig(
        length_in=14.0,
        width_in=9.0,
        total_rubber_thickness_in=2.5,
        shear_modulus_ksi=0.100,
        num_internal_layers=5,
        layer_thickness_in=0.50,
    )


# ===========================================================================
# SUBSTRUCTURE TESTS
# ===========================================================================

class TestTagAllocator:
    def test_sequential(self):
        t = TagAllocator(100)
        assert t.next() == 100
        assert t.next() == 101

    def test_batch(self):
        t = TagAllocator(1)
        tags = t.next(5)
        assert tags == [1, 2, 3, 4, 5]
        assert t.current == 6


class TestPlasticHingeLength:
    def test_basic(self):
        """Lp = 0.08L + 0.15*fy*db."""
        L = 240.0  # 20 ft
        fy = 60.0
        db = 1.0  # #8 bar
        Lp = plastic_hinge_length(L, fy, db)
        expected = 0.08 * 240.0 + 0.15 * 60.0 * 1.0  # 19.2 + 9.0 = 28.2
        assert abs(Lp - expected) < 0.01

    def test_larger_bars(self):
        """Larger bars → longer Lp."""
        Lp_small = plastic_hinge_length(240.0, 60.0, 0.625)  # #5
        Lp_large = plastic_hinge_length(240.0, 60.0, 1.41)   # #11
        assert Lp_large > Lp_small


class TestColumnMesh:
    def test_minimum_elements(self):
        """At least 6 elements."""
        lengths, is_ph = _column_mesh_lengths(240.0, 28.0)
        assert len(lengths) >= 6
        assert len(lengths) == len(is_ph)

    def test_ph_zone_elements(self):
        """At least 2 elements in each PH zone (4 total PH elements)."""
        lengths, is_ph = _column_mesh_lengths(240.0, 28.0)
        ph_count = sum(1 for p in is_ph if p)
        assert ph_count >= 4  # 2 top + 2 bottom

    def test_lengths_sum_to_height(self):
        """Element lengths should sum to total height."""
        H = 360.0
        lengths, _ = _column_mesh_lengths(H, 30.0)
        assert abs(sum(lengths) - H) < 0.01


class TestManderConfinement:
    def test_fcc_greater_than_fc(self):
        """Confined strength must exceed unconfined."""
        fcc, ecc, ecu = _mander_confinement(4.0, 60.0, 0.01, "circular")
        assert fcc > 4.0

    def test_ecu_greater_than_003(self):
        """Confined strain must exceed 0.003."""
        fcc, ecc, ecu = _mander_confinement(4.0, 60.0, 0.01, "circular")
        assert ecu > 0.003

    def test_rectangular_lower_ke(self):
        """Rectangular confinement less effective than circular."""
        fcc_circ, _, _ = _mander_confinement(4.0, 60.0, 0.01, "circular")
        fcc_rect, _, _ = _mander_confinement(4.0, 60.0, 0.01, "rectangular")
        assert fcc_circ > fcc_rect


class TestSingleColumn:
    def test_node_count(self, default_column):
        """Column should have base node + intermediate + top nodes."""
        m = build_single_column(default_column)
        # Minimum 6 elements → 7 nodes (base + 6)
        assert len(m.nodes) >= 7

    def test_plastic_hinge_elements(self, default_column):
        """Should identify PH elements at top and bottom."""
        m = build_single_column(default_column)
        assert len(m.plastic_hinge_elements) >= 4

    def test_pdelta_transform(self, default_column):
        """Must use PDelta transform."""
        m = build_single_column(default_column)
        transforms = [mat for mat in m.materials if mat.get("type") == "geomTransf"]
        assert len(transforms) >= 1
        assert all(t["transform"] == "PDelta" for t in transforms)

    def test_has_confined_concrete(self, default_column):
        """Must have confined concrete material."""
        m = build_single_column(default_column)
        confined = [mat for mat in m.materials if "confined" in mat.get("name", "")]
        assert len(confined) >= 1

    def test_cracked_stiffness_flag(self, default_column):
        """Should flag cracked stiffness."""
        m = build_single_column(default_column)
        assert m.cracked_stiffness.get("columns") == 0.5

    def test_top_and_base_nodes(self, default_column):
        """Must have exactly one top and one base node."""
        m = build_single_column(default_column)
        assert len(m.top_nodes) == 1
        assert len(m.base_nodes) == 1

    def test_top_node_at_height(self, default_column):
        """Top node should be at correct height."""
        m = build_single_column(default_column)
        top_tag = m.top_nodes[0]
        top_node = next(n for n in m.nodes if n["tag"] == top_tag)
        expected_y = default_column.height_ft * FT_TO_IN
        assert abs(top_node["y"] - expected_y) < 0.1

    def test_rectangular_column(self, rect_column):
        """Rectangular column should also work."""
        m = build_single_column(rect_column)
        assert len(m.nodes) >= 7
        sec = m.sections[0]
        assert sec["type"] == "rectangular_rc"

    def test_element_connectivity(self, default_column):
        """Elements should form a chain from base to top."""
        m = build_single_column(default_column)
        base = m.base_nodes[0]
        top = m.top_nodes[0]

        # Walk the chain
        node_map = {}
        for e in m.elements:
            n1, n2 = e["nodes"]
            node_map[n1] = n2

        current = base
        visited = {current}
        while current != top:
            assert current in node_map, f"Broken chain at node {current}"
            current = node_map[current]
            assert current not in visited, "Cycle detected"
            visited.add(current)


class TestMultiColumnBent:
    def test_column_spacing(self, default_column, default_cap):
        """Columns should be at correct spacing."""
        m = build_multi_column_bent(3, 12.0, default_column, default_cap)
        base_nodes = [n for n in m.nodes if n["tag"] in m.base_nodes]
        z_values = sorted(n["z"] for n in base_nodes)
        assert len(z_values) == 3
        # Spacing = 12 ft = 144 in
        assert abs(z_values[1] - z_values[0] - 144.0) < 0.1
        assert abs(z_values[2] - z_values[1] - 144.0) < 0.1

    def test_cap_beam_connectivity(self, default_column, default_cap):
        """Cap beam should connect column tops."""
        m = build_multi_column_bent(3, 12.0, default_column, default_cap)
        # Should have cap beam elements
        cap_elements = [e for e in m.elements if e["section"] != m.sections[0]["tag"]]
        assert len(cap_elements) > 0

    def test_cap_nodes_populated(self, default_column, default_cap):
        """Cap nodes list should have entries."""
        m = build_multi_column_bent(3, 12.0, default_column, default_cap)
        assert len(m.cap_nodes) > 0

    def test_rigid_offset_when_cap_deep(self, default_column):
        """Rigid offset when cap depth > column diameter."""
        deep_cap = CapBeamConfig(depth_in=60.0, width_in=60.0)  # > 48" diameter
        m = build_multi_column_bent(2, 12.0, default_column, deep_cap)
        assert len(m.constraints) > 0
        assert m.constraints[0]["type"] == "rigidLink"

    def test_no_rigid_offset_when_cap_shallow(self, default_column):
        """No rigid offset when cap depth <= column diameter."""
        shallow_cap = CapBeamConfig(depth_in=36.0, width_in=60.0)  # < 48"
        m = build_multi_column_bent(2, 12.0, default_column, shallow_cap)
        assert len(m.constraints) == 0

    def test_cracked_stiffness(self, default_column, default_cap):
        """Both column and cap cracked stiffness flags."""
        m = build_multi_column_bent(3, 12.0, default_column, default_cap)
        assert m.cracked_stiffness.get("columns") == 0.5
        assert m.cracked_stiffness.get("cap_beam") == 0.35


class TestWallPier:
    def test_non_slender(self, default_wall):
        """Standard wall (h/t < 25) should not be slender."""
        # h/t = 300/48 = 6.25 — not slender
        m = build_wall_pier(default_wall)
        assert len(m.warnings) == 0

    def test_slender_flag(self, slender_wall):
        """Slender wall should have warning."""
        m = build_wall_pier(slender_wall)
        assert len(m.warnings) > 0
        assert "SLENDER" in m.warnings[0]

    def test_pdelta_transform(self, default_wall):
        """Wall pier should use PDelta."""
        m = build_wall_pier(default_wall)
        transforms = [mat for mat in m.materials if mat.get("type") == "geomTransf"]
        assert all(t["transform"] == "PDelta" for t in transforms)

    def test_section_dimensions(self, default_wall):
        """Section should have wall dimensions."""
        m = build_wall_pier(default_wall)
        sec = m.sections[0]
        assert sec["width_in"] == 240.0
        assert sec["depth_in"] == 48.0


class TestPileBent:
    def test_pile_count(self):
        """Correct number of piles."""
        cfg = PileBentConfig(pile_count=5, spacing_ft=6.0, free_height_ft=10.0)
        m = build_pile_bent(cfg)
        assert len(m.base_nodes) == 5

    def test_cap_beam_present(self):
        """Cap beam elements should exist."""
        cfg = PileBentConfig(pile_count=4, spacing_ft=6.0, free_height_ft=10.0)
        m = build_pile_bent(cfg)
        assert len(m.cap_nodes) > 0

    def test_pile_height(self):
        """Pile top nodes should be at free height."""
        cfg = PileBentConfig(pile_count=3, spacing_ft=8.0, free_height_ft=15.0)
        m = build_pile_bent(cfg)
        expected_y = 15.0 * FT_TO_IN
        # Check that cap nodes are at the right height
        cap_node_tags = set(m.cap_nodes)
        cap_node_ys = [n["y"] for n in m.nodes if n["tag"] in cap_node_tags]
        for cy in cap_node_ys:
            assert abs(cy - expected_y) < 0.1


class TestIntegralAbutment:
    def test_backfill_springs(self):
        """Should include backfill springs."""
        cfg = IntegralAbutmentConfig(num_springs=5)
        m = build_integral_abutment(cfg)
        assert len(m.springs) == 5

    def test_springs_are_passive(self):
        """All springs should be backfill passive type."""
        cfg = IntegralAbutmentConfig(num_springs=3)
        m = build_integral_abutment(cfg)
        for sp in m.springs:
            assert sp["spring_type"] == "backfill_passive"

    def test_top_node(self):
        """Should have a top node for superstructure connection."""
        cfg = IntegralAbutmentConfig()
        m = build_integral_abutment(cfg)
        assert len(m.top_nodes) == 1

    def test_skew_warning(self):
        """High skew should produce warning."""
        cfg = IntegralAbutmentConfig(skew_deg=45.0)
        m = build_integral_abutment(cfg)
        assert len(m.warnings) > 0
        assert "SKEW" in m.warnings[0]

    def test_backfill_spring_params(self):
        """Spring params should have increasing Fult with depth."""
        from nlb.tools.substructure import PCF_TO_PCI
        params = _backfill_spring_params(
            height_in=72.0, width_in=36.0,
            gamma_pci=120.0 * PCF_TO_PCI, phi_deg=34.0,
            num_springs=3,
        )
        assert len(params) == 3
        # Deeper springs should have higher capacity
        assert params[2]["Fult_kip"] > params[0]["Fult_kip"]


class TestSeatAbutment:
    def test_bearing_locations(self):
        """Should have bearing seat nodes at specified locations."""
        cfg = SeatAbutmentConfig(bearing_locations_in=[-48.0, 0.0, 48.0])
        m = build_seat_abutment(cfg)
        assert len(m.top_nodes) == 3
        assert len(m.cap_nodes) == 3

    def test_no_backfill_springs(self):
        """Seat abutment should have no backfill springs."""
        cfg = SeatAbutmentConfig()
        m = build_seat_abutment(cfg)
        assert len(m.springs) == 0


class TestCreateSubstructure:
    def test_dispatch_single_column(self):
        col = ColumnConfig()
        m = create_substructure("single_column", column=col)
        assert m.substructure_type == "single_column"

    def test_dispatch_wall_pier(self):
        wall = WallPierConfig()
        m = create_substructure("wall_pier", wall=wall)
        assert m.substructure_type == "wall_pier"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_substructure("flying_buttress")


# ===========================================================================
# BEARING TESTS
# ===========================================================================

class TestElastomericStiffness:
    def test_Kh_formula(self, default_elastomeric):
        """Kh = G × A / h_rt."""
        props = _elastomeric_stiffness(default_elastomeric)
        G = default_elastomeric.shear_modulus_ksi
        A = default_elastomeric.length_in * default_elastomeric.width_in
        h = default_elastomeric.total_rubber_thickness_in
        expected_Kh = G * A / h
        assert abs(props["Kh_kip_per_in"] - expected_Kh) < 0.01

    def test_Kv_greater_than_Kh(self, default_elastomeric):
        """Kv should be much greater than Kh."""
        props = _elastomeric_stiffness(default_elastomeric)
        assert props["Kv_kip_per_in"] > props["Kh_kip_per_in"] * 10

    def test_shape_factor(self, default_elastomeric):
        """Shape factor S = A / (perimeter × t_layer)."""
        props = _elastomeric_stiffness(default_elastomeric)
        A = 14.0 * 9.0
        perim = 2 * (14.0 + 9.0)
        t = 0.50
        expected_S = A / (perim * t)
        assert abs(props["shape_factor"] - expected_S) < 0.01

    def test_rotation_capacity(self, default_elastomeric):
        """Rotation capacity should be reasonable (> 0)."""
        props = _elastomeric_stiffness(default_elastomeric)
        assert props["rotation_capacity_rad"] > 0


class TestElastomericBearing:
    def test_nodes(self, default_elastomeric):
        bm = build_elastomeric(default_elastomeric)
        assert len(bm.nodes) == 2
        assert len(bm.top_nodes) == 1
        assert len(bm.bottom_nodes) == 1

    def test_compression_only(self, default_elastomeric):
        bm = build_elastomeric(default_elastomeric)
        assert bm.compression_only is True
        ent_mats = [m for m in bm.materials if m["type"] == "ENT"]
        assert len(ent_mats) >= 1

    def test_upper_lower_bound(self, default_elastomeric):
        bm = build_elastomeric(default_elastomeric)
        assert "upper_bound" in bm.cases
        assert "lower_bound" in bm.cases
        assert bm.cases["upper_bound"]["G_factor"] > 1.0
        assert bm.cases["lower_bound"]["G_factor"] < 1.0


class TestPotBearing:
    def test_fixed_high_stiffness(self):
        cfg = PotBearingConfig(vertical_capacity_kip=800.0)
        bm = build_pot_fixed(cfg)
        assert bm.properties["Kh_kip_per_in"] >= 1.0e6

    def test_guided_free_direction(self):
        cfg = PotBearingConfig(guide_direction=1)
        bm = build_pot_guided(cfg)
        assert bm.properties["guide_direction"] == 1
        assert bm.properties["Kh_free_kip_per_in"] < 1.0

    def test_compression_only(self):
        bm = build_pot_fixed(PotBearingConfig())
        assert bm.compression_only is True


class TestPTFEFriction:
    def test_reference_in_range(self):
        """Reference friction should be in typical range."""
        mu = _ptfe_friction("glass_filled", "reference")
        assert 0.03 <= mu["mu_slow"] <= 0.12
        assert 0.05 <= mu["mu_fast"] <= 0.15

    def test_cold_higher(self):
        """Cold temperature friction should be higher."""
        mu_ref = _ptfe_friction("glass_filled", "reference")
        mu_cold = _ptfe_friction("glass_filled", "cold")
        assert mu_cold["mu_slow"] > mu_ref["mu_slow"]
        assert mu_cold["mu_fast"] > mu_ref["mu_fast"]

    def test_hot_lower(self):
        """Hot temperature friction should be lower."""
        mu_ref = _ptfe_friction("glass_filled", "reference")
        mu_hot = _ptfe_friction("glass_filled", "hot")
        assert mu_hot["mu_slow"] < mu_ref["mu_slow"]
        assert mu_hot["mu_fast"] < mu_ref["mu_fast"]

    def test_all_types_valid(self):
        """All PTFE types should produce valid friction values."""
        for ptfe_type in PTFE_FRICTION:
            mu = _ptfe_friction(ptfe_type, "reference")
            assert mu["mu_slow"] > 0
            assert mu["mu_fast"] > mu["mu_slow"]


class TestPTFEBearing:
    def test_upper_lower_bound(self):
        cfg = PTFEConfig(ptfe_type="glass_filled")
        bm = build_ptfe_sliding(cfg)
        assert bm.cases["upper_bound"]["mu_slow"] > bm.properties["mu_slow"]
        assert bm.cases["lower_bound"]["mu_slow"] < bm.properties["mu_slow"]

    def test_compression_only(self):
        bm = build_ptfe_sliding(PTFEConfig())
        assert bm.compression_only is True


class TestFPSingle:
    def test_period_calculation(self):
        """T = 2π√(R/g)."""
        cfg = FPSingleConfig(radius_in=40.0, mu=0.06)
        bm = build_fp_single(cfg)
        T_expected = 2 * math.pi * math.sqrt(40.0 / G_ACCEL)
        assert abs(bm.properties["period_sec"] - T_expected) < 0.01

    def test_upper_lower_bound(self):
        cfg = FPSingleConfig(mu=0.06)
        bm = build_fp_single(cfg)
        assert bm.cases["upper_bound"]["mu"] > 0.06
        assert bm.cases["lower_bound"]["mu"] < 0.06

    def test_compression_only(self):
        bm = build_fp_single(FPSingleConfig())
        assert bm.compression_only is True


class TestFPTriple:
    def test_multi_stage_properties(self):
        cfg = FPTripleConfig()
        bm = build_fp_triple(cfg)
        assert "R1" in bm.properties
        assert "mu4" in bm.properties
        assert bm.properties["total_displacement_capacity_in"] == sum(
            [cfg.d1, cfg.d2, cfg.d3, cfg.d4]
        )

    def test_upper_lower_bound(self):
        cfg = FPTripleConfig(mu2=0.05)
        bm = build_fp_triple(cfg)
        assert bm.cases["upper_bound"]["mu2"] > 0.05
        assert bm.cases["lower_bound"]["mu2"] < 0.05


class TestIntegralBearing:
    def test_equaldof_constraint(self):
        bm = build_integral()
        assert len(bm.constraints) == 1
        assert bm.constraints[0]["type"] == "equalDOF"
        assert bm.constraints[0]["dofs"] == [1, 2, 3, 4, 5, 6]

    def test_not_compression_only(self):
        """Integral bearings transfer tension too."""
        bm = build_integral()
        assert bm.compression_only is False

    def test_no_elements(self):
        """Integral uses constraint, not element."""
        bm = build_integral()
        assert len(bm.elements) == 0


class TestRockerRoller:
    def test_uplift_warning(self):
        cfg = RockerRollerConfig()
        bm = build_rocker_roller(cfg)
        assert len(bm.warnings) > 0
        assert "uplift" in bm.warnings[0].lower()

    def test_compression_only(self):
        bm = build_rocker_roller(RockerRollerConfig())
        assert bm.compression_only is True

    def test_hertz_capacity(self):
        cfg = RockerRollerConfig(rocker_radius_in=6.0, steel_fy_ksi=36.0)
        bm = build_rocker_roller(cfg)
        D = 12.0  # 2 × radius
        expected = math.pi / 4 * 36.0 * D
        assert abs(bm.properties["hertz_line_capacity_kip_per_in"] - expected) < 0.1


class TestBearingLayout:
    def test_correct_count(self):
        """One bearing per girder line."""
        bearings = layout_bearings(5, 8.0, "elastomeric")
        assert len(bearings) == 5

    def test_transverse_positions(self):
        """Bearings at correct transverse locations."""
        bearings = layout_bearings(3, 8.0, "pot_fixed", z_center=0.0)
        z_positions = [b.nodes[0]["z"] for b in bearings]
        spacing = 8.0 * FT_TO_IN
        assert abs(z_positions[1] - z_positions[0] - spacing) < 0.1
        assert abs(z_positions[2] - z_positions[1] - spacing) < 0.1

    def test_symmetric_about_center(self):
        """Bearings should be symmetric about center."""
        bearings = layout_bearings(4, 10.0, "elastomeric", z_center=0.0)
        z_positions = sorted(b.nodes[0]["z"] for b in bearings)
        # Should be symmetric: -180, -60, 60, 180
        for i in range(len(z_positions) // 2):
            assert abs(z_positions[i] + z_positions[-(i+1)]) < 0.1

    def test_unique_tags(self):
        """All bearings should have unique tags."""
        bearings = layout_bearings(4, 8.0, "elastomeric")
        all_node_tags = []
        all_elem_tags = []
        for b in bearings:
            all_node_tags.extend(n["tag"] for n in b.nodes)
            all_elem_tags.extend(e["tag"] for e in b.elements)
        assert len(all_node_tags) == len(set(all_node_tags))
        assert len(all_elem_tags) == len(set(all_elem_tags))


class TestCompressionOnlyAllTypes:
    """ALL non-integral bearings must have compression-only behavior."""

    @pytest.mark.parametrize("bearing_type,kwargs", [
        ("elastomeric", {}),
        ("pot_fixed", {}),
        ("pot_guided", {"guide_direction": 1}),
        ("ptfe_sliding", {}),
        ("fp_single", {}),
        ("fp_triple", {}),
        ("rocker_roller", {}),
    ])
    def test_compression_only_flag(self, bearing_type, kwargs):
        bm = create_bearing(bearing_type, **kwargs)
        assert bm.compression_only is True, (
            f"{bearing_type} must be compression-only"
        )

    def test_integral_not_compression_only(self):
        bm = create_bearing("integral")
        assert bm.compression_only is False


class TestUpperLowerBound:
    """All friction-dependent bearings must produce upper/lower bound cases."""

    @pytest.mark.parametrize("bearing_type", [
        "elastomeric",
        "ptfe_sliding",
        "fp_single",
        "fp_triple",
    ])
    def test_has_bounds(self, bearing_type):
        bm = create_bearing(bearing_type)
        assert "upper_bound" in bm.cases
        assert "lower_bound" in bm.cases


class TestCreateBearing:
    def test_dispatch(self):
        bm = create_bearing("elastomeric")
        assert bm.bearing_type == "elastomeric"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_bearing("magic_carpet")

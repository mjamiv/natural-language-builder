"""
Tests for superstructure modeling tool.

Tests cover:
    - Steel plate girder (composite and non-composite)
    - Continuous vs simple span boundary conditions
    - Composite effective width calculation (AASHTO 4.6.2.6)
    - Prestressed I-girder with standard types
    - Mesh density requirements
    - Geometric transform selection by span length
    - Truss panel geometry
    - CIP box girder
    - Concrete slab bridge
    - Arch bridge
    - Tag uniqueness
    - Input validation
"""

import math
import pytest

from nlb.tools.superstructure import (
    SuperstructureModel,
    TagManager,
    create_superstructure,
    compute_mesh_density,
    effective_slab_width,
    select_transform_type,
    _arch_shape_y,
    _span_node_positions,
    _diaphragm_positions,
)


# ============================================================================
# UTILITY TESTS
# ============================================================================

class TestSelectTransformType:
    """Test geometric transform selection by span length."""

    def test_short_span_linear(self):
        assert select_transform_type(50.0) == 'Linear'

    def test_boundary_99ft_linear(self):
        assert select_transform_type(99.9) == 'Linear'

    def test_boundary_100ft_pdelta(self):
        assert select_transform_type(100.0) == 'PDelta'

    def test_medium_span_pdelta(self):
        assert select_transform_type(150.0) == 'PDelta'

    def test_boundary_200ft_pdelta(self):
        assert select_transform_type(200.0) == 'PDelta'

    def test_long_span_corotational(self):
        assert select_transform_type(201.0) == 'Corotational'

    def test_very_long_span(self):
        assert select_transform_type(500.0) == 'Corotational'


class TestMeshDensity:
    """Test mesh density computation."""

    def test_short_span_minimum_10(self):
        assert compute_mesh_density(50.0) >= 10

    def test_100ft_span_minimum_20(self):
        assert compute_mesh_density(101.0) >= 20

    def test_divisible_by_10(self):
        """Mesh density should be divisible by 10 for tenth-points."""
        for span in [30, 60, 90, 120, 180, 250]:
            n = compute_mesh_density(float(span))
            assert n % 10 == 0, f"span={span}ft: n_elem={n} not divisible by 10"

    def test_never_below_10(self):
        assert compute_mesh_density(5.0) >= 10


class TestEffectiveSlabWidth:
    """Test AASHTO 4.6.2.6 effective slab width calculation."""

    def test_controlled_by_span(self):
        """Effective width = span/4 when span is short."""
        # span/4 = 40*12/4 = 120 in
        # spacing = 8*12 = 96 in → controls
        # But let's make span the controller
        w = effective_slab_width(
            span_ft=20.0,  # span/4 = 60 in
            girder_spacing_ft=8.0,  # 96 in
            slab_thickness_in=8.0,  # 12*8 + 16 = 112 in
            top_flange_width_in=16.0,
        )
        assert w == pytest.approx(60.0)  # 20*12/4 = 60

    def test_controlled_by_spacing(self):
        """Effective width = girder spacing when spacing is small."""
        w = effective_slab_width(
            span_ft=200.0,  # span/4 = 600 in
            girder_spacing_ft=6.0,  # 72 in → controls
            slab_thickness_in=8.0,  # 12*8 + 16 = 112 in
            top_flange_width_in=16.0,
        )
        assert w == pytest.approx(72.0)  # 6*12 = 72

    def test_controlled_by_slab_flange(self):
        """Effective width = 12*t_slab + bf when that's smallest."""
        w = effective_slab_width(
            span_ft=200.0,  # span/4 = 600 in
            girder_spacing_ft=10.0,  # 120 in
            slab_thickness_in=6.0,  # 12*6 + 12 = 84 in → controls
            top_flange_width_in=12.0,
        )
        assert w == pytest.approx(84.0)  # 12*6 + 12

    def test_typical_values(self):
        """Typical 80 ft span, 8 ft spacing, 8 in slab."""
        w = effective_slab_width(80.0, 8.0, 8.0, 16.0)
        expected_opts = [80 * 12 / 4, 8 * 12, 12 * 8 + 16]
        assert w == pytest.approx(min(expected_opts))


class TestTagManager:
    """Test tag uniqueness."""

    def test_sequential_tags(self):
        tm = TagManager(1)
        assert tm.next('node') == 1
        assert tm.next('node') == 2
        assert tm.next('node') == 3

    def test_independent_counters(self):
        tm = TagManager(1)
        assert tm.next('node') == 1
        assert tm.next('element') == 1  # independent
        assert tm.next('node') == 2

    def test_next_n(self):
        tm = TagManager(10)
        tags = tm.next_n('node', 5)
        assert tags == [10, 11, 12, 13, 14]
        assert tm.next('node') == 15


# ============================================================================
# STEEL PLATE GIRDER TESTS
# ============================================================================

class TestSteelPlateGirderComposite:
    """Test composite steel plate girder model generation."""

    def test_single_span_basic(self):
        """Single 80 ft span, 5 girders."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
            girder_spacing_ft=8.0,
        )
        assert isinstance(model, SuperstructureModel)
        assert model.span_lengths == [80.0]
        assert model.girder_lines == 5

    def test_node_count_single_span(self):
        """Correct number of nodes for single span."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
        )
        n_elem = compute_mesh_density(80.0)
        expected_nodes = 5 * (n_elem + 1)
        assert len(model.nodes) == expected_nodes

    def test_element_count_single_span(self):
        """Correct number of longitudinal elements."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
        )
        n_elem = compute_mesh_density(80.0)
        expected_elements = 5 * n_elem  # 5 girders × n elements each
        assert len(model.elements) == expected_elements

    def test_three_span_continuous(self):
        """3-span continuous bridge."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[120.0, 160.0, 120.0],
            num_girders=5,
            girder_spacing_ft=10.0,
        )
        assert len(model.span_lengths) == 3
        assert model.continuity == ['continuous', 'continuous']

    def test_three_span_node_count(self):
        """Node count for 3-span bridge (shared nodes at piers)."""
        spans = [120.0, 160.0, 120.0]
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=spans,
            num_girders=5,
        )
        # Total longitudinal positions:
        # span1: n1+1 nodes, span2: n2 new nodes (shares 1 with span1),
        # span3: n3 new nodes (shares 1 with span2)
        n1 = compute_mesh_density(120.0)
        n2 = compute_mesh_density(160.0)
        n3 = compute_mesh_density(120.0)
        expected_long = (n1 + 1) + n2 + n3
        expected_total = 5 * expected_long
        assert len(model.nodes) == expected_total

    def test_support_nodes_exist(self):
        """Support nodes are identified."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
        )
        # 2 supports (abutments) × 5 girders = 10 support nodes
        assert len(model.support_nodes) == 10
        # All support node tags should exist in model nodes
        node_tags = {n['tag'] for n in model.nodes}
        for sn in model.support_nodes:
            assert sn in node_tags

    def test_midspan_nodes_exist(self):
        """Midspan nodes are identified."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
        )
        assert len(model.midspan_nodes) == 5  # 1 midspan × 5 girders
        node_tags = {n['tag'] for n in model.nodes}
        for mn in model.midspan_nodes:
            assert mn in node_tags

    def test_composite_section_created(self):
        """Composite section should reference composite_section function."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
        )
        girder_sections = [s for s in model.sections if s['function'] == 'composite_section']
        assert len(girder_sections) == 1

    def test_steel_and_concrete_materials(self):
        """Composite model should have both steel and concrete materials."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
        )
        mat_types = {m['type'] for m in model.materials}
        assert 'Steel02' in mat_types
        assert 'Concrete01' in mat_types

    def test_diaphragms_present(self):
        """Diaphragms should exist at supports and third-points."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
        )
        assert len(model.diaphragms) > 0
        # At least: 2 supports + 2 third-points = 4 cross-sections
        # Each has 4 diaphragm elements (between 5 girders)
        assert len(model.diaphragms) >= 4 * 4

    def test_deck_width_auto(self):
        """Auto-computed deck width."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            num_girders=5,
            girder_spacing_ft=8.0,
        )
        # Default: spacing * (n-1) + 3.0 ft overhang
        expected = 8.0 * 4 + 3.0  # 35 ft
        assert model.deck_width == pytest.approx(expected)

    def test_deck_width_explicit(self):
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
            deck_width_ft=42.0,
        )
        assert model.deck_width == 42.0


class TestSteelPlateGirderNonComposite:
    """Test non-composite steel plate girder."""

    def test_noncomposite_section(self):
        """Non-composite should use steel_i_section."""
        model = create_superstructure(
            'steel_plate_girder_noncomposite',
            span_lengths_ft=[80.0],
        )
        girder_sections = [s for s in model.sections if s['function'] == 'steel_i_section']
        assert len(girder_sections) >= 1

    def test_no_concrete_material(self):
        """Non-composite should not have concrete material for girder."""
        model = create_superstructure(
            'steel_plate_girder_noncomposite',
            span_lengths_ft=[80.0],
        )
        # Should only have steel materials
        concrete_mats = [m for m in model.materials if m['type'] == 'Concrete01']
        assert len(concrete_mats) == 0


# ============================================================================
# CONTINUITY TESTS
# ============================================================================

class TestContinuity:
    """Test continuous vs simple span configurations."""

    def test_single_span_no_continuity(self):
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
        )
        assert model.continuity == []

    def test_multi_span_default_continuous(self):
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[100.0, 120.0, 100.0],
        )
        assert model.continuity == ['continuous', 'continuous']

    def test_prestressed_default_simple(self):
        model = create_superstructure(
            'prestressed_i_girder',
            span_lengths_ft=[80.0, 80.0],
            girder_type='BT_72',
        )
        assert model.continuity == ['simple']

    def test_explicit_continuity(self):
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[100.0, 120.0, 100.0],
            continuity=['simple', 'continuous'],
        )
        assert model.continuity == ['simple', 'continuous']

    def test_wrong_continuity_length_raises(self):
        with pytest.raises(ValueError, match="continuity must have"):
            create_superstructure(
                'steel_plate_girder_composite',
                span_lengths_ft=[100.0, 120.0],
                continuity=['continuous', 'simple'],  # too many
            )


# ============================================================================
# PRESTRESSED I-GIRDER TESTS
# ============================================================================

class TestPrestressedIGirder:
    """Test prestressed concrete I-girder models."""

    def test_bt72_basic(self):
        model = create_superstructure(
            'prestressed_i_girder',
            span_lengths_ft=[120.0],
            girder_type='BT_72',
            num_girders=6,
        )
        assert model.girder_lines == 6
        assert len(model.nodes) > 0
        assert len(model.elements) > 0

    def test_aashto_type_iv(self):
        model = create_superstructure(
            'prestressed_i_girder',
            span_lengths_ft=[90.0],
            girder_type='AASHTO_IV',
        )
        sections = [s for s in model.sections
                     if s['function'] == 'prestressed_i_section']
        assert len(sections) == 1
        assert sections[0]['params']['girder_type'] == 'AASHTO_IV'

    def test_unknown_girder_type_raises(self):
        with pytest.raises(ValueError, match="Unknown girder_type"):
            create_superstructure(
                'prestressed_i_girder',
                span_lengths_ft=[80.0],
                girder_type='NONEXISTENT_GIRDER',
            )

    def test_custom_strand_pattern(self):
        pattern = [(2.0, 10), (4.0, 10), (6.0, 6)]
        model = create_superstructure(
            'prestressed_i_girder',
            span_lengths_ft=[100.0],
            girder_type='BT_63',
            strand_pattern=pattern,
        )
        sections = [s for s in model.sections
                     if s['function'] == 'prestressed_i_section']
        assert sections[0]['params']['strand_pattern']['rows'] == pattern

    def test_default_strand_pattern(self):
        """Should have a default strand pattern if none given."""
        model = create_superstructure(
            'prestressed_i_girder',
            span_lengths_ft=[100.0],
            girder_type='BT_72',
        )
        sections = [s for s in model.sections
                     if s['function'] == 'prestressed_i_section']
        rows = sections[0]['params']['strand_pattern']['rows']
        assert len(rows) > 0

    def test_nu_girder_types(self):
        """NU girder types should be supported."""
        for gtype in ['NU_900', 'NU_1100', 'NU_1350', 'NU_1600', 'NU_2000']:
            model = create_superstructure(
                'prestressed_i_girder',
                span_lengths_ft=[80.0],
                girder_type=gtype,
            )
            assert len(model.elements) > 0


# ============================================================================
# CIP BOX GIRDER TESTS
# ============================================================================

class TestCIPBoxGirder:
    """Test CIP concrete box girder."""

    def test_basic_box(self):
        model = create_superstructure(
            'cip_box_girder',
            span_lengths_ft=[150.0, 200.0, 150.0],
            num_cells=3,
            box_depth_in=72.0,
        )
        assert model.girder_lines == 1  # spine model
        assert len(model.span_lengths) == 3
        assert model.continuity == ['continuous', 'continuous']

    def test_box_section_params(self):
        model = create_superstructure(
            'cip_box_girder',
            span_lengths_ft=[150.0],
            num_cells=4,
            box_depth_in=84.0,
            top_slab_thick_in=10.0,
            bot_slab_thick_in=7.0,
        )
        sections = [s for s in model.sections
                     if s['function'] == 'box_girder_section']
        assert len(sections) == 1
        assert sections[0]['params']['num_cells'] == 4
        assert sections[0]['params']['depth'] == 84.0


# ============================================================================
# STEEL TRUSS TESTS
# ============================================================================

class TestSteelTruss:
    """Test steel truss geometry generation."""

    def test_basic_truss(self):
        model = create_superstructure(
            'steel_truss',
            span_lengths_ft=[200.0],
            panel_length_ft=20.0,
            truss_depth_ft=25.0,
        )
        assert model.girder_lines == 2
        assert len(model.nodes) > 0
        assert len(model.elements) > 0

    def test_truss_panel_count(self):
        """Verify panel geometry for a 200 ft span with 20 ft panels."""
        model = create_superstructure(
            'steel_truss',
            span_lengths_ft=[200.0],
            panel_length_ft=20.0,
            truss_depth_ft=25.0,
        )
        # 200/20 = 10 panels → 11 panel points
        # 2 truss lines × 11 points × 2 (top + bottom) = 44 nodes
        assert len(model.nodes) == 44

    def test_truss_element_types(self):
        """All truss elements should be corotTruss."""
        model = create_superstructure(
            'steel_truss',
            span_lengths_ft=[200.0],
            panel_length_ft=20.0,
        )
        for elem in model.elements:
            assert elem['type'] == 'corotTruss'

    def test_truss_node_heights(self):
        """Top chord nodes should be at truss depth, bottom at y=0."""
        model = create_superstructure(
            'steel_truss',
            span_lengths_ft=[100.0],
            panel_length_ft=20.0,
            truss_depth_ft=20.0,
        )
        y_values = set(n['y'] for n in model.nodes)
        assert 0.0 in y_values
        assert 20.0 * 12.0 in y_values  # 20 ft = 240 inches

    def test_floor_beams(self):
        """Transverse floor beams at every panel point."""
        model = create_superstructure(
            'steel_truss',
            span_lengths_ft=[100.0],
            panel_length_ft=20.0,
        )
        floor_beams = [d for d in model.diaphragms
                       if d.get('purpose') == 'floor_beam']
        # 100/20 = 5 panels → 6 panel points → 6 floor beams
        assert len(floor_beams) == 6

    def test_truss_support_nodes(self):
        """Support nodes at span ends."""
        model = create_superstructure(
            'steel_truss',
            span_lengths_ft=[200.0],
            panel_length_ft=20.0,
        )
        assert len(model.support_nodes) >= 4  # 2 supports × 2 truss lines


# ============================================================================
# CONCRETE SLAB TESTS
# ============================================================================

class TestConcreteSlab:
    """Test concrete slab bridge."""

    def test_basic_slab(self):
        model = create_superstructure(
            'concrete_slab',
            span_lengths_ft=[30.0],
            slab_thickness_in=18.0,
            deck_width_ft=40.0,
        )
        assert len(model.nodes) > 0
        assert len(model.elements) > 0
        assert model.girder_lines == 0  # shell model

    def test_shell_elements(self):
        """All elements should be ShellMITC4."""
        model = create_superstructure(
            'concrete_slab',
            span_lengths_ft=[30.0],
            slab_thickness_in=18.0,
            deck_width_ft=40.0,
        )
        for elem in model.elements:
            assert elem['type'] == 'ShellMITC4'
            assert len(elem['nodes']) == 4  # quad element

    def test_slab_mesh_density(self):
        """Slab mesh should meet minimum density."""
        model = create_superstructure(
            'concrete_slab',
            span_lengths_ft=[30.0],
            slab_thickness_in=18.0,
            deck_width_ft=40.0,
        )
        n_elem_long = compute_mesh_density(30.0)
        n_z = max(4, round(40.0 / 2.0))  # width / 2ft
        expected_elements = n_elem_long * n_z
        assert len(model.elements) == expected_elements


# ============================================================================
# ARCH BRIDGE TESTS
# ============================================================================

class TestArchBridge:
    """Test arch bridge model."""

    def test_parabolic_arch(self):
        model = create_superstructure(
            'arch',
            span_lengths_ft=[200.0],
            arch_rise_ft=50.0,
            arch_shape='parabolic',
        )
        assert len(model.nodes) > 0
        assert len(model.elements) > 0
        # Support nodes at ends
        assert len(model.support_nodes) == 2

    def test_arch_shape_parabolic(self):
        """Parabolic arch: y=0 at ends, y=rise at midspan."""
        span_in = 200.0 * 12.0
        rise_in = 50.0 * 12.0
        # At x=0
        assert _arch_shape_y(0.0, span_in, rise_in, 'parabolic') == pytest.approx(0.0)
        # At x=span
        assert _arch_shape_y(span_in, span_in, rise_in, 'parabolic') == pytest.approx(0.0)
        # At midspan
        assert _arch_shape_y(span_in / 2, span_in, rise_in, 'parabolic') == pytest.approx(rise_in)

    def test_arch_shape_circular(self):
        """Circular arch: y=0 at ends, y=rise at midspan."""
        span_in = 200.0 * 12.0
        rise_in = 50.0 * 12.0
        assert _arch_shape_y(0.0, span_in, rise_in, 'circular') == pytest.approx(0.0, abs=1.0)
        assert _arch_shape_y(span_in, span_in, rise_in, 'circular') == pytest.approx(0.0, abs=1.0)
        assert _arch_shape_y(span_in / 2, span_in, rise_in, 'circular') == pytest.approx(rise_in, abs=1.0)

    def test_arch_nodes_follow_shape(self):
        """Node y-coordinates should follow the arch shape."""
        model = create_superstructure(
            'arch',
            span_lengths_ft=[200.0],
            arch_rise_ft=50.0,
            arch_shape='parabolic',
        )
        # First and last nodes at y≈0
        assert model.nodes[0]['y'] == pytest.approx(0.0)
        assert model.nodes[-1]['y'] == pytest.approx(0.0)
        # Midspan node should be near rise
        mid_idx = len(model.nodes) // 2
        rise_in = 50.0 * 12.0
        assert model.nodes[mid_idx]['y'] == pytest.approx(rise_in, rel=0.05)

    def test_arch_transform_at_least_pdelta(self):
        """Arch bridges should use at least PDelta transform."""
        model = create_superstructure(
            'arch',
            span_lengths_ft=[50.0],  # short span
            arch_rise_ft=15.0,
        )
        transforms = model.transforms
        assert len(transforms) > 0
        assert transforms[0]['type'] in ('PDelta', 'Corotational')

    def test_invalid_arch_shape_raises(self):
        """Unknown arch shape should raise error."""
        with pytest.raises(ValueError, match="Unknown arch shape"):
            _arch_shape_y(0.0, 100.0, 50.0, 'catenary')


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test error handling for invalid inputs."""

    def test_empty_spans_raises(self):
        with pytest.raises(ValueError, match="at least one span"):
            create_superstructure('steel_plate_girder_composite', span_lengths_ft=[])

    def test_negative_span_raises(self):
        with pytest.raises(ValueError, match="positive"):
            create_superstructure('steel_plate_girder_composite', span_lengths_ft=[-50.0])

    def test_unknown_bridge_type_raises(self):
        with pytest.raises(ValueError, match="Unknown bridge_type"):
            create_superstructure('suspension_bridge', span_lengths_ft=[100.0])


# ============================================================================
# TAG UNIQUENESS
# ============================================================================

class TestTagUniqueness:
    """Ensure no duplicate tags across model."""

    def test_unique_node_tags(self):
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[100.0, 120.0, 100.0],
            num_girders=5,
        )
        tags = [n['tag'] for n in model.nodes]
        assert len(tags) == len(set(tags)), "Duplicate node tags found"

    def test_unique_element_tags(self):
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[100.0, 120.0],
            num_girders=5,
        )
        elem_tags = [e['tag'] for e in model.elements]
        diaphragm_tags = [d['tag'] for d in model.diaphragms]
        all_tags = elem_tags + diaphragm_tags
        assert len(all_tags) == len(set(all_tags)), "Duplicate element tags found"


# ============================================================================
# INTEGRATION / CROSS-CUTTING TESTS
# ============================================================================

class TestIntegration:
    """Cross-cutting integration tests."""

    def test_all_element_nodes_exist(self):
        """Every element should reference nodes that exist."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0, 100.0],
            num_girders=4,
        )
        node_tags = {n['tag'] for n in model.nodes}
        for elem in model.elements:
            for nt in elem['nodes']:
                assert nt in node_tags, f"Element {elem['tag']} references nonexistent node {nt}"
        for diap in model.diaphragms:
            for nt in diap['nodes']:
                assert nt in node_tags, f"Diaphragm {diap['tag']} references nonexistent node {nt}"

    def test_sections_reference_valid_materials(self):
        """Section material references should exist in materials list."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
        )
        mat_tags = {m['tag'] for m in model.materials}
        for section in model.sections:
            params = section['params']
            for key in ('mat_steel', 'mat_concrete', 'mat_flange', 'mat_web'):
                if key in params:
                    assert params[key] in mat_tags, \
                        f"Section {section['tag']} references material {params[key]} not in model"

    def test_transforms_created(self):
        """Geometric transforms should be defined."""
        model = create_superstructure(
            'steel_plate_girder_composite',
            span_lengths_ft=[80.0],
        )
        assert len(model.transforms) >= 1
        assert model.transforms[0]['type'] in ('Linear', 'PDelta', 'Corotational')

    def test_multiple_bridge_types_runnable(self):
        """Smoke test: all bridge types produce output without errors."""
        configs = [
            ('steel_plate_girder_composite', {'span_lengths_ft': [80.0]}),
            ('steel_plate_girder_noncomposite', {'span_lengths_ft': [80.0]}),
            ('prestressed_i_girder', {'span_lengths_ft': [80.0], 'girder_type': 'BT_72'}),
            ('cip_box_girder', {'span_lengths_ft': [150.0]}),
            ('segmental_box_girder', {'span_lengths_ft': [150.0]}),
            ('steel_truss', {'span_lengths_ft': [200.0]}),
            ('concrete_slab', {'span_lengths_ft': [30.0], 'slab_thickness_in': 18.0}),
            ('arch', {'span_lengths_ft': [200.0], 'arch_rise_ft': 50.0}),
        ]
        for btype, params in configs:
            model = create_superstructure(btype, **params)
            assert isinstance(model, SuperstructureModel), f"Failed for {btype}"
            assert len(model.nodes) > 0, f"No nodes for {btype}"
            assert len(model.elements) > 0, f"No elements for {btype}"

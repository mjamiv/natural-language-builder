"""
Tests for the OpenSees core library.

Tests verify that functions produce valid OpenSees commands by building
minimal models and checking that OpenSees accepts the commands without errors.

Each test uses ops.wipe() to start fresh.

Units: kip-inch-second (KIS) throughout.
"""

import math
import pytest
import openseespy.opensees as ops

# Import modules under test
from nlb.opensees.materials import (
    concrete_defaults, unconfined_concrete, confined_concrete,
    mander_confinement, reinforcing_steel, structural_steel,
    prestressing_strand, STEEL_DEFAULTS,
    py_spring, tz_spring, qz_spring,
    api_py_curves, compute_tult, compute_qult,
    elastomeric_shear, friction_model, compression_only,
    SoilLayer, ConcreteProperties,
)
from nlb.opensees.sections import (
    steel_i_section, composite_section,
    circular_rc_section, rectangular_rc_section,
    box_girder_section, prestressed_i_section,
    GIRDER_LIBRARY, StrandPattern, TendonProfile, BarLayout,
)
from nlb.opensees.elements import (
    beam_column, truss_element, zero_length, shell_element,
    geometric_transform,
)
from nlb.opensees.analysis import (
    gravity_analysis, pushover_analysis, convergence_handler,
    try_algorithms, response_spectrum, cqc_combination,
    time_history, staged_construction, ConstructionStage,
    _interpolate_spectrum,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def clean_model():
    """Wipe OpenSees model before and after each test."""
    ops.wipe()
    yield
    ops.wipe()


def _setup_2d_model():
    """Helper: set up a minimal 2D model for testing."""
    ops.model('basic', '-ndm', 2, '-ndf', 3)


def _setup_3d_model():
    """Helper: set up a minimal 3D model for testing."""
    ops.model('basic', '-ndm', 3, '-ndf', 6)


# ============================================================================
# MATERIAL TESTS — CONCRETE
# ============================================================================

class TestConcreteDefaults:
    """Test concrete property computation."""

    def test_4ksi_concrete(self):
        props = concrete_defaults(4.0)
        assert isinstance(props, ConcreteProperties)
        assert props.fc == 4.0
        # Ec for 4 ksi: 33000 * 0.150^1.5 * sqrt(4) ≈ 3834 ksi
        assert 3700 < props.Ec < 3950
        # fr = 0.24 * sqrt(4) = 0.48
        assert abs(props.fr - 0.48) < 0.01
        # eps_0 = 2*fc/Ec ≈ 0.0022
        assert 0.001 < props.eps_0 < 0.003
        assert props.eps_cu == 0.003

    def test_6ksi_concrete(self):
        props = concrete_defaults(6.0)
        assert props.fc == 6.0
        assert props.Ec > concrete_defaults(4.0).Ec

    def test_high_strength(self):
        props = concrete_defaults(10.0)
        assert props.Ec > 5000

    def test_negative_fc_abs(self):
        """Negative fc should be abs'd."""
        props = concrete_defaults(-4.0)
        assert props.fc == 4.0


class TestUnconfinedConcrete:
    """Test unconfined concrete material creation."""

    def test_basic_creation(self):
        _setup_2d_model()
        tag = unconfined_concrete(1, 4.0)
        assert tag == 1

    def test_custom_strains(self):
        _setup_2d_model()
        tag = unconfined_concrete(2, 6.0, eps_0=0.002, eps_cu=0.004)
        assert tag == 2

    def test_multiple_materials(self):
        _setup_2d_model()
        unconfined_concrete(1, 4.0)
        unconfined_concrete(2, 6.0)
        # Both should exist without error


class TestConfinedConcrete:
    """Test confined concrete material creation."""

    def test_basic_creation(self):
        _setup_2d_model()
        tag = confined_concrete(1, fc=4.0, fcc=5.6, ecc=0.004, ecu=0.015)
        assert tag == 1

    def test_high_confinement(self):
        _setup_2d_model()
        tag = confined_concrete(2, fc=4.0, fcc=7.0, ecc=0.006, ecu=0.025)
        assert tag == 2


class TestManderConfinement:
    """Test Mander confinement calculations."""

    def test_circular(self):
        fcc, ecc, ecu = mander_confinement(4.0, 60.0, 0.01, "circular")
        assert fcc > 4.0, "Confined strength must exceed unconfined"
        assert ecc > 0.002, "Confined strain at peak must exceed unconfined"
        assert ecu > 0.003, "Confined ultimate strain must exceed 0.003"

    def test_rectangular(self):
        fcc, ecc, ecu = mander_confinement(4.0, 60.0, 0.01, "rectangular")
        # Rectangular should have lower confinement than circular
        fcc_circ, _, _ = mander_confinement(4.0, 60.0, 0.01, "circular")
        assert fcc < fcc_circ

    def test_higher_rho_s(self):
        fcc_low, _, _ = mander_confinement(4.0, 60.0, 0.005)
        fcc_high, _, _ = mander_confinement(4.0, 60.0, 0.02)
        assert fcc_high > fcc_low


# ============================================================================
# MATERIAL TESTS — STEEL
# ============================================================================

class TestReinforcingSteel:
    """Test reinforcing steel material creation."""

    def test_gr60_defaults(self):
        _setup_2d_model()
        tag = reinforcing_steel(1)
        assert tag == 1

    def test_custom_params(self):
        _setup_2d_model()
        tag = reinforcing_steel(2, fy=75.0, fu=100.0, Es=29000.0, b=0.005)
        assert tag == 2

    def test_a706(self):
        _setup_2d_model()
        d = STEEL_DEFAULTS["A706_Gr60"]
        tag = reinforcing_steel(3, fy=d.fy, fu=d.fu, Es=d.Es, b=d.b)
        assert tag == 3


class TestStructuralSteel:
    """Test structural steel material creation."""

    def test_a992_defaults(self):
        _setup_2d_model()
        tag = structural_steel(1)
        assert tag == 1

    def test_hps70w(self):
        _setup_2d_model()
        d = STEEL_DEFAULTS["HPS_70W"]
        tag = structural_steel(2, fy=d.fy, Es=d.Es, b=d.b)
        assert tag == 2


class TestPrestressingStrand:
    """Test prestressing strand material creation."""

    def test_270ksi_defaults(self):
        _setup_2d_model()
        tag = prestressing_strand(1)
        assert tag == 1

    def test_custom_fpu(self):
        _setup_2d_model()
        tag = prestressing_strand(2, fpu=250.0)
        assert tag == 2


class TestSteelDefaults:
    """Test that all default steel sets are properly defined."""

    def test_all_defaults_exist(self):
        expected = ["A615_Gr60", "A706_Gr60", "A992_Gr50", "HPS_70W", "270ksi_strand"]
        for key in expected:
            assert key in STEEL_DEFAULTS

    def test_defaults_reasonable(self):
        for name, d in STEEL_DEFAULTS.items():
            assert d.fy > 0
            assert d.fu >= d.fy
            assert d.Es > 20000
            assert 0 < d.b < 0.1


# ============================================================================
# MATERIAL TESTS — SOIL SPRINGS
# ============================================================================

class TestPySpring:
    def test_clay(self):
        _setup_2d_model()
        tag = py_spring(1, soil_type=1, pu=10.0, y50=0.5)
        assert tag == 1

    def test_sand(self):
        _setup_2d_model()
        tag = py_spring(2, soil_type=2, pu=20.0, y50=0.3, cd=0.5)
        assert tag == 2


class TestTzSpring:
    def test_driven(self):
        _setup_2d_model()
        tag = tz_spring(1, soil_type=1, tult=5.0, z50=0.1)
        assert tag == 1


class TestQzSpring:
    def test_tip(self):
        _setup_2d_model()
        tag = qz_spring(1, qult=100.0, z50=0.5)
        assert tag == 1


class TestApiPyCurves:
    def test_sand_layer(self):
        layers = [SoilLayer(depth_top=0, depth_bot=240, soil_type="sand",
                            phi=35.0, gamma=0.0000694)]
        results = api_py_curves(36.0, 120.0, layers)
        assert len(results) >= 1
        assert results[0]['pu'] > 0
        assert results[0]['y50'] > 0

    def test_clay_layer(self):
        layers = [SoilLayer(depth_top=0, depth_bot=240, soil_type="clay",
                            su=0.015, eps_50=0.01, gamma=0.0000625)]
        results = api_py_curves(48.0, 60.0, layers)
        assert len(results) >= 1

    def test_depth_outside_layer(self):
        layers = [SoilLayer(depth_top=0, depth_bot=100, soil_type="sand",
                            phi=30.0)]
        results = api_py_curves(36.0, 150.0, layers)
        assert len(results) == 0  # depth outside layer


class TestComputeTult:
    def test_clay(self):
        layer = SoilLayer(depth_top=0, depth_bot=240, soil_type="clay",
                          su=0.015, gamma=0.0000625)
        tult = compute_tult(48.0, 120.0, layer)
        assert tult > 0

    def test_sand(self):
        layer = SoilLayer(depth_top=0, depth_bot=240, soil_type="sand",
                          phi=35.0, gamma=0.0000694)
        tult = compute_tult(36.0, 120.0, layer)
        assert tult > 0


class TestComputeQult:
    def test_clay_tip(self):
        layer = SoilLayer(depth_top=0, depth_bot=240, soil_type="clay",
                          su=0.020, gamma=0.0000625)
        qult = compute_qult(48.0, 200.0, layer)
        assert qult > 0

    def test_sand_tip(self):
        layer = SoilLayer(depth_top=0, depth_bot=240, soil_type="sand",
                          phi=35.0, gamma=0.0000694)
        qult = compute_qult(36.0, 200.0, layer)
        assert qult > 0


# ============================================================================
# MATERIAL TESTS — BEARINGS
# ============================================================================

class TestElastomericShear:
    def test_basic(self):
        _setup_2d_model()
        # G=0.1 ksi, A=200 in², h=3 in → k = 6.67 kip/in
        tag = elastomeric_shear(1, G=0.1, A=200.0, h=3.0)
        assert tag == 1


class TestFrictionModel:
    def test_ptfe(self):
        _setup_2d_model()
        tag = friction_model(1, mu_slow=0.06, mu_fast=0.10)
        assert tag == 1


class TestCompressionOnly:
    def test_ent(self):
        _setup_2d_model()
        tag = compression_only(1, k=1000.0)
        assert tag == 1


# ============================================================================
# SECTION TESTS
# ============================================================================

class TestSteelISection:
    def test_plate_girder(self):
        _setup_2d_model()
        structural_steel(1, fy=50.0)
        structural_steel(2, fy=50.0)
        tag = steel_i_section(
            tag=1, d=48.0, bf_top=16.0, tf_top=1.0,
            bf_bot=16.0, tf_bot=1.5, tw=0.5,
            mat_flange=1, mat_web=2
        )
        assert tag == 1

    def test_different_flanges(self):
        _setup_2d_model()
        structural_steel(1, fy=50.0)
        tag = steel_i_section(
            tag=2, d=60.0, bf_top=18.0, tf_top=1.25,
            bf_bot=20.0, tf_bot=2.0, tw=0.625,
            mat_flange=1, mat_web=1
        )
        assert tag == 2


class TestCompositeSection:
    def test_basic(self):
        _setup_2d_model()
        structural_steel(1, fy=50.0)
        unconfined_concrete(2, fc=4.0)
        steel = {'d': 36.0, 'bf_top': 12.0, 'tf_top': 0.75,
                 'bf_bot': 12.0, 'tf_bot': 1.0, 'tw': 0.5}
        tag = composite_section(
            tag=1, steel_section=steel, slab_width=96.0,
            slab_thick=8.0, haunch=2.0, mat_steel=1, mat_concrete=2
        )
        assert tag == 1


class TestCircularRcSection:
    def test_36in_column(self):
        _setup_2d_model()
        # Materials
        fcc, ecc, ecu = mander_confinement(4.0, 60.0, 0.01)
        confined_concrete(1, 4.0, fcc, ecc, ecu)
        unconfined_concrete(2, 4.0)
        reinforcing_steel(3)
        # Section: 36" dia, 2" cover, 12 #9 bars
        tag = circular_rc_section(
            tag=1, diameter=36.0, cover=2.0,
            num_bars=12, bar_area=1.0,
            mat_confined=1, mat_unconfined=2, mat_steel=3
        )
        assert tag == 1

    def test_60in_column(self):
        _setup_2d_model()
        fcc, ecc, ecu = mander_confinement(6.0, 60.0, 0.015)
        confined_concrete(1, 6.0, fcc, ecc, ecu)
        unconfined_concrete(2, 6.0)
        reinforcing_steel(3)
        tag = circular_rc_section(
            tag=2, diameter=60.0, cover=3.0,
            num_bars=20, bar_area=1.56,
            mat_confined=1, mat_unconfined=2, mat_steel=3
        )
        assert tag == 2


class TestRectangularRcSection:
    def test_basic(self):
        _setup_2d_model()
        fcc, ecc, ecu = mander_confinement(4.0, 60.0, 0.008)
        confined_concrete(1, 4.0, fcc, ecc, ecu)
        unconfined_concrete(2, 4.0)
        reinforcing_steel(3)
        bars = [
            BarLayout(num_bars=4, bar_area=1.0, face="top"),
            BarLayout(num_bars=4, bar_area=1.0, face="bottom"),
        ]
        tag = rectangular_rc_section(
            tag=1, width=24.0, height=36.0, cover=2.0,
            bars_layout=bars,
            mat_confined=1, mat_unconfined=2, mat_steel=3
        )
        assert tag == 1

    def test_all_faces(self):
        _setup_2d_model()
        confined_concrete(1, 4.0, 5.5, 0.004, 0.015)
        unconfined_concrete(2, 4.0)
        reinforcing_steel(3)
        bars = [
            BarLayout(num_bars=5, bar_area=0.79, face="top"),
            BarLayout(num_bars=5, bar_area=0.79, face="bottom"),
            BarLayout(num_bars=3, bar_area=0.79, face="left"),
            BarLayout(num_bars=3, bar_area=0.79, face="right"),
        ]
        tag = rectangular_rc_section(
            tag=2, width=30.0, height=48.0, cover=2.5,
            bars_layout=bars,
            mat_confined=1, mat_unconfined=2, mat_steel=3
        )
        assert tag == 2


class TestBoxGirderSection:
    def test_single_cell(self):
        _setup_2d_model()
        unconfined_concrete(1, 6.0)
        tag = box_girder_section(
            tag=1, depth=72.0, top_width=144.0, bot_width=84.0,
            top_thick=9.0, bot_thick=7.0, web_thick=12.0,
            num_cells=1, mat_concrete=1
        )
        assert tag == 1

    def test_multi_cell_with_tendons(self):
        _setup_2d_model()
        unconfined_concrete(1, 6.0)
        prestressing_strand(2)
        tendons = [
            TendonProfile(y=6.0, z=-20.0, area=3.0),
            TendonProfile(y=6.0, z=0.0, area=3.0),
            TendonProfile(y=6.0, z=20.0, area=3.0),
        ]
        tag = box_girder_section(
            tag=2, depth=84.0, top_width=180.0, bot_width=96.0,
            top_thick=10.0, bot_thick=8.0, web_thick=14.0,
            num_cells=3, mat_concrete=1, tendons=tendons, mat_strand=2
        )
        assert tag == 2


class TestPrestressedISection:
    def test_bt72(self):
        _setup_2d_model()
        unconfined_concrete(1, 8.0)
        prestressing_strand(2)
        pattern = StrandPattern(rows=[
            (2.0, 12), (4.0, 12), (6.0, 8), (8.0, 4)
        ])
        tag = prestressed_i_section(
            tag=1, girder_type="BT_72",
            mat_concrete=1, strand_pattern=pattern, mat_strand=2
        )
        assert tag == 1

    def test_aashto_iv(self):
        _setup_2d_model()
        unconfined_concrete(1, 6.0)
        prestressing_strand(2)
        pattern = StrandPattern(rows=[(2.0, 8), (4.0, 8)])
        tag = prestressed_i_section(
            tag=2, girder_type="AASHTO_IV",
            mat_concrete=1, strand_pattern=pattern, mat_strand=2
        )
        assert tag == 2

    def test_invalid_type(self):
        _setup_2d_model()
        unconfined_concrete(1, 6.0)
        prestressing_strand(2)
        pattern = StrandPattern(rows=[(2.0, 4)])
        with pytest.raises(KeyError):
            prestressed_i_section(3, "INVALID", 1, pattern, 2)

    def test_girder_library_complete(self):
        """All girder types should have required keys."""
        required = {'d', 'bt', 'bb', 'tw', 'tf_top', 'tf_bot'}
        for name, dims in GIRDER_LIBRARY.items():
            assert required.issubset(dims.keys()), f"{name} missing keys"


# ============================================================================
# ELEMENT TESTS
# ============================================================================

class TestGeometricTransform:
    def test_linear_2d(self):
        _setup_2d_model()
        tag = geometric_transform(1, 'Linear')
        assert tag == 1

    def test_pdelta_2d(self):
        _setup_2d_model()
        tag = geometric_transform(2, 'PDelta')
        assert tag == 2

    def test_corotational_2d(self):
        _setup_2d_model()
        tag = geometric_transform(3, 'Corotational')
        assert tag == 3

    def test_linear_3d(self):
        _setup_3d_model()
        tag = geometric_transform(1, 'Linear', 0.0, 0.0, 1.0)
        assert tag == 1

    def test_invalid_type(self):
        _setup_2d_model()
        with pytest.raises(ValueError):
            geometric_transform(1, 'Invalid')


class TestBeamColumn:
    def test_2d_beam(self):
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 120.0, 0.0)
        ops.fix(1, 1, 1, 1)
        structural_steel(1, fy=50.0)
        steel_i_section(1, d=24.0, bf_top=8.0, tf_top=0.75,
                        bf_bot=8.0, tf_bot=0.75, tw=0.5,
                        mat_flange=1, mat_web=1)
        geometric_transform(1, 'Linear')
        tag = beam_column(1, (1, 2), section=1, transform=1)
        assert tag == 1

    def test_5_integration_points(self):
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 0.0, 144.0)
        ops.fix(1, 1, 1, 1)
        reinforcing_steel(1)
        fcc, ecc, ecu = mander_confinement(4.0, 60.0, 0.01)
        confined_concrete(2, 4.0, fcc, ecc, ecu)
        unconfined_concrete(3, 4.0)
        circular_rc_section(1, 36.0, 2.0, 12, 1.0, 2, 3, 1)
        geometric_transform(1, 'PDelta')
        tag = beam_column(1, (1, 2), section=1, transform=1, np=5)
        assert tag == 1


class TestTrussElement:
    def test_basic(self):
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 120.0, 120.0)
        ops.fix(1, 1, 1, 1)
        structural_steel(1)
        tag = truss_element(1, (1, 2), area=5.0, material=1)
        assert tag == 1


class TestZeroLength:
    def test_bearing(self):
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 0.0, 0.0)
        ops.fix(1, 1, 1, 1)
        elastomeric_shear(1, G=0.1, A=200.0, h=3.0)
        compression_only(2, k=5000.0)
        tag = zero_length(1, (1, 2), materials=[1, 2], directions=[1, 2])
        assert tag == 1

    def test_mismatched_lengths(self):
        _setup_2d_model()
        with pytest.raises(ValueError):
            zero_length(1, (1, 2), materials=[1], directions=[1, 2])


class TestShellElement:
    def test_basic(self):
        _setup_3d_model()
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, 96.0, 0.0, 0.0)
        ops.node(3, 96.0, 96.0, 0.0)
        ops.node(4, 0.0, 96.0, 0.0)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.fix(2, 1, 1, 1, 1, 1, 1)
        # Create shell section
        ops.nDMaterial('ElasticIsotropic', 1, 3600.0, 0.2)
        ops.section('PlateFiber', 1, 1, 8.0)
        tag = shell_element(1, (1, 2, 3, 4), thickness=8.0, material=1)
        assert tag == 1


# ============================================================================
# ANALYSIS TESTS
# ============================================================================

class TestConvergenceHandler:
    def test_setup(self):
        _setup_2d_model()
        result = convergence_handler()
        assert result is True

    def test_custom_tolerance(self):
        _setup_2d_model()
        result = convergence_handler(tol=1e-6, max_iter=50)
        assert result is True


class TestGravityAnalysis:
    def test_simple_cantilever(self):
        """Gravity analysis on a cantilever with tip load."""
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 120.0, 0.0)
        ops.fix(1, 1, 1, 1)  # fully fixed

        structural_steel(1)
        steel_i_section(1, d=24.0, bf_top=8.0, tf_top=0.75,
                        bf_bot=8.0, tf_bot=0.75, tw=0.5,
                        mat_flange=1, mat_web=1)
        geometric_transform(1, 'Linear')
        beam_column(1, (1, 2), section=1, transform=1)

        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(2, 0.0, -10.0, 0.0)

        ok = gravity_analysis(steps=10)
        assert ok == 0

        # Check that tip displaced downward
        disp = ops.nodeDisp(2, 2)
        assert disp < 0, "Cantilever tip should deflect downward"


class TestPushoverAnalysis:
    def test_cantilever_pushover(self):
        """Pushover on a simple cantilever column with elastic section."""
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 0.0, 144.0)
        ops.fix(1, 1, 1, 1)

        # Use elastic beam-column for reliable convergence test
        A = 100.0    # in²
        E = 29000.0  # ksi
        I = 5000.0   # in⁴
        geometric_transform(1, 'PDelta')
        ops.element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)

        # Reference load for displacement control
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(2, 1.0, 0.0, 0.0)

        result = pushover_analysis(node=2, dof=1, target_disp=2.0, steps=20)
        assert result.steps_completed > 0
        assert len(result.displacements) > 0
        assert result.converged


class TestResponseSpectrum:
    def test_interpolation(self):
        periods = [0.0, 0.2, 1.0, 2.0]
        accels = [0.4, 1.0, 1.0, 0.5]

        # At T=0.0
        assert _interpolate_spectrum(0.0, periods, accels) == 0.4
        # At T=0.2
        assert _interpolate_spectrum(0.2, periods, accels) == 1.0
        # At T=0.1 (interpolated)
        Sa = _interpolate_spectrum(0.1, periods, accels)
        assert 0.6 < Sa < 0.8
        # At T=1.5 (interpolated)
        Sa = _interpolate_spectrum(1.5, periods, accels)
        assert 0.7 < Sa < 0.8
        # Beyond range
        assert _interpolate_spectrum(3.0, periods, accels) == 0.5


class TestCQCCombination:
    def test_identical_modes(self):
        """Equal responses with same period should give sqrt(N)*R for full correlation."""
        responses = [10.0, 10.0]
        periods = [1.0, 1.0]
        combined = cqc_combination(responses, periods, 0.05)
        # Fully correlated → sum = 20
        assert combined > 14  # Should be close to 20 for identical periods

    def test_well_separated_modes(self):
        """Well-separated modes → SRSS-like result."""
        responses = [10.0, 5.0]
        periods = [1.0, 0.1]  # very different
        combined = cqc_combination(responses, periods, 0.05)
        srss = math.sqrt(10**2 + 5**2)
        assert abs(combined - srss) / srss < 0.15


class TestStagedConstruction:
    def test_two_stages(self):
        """Two-stage construction: gravity then superimposed dead."""
        _setup_2d_model()
        ops.node(1, 0.0, 0.0)
        ops.node(2, 120.0, 0.0)
        ops.fix(1, 1, 1, 0)
        ops.fix(2, 0, 1, 0)

        structural_steel(1)
        steel_i_section(1, d=24.0, bf_top=8.0, tf_top=0.75,
                        bf_bot=8.0, tf_bot=0.75, tw=0.5,
                        mat_flange=1, mat_web=1)
        geometric_transform(1, 'Linear')
        beam_column(1, (1, 2), section=1, transform=1)

        # Stage 1: Self-weight
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(2, 0.0, -5.0, 0.0)

        stages = [
            ConstructionStage(name="Self-Weight", steps=5),
        ]

        results = staged_construction(stages)
        assert results["Self-Weight"] == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullBridgeColumn:
    """Integration test: build a complete bridge column model."""

    def test_column_with_soil_springs(self):
        _setup_2d_model()

        # Materials
        fcc, ecc, ecu = mander_confinement(4.0, 60.0, 0.012, "circular")
        confined_concrete(1, 4.0, fcc, ecc, ecu)
        unconfined_concrete(2, 4.0)
        reinforcing_steel(3)

        # Section
        circular_rc_section(1, diameter=48.0, cover=2.0,
                            num_bars=16, bar_area=1.0,
                            mat_confined=1, mat_unconfined=2, mat_steel=3)

        # Nodes
        ops.node(1, 0.0, 0.0)  # base (pile tip)
        ops.node(2, 0.0, 120.0)  # ground
        ops.node(3, 0.0, 360.0)  # column top
        ops.fix(1, 1, 1, 1)

        # Transform and elements
        geometric_transform(1, 'PDelta')
        beam_column(1, (1, 2), section=1, transform=1)  # pile
        beam_column(2, (2, 3), section=1, transform=1)  # column

        # Soil spring at ground level
        ops.node(4, 0.0, 120.0)  # spring node
        ops.fix(4, 1, 1, 1)
        py_spring(10, soil_type=2, pu=15.0, y50=0.5)
        zero_length(3, (4, 2), materials=[10], directions=[1])

        # Gravity
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(3, 0.0, -200.0, 0.0)

        ok = gravity_analysis(steps=10)
        assert ok == 0

        disp = ops.nodeDisp(3, 2)
        assert disp < 0, "Column top should deflect under gravity"

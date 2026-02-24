"""Tests for HL-93 moving load analysis integration in assembler.

Tests the generated OpenSees script code for moving load analysis,
vehicle definitions, envelope logic, and result extraction.
"""

import re
import unittest
from unittest.mock import patch

from nlb.tools.assembler import (
    AnalysisResults,
    _generate_analysis_script,
    build_analysis_sequence,
)


class TestMovingLoadScriptGeneration(unittest.TestCase):
    """Tests for the moving load analysis code generated into the OpenSees script."""

    def _make_elements(self, n=5, start_tag=30000):
        """Create mock superstructure elements."""
        els = []
        for i in range(n):
            els.append({
                "tag": start_tag + i,
                "type": "elasticBeamColumn",
                "_component": "superstructure",
                "nodes": [30100 + i, 30100 + i + 1],
            })
        return els

    def _make_nodes(self, n=6, start_tag=30100):
        """Create mock superstructure nodes."""
        return [
            {"tag": start_tag + i, "_component": "superstructure"}
            for i in range(n)
        ]

    def _generate(self, elements=None, nodes=None, has_moving_load=True):
        """Generate analysis script with moving load step."""
        seq = build_analysis_sequence(has_moving_load=has_moving_load)
        if elements is None:
            elements = self._make_elements()
        if nodes is None:
            nodes = self._make_nodes()
        return _generate_analysis_script(seq, elements, nodes)

    def test_placeholder_replaced(self):
        """The old placeholder text should not appear."""
        script = self._generate()
        self.assertNotIn("print('Moving load analysis: placeholder')", script)

    def test_hl93_header_present(self):
        """Generated script should have the HL-93 header."""
        script = self._generate()
        self.assertIn("Moving Load Analysis (HL-93)", script)

    def test_truck_axles_defined(self):
        """Design truck axles should appear in the generated code."""
        script = self._generate()
        # 8k front axle with 33% IM = 10.64
        self.assertIn("8.0 * (1 + _IM)", script)
        self.assertIn("32.0 * (1 + _IM)", script)

    def test_tandem_axles_defined(self):
        """Design tandem axles should appear in the generated code."""
        script = self._generate()
        self.assertIn("25.0 * (1 + _IM)", script)
        # 4 ft spacing
        self.assertIn("4.0 * 12.0", script)

    def test_lane_load_defined(self):
        """Lane load = 0.64 klf / 12 = kip/in."""
        script = self._generate()
        self.assertIn("0.64 / 12.0", script)

    def test_im_factor(self):
        """IM = 0.33 should be defined."""
        script = self._generate()
        self.assertIn("_IM = 0.33", script)

    def test_sup_nodes_embedded(self):
        """Superstructure node tags should be embedded in the script."""
        elements = self._make_elements(3, start_tag=30000)
        nodes = self._make_nodes(4, start_tag=30100)
        script = self._generate(elements=elements, nodes=nodes)
        # Should contain the node tags from elements
        self.assertIn("30100", script)
        self.assertIn("30101", script)
        self.assertIn("30102", script)
        self.assertIn("30103", script)

    def test_sup_elements_embedded(self):
        """Superstructure element tags should be embedded."""
        elements = self._make_elements(3, start_tag=30000)
        script = self._generate(elements=elements)
        self.assertIn("30000", script)
        self.assertIn("30001", script)
        self.assertIn("30002", script)

    def test_gravity_guard(self):
        """Moving load should only run if gravity_ok is True."""
        script = self._generate()
        # The moving load block is inside 'if gravity_ok:'
        self.assertIn("if gravity_ok:", script)

    def test_envelope_tracking(self):
        """Script should track max/min envelopes per element."""
        script = self._generate()
        self.assertIn("_ml_env_max", script)
        self.assertIn("_ml_env_min", script)

    def test_pattern_removal(self):
        """Each load position pattern should be removed after analysis."""
        script = self._generate()
        self.assertIn("ops.remove('loadPattern'", script)
        self.assertIn("ops.remove('timeSeries'", script)

    def test_both_vehicles_traversed(self):
        """Both truck and tandem should be traversed."""
        script = self._generate()
        self.assertIn("HL93_Truck", script)
        self.assertIn("HL93_Tandem", script)

    def test_results_stored_as_ml_results(self):
        """Results should be stored in _ml_results dict."""
        script = self._generate()
        self.assertIn("_ml_results", script)

    def test_no_moving_load_when_disabled(self):
        """When has_moving_load=False, no HL-93 code should appear."""
        seq = build_analysis_sequence(has_moving_load=False)
        script = _generate_analysis_script(seq, self._make_elements(), self._make_nodes())
        self.assertNotIn("HL93_Truck", script)
        self.assertNotIn("_ml_results", script)

    def test_non_superstructure_elements_excluded(self):
        """Foundation/substructure elements should NOT be included in moving load."""
        elements = [
            {"tag": 1000, "type": "elasticBeamColumn", "_component": "foundation", "nodes": [100, 101]},
            {"tag": 30000, "type": "elasticBeamColumn", "_component": "superstructure", "nodes": [30100, 30101]},
        ]
        nodes = [
            {"tag": 100, "_component": "foundation"},
            {"tag": 101, "_component": "foundation"},
            {"tag": 30100, "_component": "superstructure"},
            {"tag": 30101, "_component": "superstructure"},
        ]
        script = self._generate(elements=elements, nodes=nodes)
        # The _ml_sup_eles should only have 30000
        match = re.search(r'_ml_sup_eles = \[([^\]]*)\]', script)
        self.assertIsNotNone(match)
        self.assertIn("30000", match.group(1))
        self.assertNotIn("1000", match.group(1))

    def test_empty_elements_graceful(self):
        """No crash if no elements passed."""
        script = self._generate(elements=[], nodes=[])
        self.assertIn("Moving Load Analysis", script)
        self.assertIn("_ml_sup_eles = []", script)

    def test_force_labels_12dof(self):
        """Force labels should cover all 12 DOF (i and j ends)."""
        script = self._generate()
        for label in ['N_i', 'Vy_i', 'Vz_i', 'T_i', 'My_i', 'Mz_i',
                       'N_j', 'Vy_j', 'Vz_j', 'T_j', 'My_j', 'Mz_j']:
            self.assertIn(label, script)

    def test_controlling_combo_label(self):
        """Envelope entries should have HL93 as controlling_combo."""
        script = self._generate()
        self.assertIn("'controlling_combo': 'HL93'", script)


class TestAnalysisResultsMovingLoadField(unittest.TestCase):
    """Tests for the moving_load_envelopes field on AnalysisResults."""

    def test_field_exists(self):
        """AnalysisResults should have moving_load_envelopes field."""
        r = AnalysisResults()
        self.assertIsInstance(r.moving_load_envelopes, dict)

    def test_default_empty(self):
        """Default should be empty dict."""
        r = AnalysisResults()
        self.assertEqual(r.moving_load_envelopes, {})

    def test_can_set(self):
        """Should be able to set moving_load_envelopes."""
        r = AnalysisResults()
        r.moving_load_envelopes = {30000: {"N_i": {"max": 10.0, "min": -5.0}}}
        self.assertEqual(len(r.moving_load_envelopes), 1)


class TestExtractionCodeMovingLoads(unittest.TestCase):
    """Tests for the extraction code that merges moving load results."""

    def test_extraction_code_has_ml_merge(self):
        """The extraction code template should merge _ml_results."""
        from nlb.tools import assembler
        import inspect
        source = inspect.getsource(assembler.run_analysis)
        self.assertIn("moving_load_envelopes", source)

    def test_extraction_parses_ml_envelopes(self):
        """run_analysis should map moving_load_envelopes from parsed data."""
        from nlb.tools import assembler
        import inspect
        source = inspect.getsource(assembler.run_analysis)
        self.assertIn("moving_load_envelopes", source)


class TestBuildSequenceMovingLoad(unittest.TestCase):
    """Tests for build_analysis_sequence with moving load flag."""

    def test_moving_load_in_sequence(self):
        """moving_load_analysis should be in sequence when enabled."""
        seq = build_analysis_sequence(has_moving_load=True)
        self.assertIn("moving_load_analysis", seq)

    def test_moving_load_not_in_sequence(self):
        """moving_load_analysis should NOT be in sequence when disabled."""
        seq = build_analysis_sequence(has_moving_load=False)
        self.assertNotIn("moving_load_analysis", seq)

    def test_moving_load_after_gravity(self):
        """moving_load_analysis should come after gravity_analysis."""
        seq = build_analysis_sequence(has_moving_load=True)
        gi = seq.index("gravity_analysis")
        mi = seq.index("moving_load_analysis")
        self.assertGreater(mi, gi)


class TestHL93VehicleConstants(unittest.TestCase):
    """Tests for HL-93 vehicle data correctness in generated code."""

    def _get_script(self):
        els = [{"tag": 30000, "type": "elasticBeamColumn",
                "_component": "superstructure", "nodes": [30100, 30101]}]
        nodes = [{"tag": 30100, "_component": "superstructure"},
                 {"tag": 30101, "_component": "superstructure"}]
        seq = build_analysis_sequence(has_moving_load=True)
        return _generate_analysis_script(seq, els, nodes)

    def test_truck_front_axle_8k(self):
        """Front axle = 8 kip."""
        script = self._get_script()
        self.assertIn("8.0 * (1 + _IM)", script)

    def test_truck_middle_rear_32k(self):
        """Middle and rear axles = 32 kip each."""
        script = self._get_script()
        # Should have 32.0 twice (middle + rear)
        count = script.count("32.0 * (1 + _IM)")
        self.assertEqual(count, 2, "Should have two 32k axles")

    def test_truck_14ft_spacing(self):
        """Front-to-middle = 14ft = 168in."""
        script = self._get_script()
        self.assertIn("14.0 * 12.0", script)

    def test_truck_28ft_rear(self):
        """Front-to-rear = 28ft (14+14 min spacing)."""
        script = self._get_script()
        self.assertIn("28.0 * 12.0", script)

    def test_tandem_25k_axles(self):
        """Two 25k axles."""
        script = self._get_script()
        count = script.count("25.0 * (1 + _IM)")
        self.assertEqual(count, 2, "Should have two 25k axles")

    def test_tandem_4ft_spacing(self):
        """4ft apart = 48in."""
        script = self._get_script()
        self.assertIn("4.0 * 12.0", script)

    def test_lane_load_064_klf(self):
        """Lane load = 0.64 klf."""
        script = self._get_script()
        self.assertIn("0.64", script)


class TestMovingLoadAnalysisFlow(unittest.TestCase):
    """Tests for the analysis flow logic in generated code."""

    def _get_script(self, n_elements=5):
        els = []
        for i in range(n_elements):
            els.append({
                "tag": 30000 + i,
                "type": "elasticBeamColumn",
                "_component": "superstructure",
                "nodes": [30100 + i, 30100 + i + 1],
            })
        nodes = [{"tag": 30100 + i, "_component": "superstructure"}
                 for i in range(n_elements + 1)]
        seq = build_analysis_sequence(has_moving_load=True)
        return _generate_analysis_script(seq, els, nodes)

    def test_wipe_analysis_per_position(self):
        """Each position should wipe analysis first."""
        script = self._get_script()
        self.assertIn("ops.wipeAnalysis()", script)

    def test_static_analysis(self):
        """Should use static analysis."""
        script = self._get_script()
        self.assertIn("ops.analysis('Static')", script)

    def test_umfpack_solver(self):
        """Should use UmfPack solver."""
        script = self._get_script()
        self.assertIn("ops.system('UmfPack')", script)

    def test_load_control_integrator(self):
        """Should use LoadControl integrator."""
        script = self._get_script()
        self.assertIn("ops.integrator('LoadControl', 1.0)", script)

    def test_position_count_printed(self):
        """Should print position count when done."""
        script = self._get_script()
        self.assertIn("positions evaluated", script)

    def test_skip_message_on_no_gravity(self):
        """Should print skip message when gravity doesn't converge."""
        script = self._get_script()
        self.assertIn("Skipping moving load", script)

    def test_nearest_node_function(self):
        """Generated code should have nearest node helper."""
        script = self._get_script()
        self.assertIn("def _ml_nearest_node", script)

    def test_ele_load_for_lane(self):
        """Lane load applied via eleLoad."""
        script = self._get_script()
        self.assertIn("ops.eleLoad('-ele'", script)
        self.assertIn("-_lane_w", script)

    def test_node_load_for_axles(self):
        """Axle loads applied via ops.load at nodes."""
        script = self._get_script()
        self.assertIn("ops.load(_nn", script)


class TestExtractionMergeCode(unittest.TestCase):
    """Tests for the extraction code that merges ML results into envelopes."""

    def _get_extraction_code(self):
        """Get the extraction code template from the assembler source."""
        import inspect
        from nlb.tools import assembler
        source = inspect.getsource(assembler.run_analysis)
        return source

    def test_ml_results_merged_to_results(self):
        """Extraction code should merge _ml_results into results."""
        src = self._get_extraction_code()
        self.assertIn("moving_load_envelopes", src)

    def test_try_except_nameError(self):
        """Should handle NameError if _ml_results not defined."""
        src = self._get_extraction_code()
        # The extraction template should handle the case gracefully
        self.assertIn("NameError", src)


if __name__ == "__main__":
    unittest.main()

# ==============================================================================
# Project Prometheus: Tests for Delphi Oracle
# Version 0.1
#
# Description:
# Unit tests for the Delphi Oracle components.
# ==============================================================================

import unittest
from unittest.mock import MagicMock, patch
from src.delphi_oracle.delphi import Delphi
from mnemosyne_core import Mnemosyne
from aegis_layer import Aegis

class TestDelphi(unittest.TestCase):
    def setUp(self):
        self.mnemosyne = MagicMock(spec=Mnemosyne)
        self.aegis = MagicMock(spec=Aegis)
        self.delphi = Delphi(self.mnemosyne, self.aegis)

    def test_prediction_core_consequence_analysis(self):
        result = self.delphi.prediction_core.consequence_analysis("test action")
        self.assertIn("average_risk", result)
        self.assertIn("samples", result)
        self.assertEqual(len(result["samples"]), 100)

    def test_prediction_core_forecasting(self):
        data = [1, 2, 3, 4, 5]
        result = self.delphi.prediction_core.forecasting(data)
        self.assertIn("forecast", result)
        self.assertEqual(result["forecast"], 6.0)

    def test_prediction_core_hypothesis_testing(self):
        data = [1, 2, 3]
        result = self.delphi.prediction_core.hypothesis_testing("test hypothesis", data)
        self.assertIn("mean", result)
        self.assertTrue(result["passes"])

    def test_data_integrator_fetch_and_merge(self):
        sources = {
            'mnemosyne': 'test query',
            'antenor': 'test search'
        }
        self.mnemosyne.retrieve.return_value = "memory data"
        with patch('src.delphi_oracle.delphi.search_web') as mock_search:
            mock_search.return_value = [{'href': 'http://test.com'}]
            with patch('src.delphi_oracle.delphi.scrape_url') as mock_scrape:
                mock_scrape.return_value = "web content"
                result = self.delphi.data_integrator.fetch_and_merge(sources)
                self.assertIn('mnemosyne', result)
                self.assertIn('antenor', result)

    def test_risk_assessor_validate_prediction(self):
        self.aegis.validate_prediction.return_value = True
        result = self.delphi.risk_assessor.validate_prediction("safe prediction")
        self.assertTrue(result)

    def test_predict(self):
        request = {
            'type': 'forecasting',
            'sources': {},
            'data': [1, 2, 3]
        }
        self.delphi.risk_assessor.validate_prediction.return_value = True
        result = self.delphi.predict(request)
        self.assertIn("forecast", result)

if __name__ == '__main__':
    unittest.main()
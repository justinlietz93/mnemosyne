import unittest
from unittest.mock import patch, MagicMock
from kernel import PrometheusKernel
from mnemosyne_core import Mnemosyne
from src.delphi_oracle.delphi import Delphi
import ollama

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.mnemosyne = MagicMock()
        self.mnemosyne.collection = MagicMock()
        self.mnemosyne.collection.query.return_value = {'documents': [[]], 'distances': [[]], 'ids': [[]], 'metadatas': [[]]}
        self.mnemosyne.model = 'dummy_model'
        self.mnemosyne.query_prefix = ''
        self.kernel = PrometheusKernel(self.mnemosyne, 'dummy', 'dummy', aegis_enabled=True, clean_start=True)

    def test_process_prompt_integration(self):
        query = "Test query"
        with patch('kernel.ollama.chat') as mock_chat, patch('kernel.ollama.embeddings') as mock_embed, patch('kernel.search_web') as mock_search, patch('kernel.scrape_url') as mock_scrape, patch('aegis_layer.Aegis.validate_response') as mock_validate:
            mock_chat.return_value = iter([{'message': {'content': 'Test '}}, {'message': {'content': 'response'}}])
            mock_embed.return_value = {'embedding': [0.1] * 768}
            mock_search.return_value = [{'href': 'http://test.com'}]
            mock_scrape.return_value = "test content"
            mock_predict = MagicMock(return_value={'average_risk': 0.5, 'samples': []})
            self.kernel.delphi.predict = mock_predict
            mock_validate.return_value = True
            self.kernel.process_prompt(query)
            mock_predict.assert_called_once()
            mock_validate.assert_called_once()

    def test_safety_check_blocks(self):
        query = "Test query"
        with patch('kernel.ollama.chat') as mock_chat, patch('kernel.ollama.embeddings') as mock_embed, patch('kernel.search_web') as mock_search, patch('kernel.scrape_url') as mock_scrape, patch('aegis_layer.Aegis.validate_response') as mock_validate:
            mock_chat.return_value = iter([{'message': {'content': 'Unsafe '}}, {'message': {'content': 'response'}}])
            mock_embed.return_value = {'embedding': [0.1] * 768}
            mock_search.return_value = [{'href': 'http://test.com'}]
            mock_scrape.return_value = "test content"
            mock_predict = MagicMock(return_value={'average_risk': 0.5, 'samples': []})
            self.kernel.delphi.predict = mock_predict
            mock_validate.return_value = False
            self.kernel.process_prompt(query)
            mock_predict.assert_called_once()
            mock_validate.assert_called_once()

    def test_scalability_parallel_simulations(self):
        from src.delphi_oracle.delphi import Delphi
        with patch('src.delphi_oracle.delphi.torch'), patch('src.delphi_oracle.delphi.pyro'), patch('src.delphi_oracle.delphi.dist'), patch('src.delphi_oracle.delphi.SVI'), patch('src.delphi_oracle.delphi.Trace_ELBO'):
            delphi = Delphi(MagicMock(), MagicMock())
            def dummy_model():
                return {"risk": 0.5}
            results = delphi.simulation_engine.run_parallel_simulations(dummy_model, num_simulations=100)
            self.assertEqual(len(results), 100)

if __name__ == '__main__':
    unittest.main()
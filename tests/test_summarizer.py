"""
Tests for the abstractive text summarization module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from summarizer import AbstractiveSummarizer, SummarizationConfig, SummarizationResult, create_sample_dataset
from config import ConfigManager, AppConfig, ModelConfig, UIConfig, LoggingConfig


class TestSummarizationConfig:
    """Test SummarizationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SummarizationConfig()
        assert config.model_name == "facebook/bart-large-cnn"
        assert config.max_length == 50
        assert config.min_length == 25
        assert config.do_sample is False
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.device == "auto"
        assert config.batch_size == 1
        assert config.use_fast_tokenizer is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SummarizationConfig(
            model_name="t5-small",
            max_length=100,
            min_length=50,
            do_sample=True,
            temperature=0.8
        )
        assert config.model_name == "t5-small"
        assert config.max_length == 100
        assert config.min_length == 50
        assert config.do_sample is True
        assert config.temperature == 0.8


class TestSummarizationResult:
    """Test SummarizationResult class."""
    
    def test_result_creation(self):
        """Test SummarizationResult creation."""
        result = SummarizationResult(
            summary="This is a test summary.",
            original_length=100,
            summary_length=20,
            compression_ratio=0.2,
            model_used="test-model",
            processing_time=1.5
        )
        
        assert result.summary == "This is a test summary."
        assert result.original_length == 100
        assert result.summary_length == 20
        assert result.compression_ratio == 0.2
        assert result.model_used == "test-model"
        assert result.processing_time == 1.5


class TestAbstractiveSummarizer:
    """Test AbstractiveSummarizer class."""
    
    @patch('summarizer.AutoTokenizer')
    @patch('summarizer.AutoModelForSeq2SeqLM')
    @patch('summarizer.pipeline')
    def test_initialization(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test summarizer initialization."""
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_pipeline_instance = Mock()
        
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance
        
        config = SummarizationConfig()
        summarizer = AbstractiveSummarizer(config)
        
        assert summarizer.config == config
        assert summarizer._tokenizer == mock_tokenizer_instance
        assert summarizer._model == mock_model_instance
        assert summarizer._pipeline == mock_pipeline_instance
    
    @patch('summarizer.AutoTokenizer')
    @patch('summarizer.AutoModelForSeq2SeqLM')
    @patch('summarizer.pipeline')
    def test_get_device(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test device detection."""
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        config = SummarizationConfig(device="cpu")
        summarizer = AbstractiveSummarizer(config)
        
        assert summarizer.device == "cpu"
    
    @patch('summarizer.AutoTokenizer')
    @patch('summarizer.AutoModelForSeq2SeqLM')
    @patch('summarizer.pipeline')
    def test_summarize(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test text summarization."""
        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'summary_text': 'Test summary'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        config = SummarizationConfig()
        summarizer = AbstractiveSummarizer(config)
        
        text = "This is a test text for summarization."
        result = summarizer.summarize(text)
        
        assert isinstance(result, SummarizationResult)
        assert result.summary == "Test summary"
        assert result.original_length > 0
        assert result.summary_length > 0
        assert result.compression_ratio > 0
        assert result.model_used == config.model_name
        assert result.processing_time >= 0
    
    @patch('summarizer.AutoTokenizer')
    @patch('summarizer.AutoModelForSeq2SeqLM')
    @patch('summarizer.pipeline')
    def test_summarize_batch(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test batch summarization."""
        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'summary_text': 'Test summary'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        config = SummarizationConfig()
        summarizer = AbstractiveSummarizer(config)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        results = summarizer.summarize_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SummarizationResult)
            assert result.summary == "Test summary"
    
    @patch('summarizer.AutoTokenizer')
    @patch('summarizer.AutoModelForSeq2SeqLM')
    @patch('summarizer.pipeline')
    def test_get_model_info(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test model info retrieval."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50000
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_model.return_value = mock_model_instance
        
        mock_pipeline.return_value = Mock()
        
        config = SummarizationConfig()
        summarizer = AbstractiveSummarizer(config)
        
        info = summarizer.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "num_parameters" in info
        assert "vocab_size" in info


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config_manager = ConfigManager()
        config = config_manager._get_default_config()
        
        assert isinstance(config, AppConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config_manager = ConfigManager()
        config = config_manager._get_default_config()
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "ui" in config_dict
        assert "logging" in config_dict
        
        # Test deserialization
        new_config = AppConfig.from_dict(config_dict)
        assert isinstance(new_config, AppConfig)
        assert new_config.model.name == config.model.name
    
    def test_config_file_operations(self):
        """Test configuration file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_manager = ConfigManager(config_path)
            
            # Test saving config
            config = config_manager._get_default_config()
            config_manager.save_config(config)
            
            assert config_path.exists()
            
            # Test loading config
            loaded_config = config_manager.load_config()
            assert isinstance(loaded_config, AppConfig)
            assert loaded_config.model.name == config.model.name


class TestSampleDataset:
    """Test sample dataset creation."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        dataset = create_sample_dataset()
        
        assert len(dataset) > 0
        assert "text" in dataset.column_names
        assert "reference_summary" in dataset.column_names
        
        # Check that all texts and summaries are non-empty
        for i in range(len(dataset)):
            assert len(dataset[i]["text"].strip()) > 0
            assert len(dataset[i]["reference_summary"].strip()) > 0


class TestIntegration:
    """Integration tests."""
    
    @patch('summarizer.AutoTokenizer')
    @patch('summarizer.AutoModelForSeq2SeqLM')
    @patch('summarizer.pipeline')
    def test_end_to_end_workflow(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test end-to-end workflow."""
        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'summary_text': 'AI is machine intelligence.'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Create summarizer
        config = SummarizationConfig()
        summarizer = AbstractiveSummarizer(config)
        
        # Test summarization
        text = "Artificial intelligence is a field of computer science."
        result = summarizer.summarize(text)
        
        assert result.summary == "AI is machine intelligence."
        assert result.original_length > 0
        assert result.summary_length > 0
        
        # Test model info
        info = summarizer.get_model_info()
        assert info["model_name"] == config.model_name


if __name__ == "__main__":
    pytest.main([__file__])

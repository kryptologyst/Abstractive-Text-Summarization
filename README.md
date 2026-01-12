# Abstractive Text Summarization

A production-ready implementation of abstractive text summarization using state-of-the-art transformer models from Hugging Face. This project provides both a web interface and command-line tools for generating concise summaries of text documents.

## Features

- **Multiple Model Support**: Compatible with BART, T5, and other transformer-based summarization models
- **Web Interface**: Beautiful Streamlit-based web application for interactive summarization
- **Command-Line Interface**: Powerful CLI for batch processing and automation
- **Evaluation Metrics**: Built-in ROUGE and BLEU evaluation capabilities
- **Batch Processing**: Efficient processing of multiple documents
- **Type Safety**: Full type hints and modern Python practices
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: Full test suite with pytest
- **Sample Dataset**: Included sample data for testing and demonstration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA support (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Abstractive-Text-Summarization.git
   cd Abstractive-Text-Summarization
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create default configuration**:
   ```bash
   python src/config.py
   ```

## Quick Start

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` to access the interactive interface.

### Command Line Interface

Summarize text from command line:

```bash
python cli.py --text "Your text to summarize here"
```

Summarize text from file:

```bash
python cli.py --file input.txt --output summary.json
```

Batch process multiple files:

```bash
python cli.py --batch file1.txt file2.txt file3.txt --output-dir results/
```

### Python API

```python
from src.summarizer import AbstractiveSummarizer, SummarizationConfig

# Initialize summarizer
config = SummarizationConfig(
    model_name="facebook/bart-large-cnn",
    max_length=50,
    min_length=25
)
summarizer = AbstractiveSummarizer(config)

# Summarize text
text = "Your text here..."
result = summarizer.summarize(text)
print(f"Summary: {result.summary}")
print(f"Compression ratio: {result.compression_ratio:.2f}")
```

## 📁 Project Structure

```
abstractive-text-summarization/
├── src/                    # Core source code
│   ├── summarizer.py      # Main summarization module
│   └── config.py          # Configuration management
├── web_app/               # Streamlit web interface
│   └── app.py            # Main web application
├── tests/                 # Test suite
│   └── test_summarizer.py # Unit tests
├── config/                # Configuration files
│   └── config.yaml       # Default configuration
├── data/                  # Sample data and datasets
├── models/                # Saved models (optional)
├── cli.py                # Command-line interface
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Configuration

The application uses YAML configuration files for easy customization. Key configuration options:

### Model Configuration
- `model_name`: Hugging Face model identifier
- `max_length`: Maximum summary length
- `min_length`: Minimum summary length
- `do_sample`: Enable sampling for generation
- `temperature`: Sampling temperature
- `device`: Device to use (auto, cpu, cuda, mps)

### UI Configuration
- `title`: Web application title
- `max_text_length`: Maximum input text length
- `show_evaluation_metrics`: Enable/disable metrics display
- `enable_batch_processing`: Enable/disable batch processing

Example configuration file (`config/config.yaml`):

```yaml
model:
  name: "facebook/bart-large-cnn"
  max_length: 50
  min_length: 25
  do_sample: false
  temperature: 1.0
  device: "auto"

ui:
  title: "Abstractive Text Summarization"
  max_text_length: 5000
  show_evaluation_metrics: true
  enable_batch_processing: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 🔧 Available Models

The system supports various pre-trained models from Hugging Face:

- **facebook/bart-large-cnn**: High-quality CNN/DailyMail trained model
- **facebook/bart-large-xsum**: BBC XSum trained model
- **t5-small**: Smaller T5 model for faster inference
- **t5-base**: Balanced T5 model
- **t5-large**: Larger T5 model for better quality
- **google/pegasus-cnn_dailymail**: Google's Pegasus model

## Evaluation Metrics

The system provides comprehensive evaluation using standard metrics:

- **ROUGE-1**: Unigram overlap between summary and reference
- **ROUGE-2**: Bigram overlap between summary and reference  
- **ROUGE-L**: Longest common subsequence
- **BLEU**: Bilingual Evaluation Understudy score

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_summarizer.py
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled PyTorch for faster inference
2. **Model Selection**: Choose smaller models (t5-small) for faster processing
3. **Batch Processing**: Process multiple texts together for efficiency
4. **Text Length**: Shorter input texts generally produce better summaries

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Issues**: Check internet connection and Hugging Face access
3. **Import Errors**: Ensure all dependencies are installed correctly

### Debug Mode

Enable debug logging:

```bash
python cli.py --text "Your text" --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- The open-source community for the various models and tools used

## References

- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- [BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)

## Future Enhancements

- [ ] Support for more languages
- [ ] Fine-tuning capabilities
- [ ] Integration with more evaluation metrics
- [ ] Docker containerization
- [ ] API endpoint for external services
- [ ] Real-time summarization of streaming text
- [ ] Multi-document summarization
- [ ] Custom model training pipeline
# Abstractive-Text-Summarization

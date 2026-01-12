"""
Abstractive Text Summarization Module

This module provides a modern, type-safe implementation of abstractive text summarization
using state-of-the-art transformer models from Hugging Face.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Pipeline,
    PreTrainedTokenizer,
    PreTrainedModel
)
from datasets import Dataset
import evaluate
import numpy as np


@dataclass
class SummarizationConfig:
    """Configuration class for summarization parameters."""
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 50
    min_length: int = 25
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    device: str = "auto"
    batch_size: int = 1
    use_fast_tokenizer: bool = True


@dataclass
class SummarizationResult:
    """Result container for summarization output."""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    model_used: str
    processing_time: float


class AbstractiveSummarizer:
    """
    Modern abstractive text summarizer with support for multiple models and configurations.
    
    This class provides a clean interface for abstractive summarization using various
    transformer models from Hugging Face, with built-in evaluation capabilities.
    """
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize the summarizer with given configuration.
        
        Args:
            config: SummarizationConfig object with model and generation parameters.
                   If None, uses default configuration.
        """
        self.config = config or SummarizationConfig()
        self.logger = logging.getLogger(__name__)
        self._pipeline: Optional[Pipeline] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._model: Optional[PreTrainedModel] = None
        
        # Initialize device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize evaluation metrics
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("bleu")
    
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=self.config.use_fast_tokenizer
            )
            
            # Load model
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move model to device
            self._model = self._model.to(self.device)
            
            # Create pipeline
            self._pipeline = pipeline(
                "summarization",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def summarize(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None
    ) -> SummarizationResult:
        """
        Summarize a single text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary (overrides config)
            min_length: Minimum length of summary (overrides config)
            do_sample: Whether to use sampling (overrides config)
            
        Returns:
            SummarizationResult object with summary and metadata
        """
        import time
        start_time = time.time()
        
        # Use provided parameters or fall back to config
        max_len = max_length or self.config.max_length
        min_len = min_length or self.config.min_length
        sample = do_sample if do_sample is not None else self.config.do_sample
        
        try:
            # Generate summary
            result = self._pipeline(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            summary = result[0]['summary_text']
            processing_time = time.time() - start_time
            
            return SummarizationResult(
                summary=summary,
                original_length=len(text.split()),
                summary_length=len(summary.split()),
                compression_ratio=len(summary.split()) / len(text.split()),
                model_used=self.config.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}")
            raise
    
    def summarize_batch(
        self, 
        texts: List[str],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> List[SummarizationResult]:
        """
        Summarize multiple texts in batch.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length of summaries
            min_length: Minimum length of summaries
            
        Returns:
            List of SummarizationResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            self.logger.info(f"Processing text {i+1}/{len(texts)}")
            try:
                result = self.summarize(text, max_length, min_length)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing text {i+1}: {e}")
                # Create error result
                results.append(SummarizationResult(
                    summary="",
                    original_length=len(text.split()),
                    summary_length=0,
                    compression_ratio=0.0,
                    model_used=self.config.model_name,
                    processing_time=0.0
                ))
        
        return results
    
    def evaluate_summary(
        self, 
        reference: str, 
        summary: str
    ) -> Dict[str, float]:
        """
        Evaluate summary quality using ROUGE and BLEU metrics.
        
        Args:
            reference: Reference summary
            summary: Generated summary
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # ROUGE evaluation
            rouge_scores = self.rouge_metric.compute(
                predictions=[summary],
                references=[reference]
            )
            
            # BLEU evaluation
            bleu_scores = self.bleu_metric.compute(
                predictions=[summary],
                references=[reference]
            )
            
            return {
                "rouge-1": rouge_scores["rouge1"],
                "rouge-2": rouge_scores["rouge2"],
                "rouge-l": rouge_scores["rougeL"],
                "bleu": bleu_scores["bleu"]
            }
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model."""
        if self._model is None:
            return {}
        
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self._model.parameters()),
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0
        }
    
    def change_model(self, model_name: str) -> None:
        """
        Change the model to a different one.
        
        Args:
            model_name: Name of the new model to load
        """
        self.config.model_name = model_name
        self._load_model()
        self.logger.info(f"Changed model to: {model_name}")


def create_sample_dataset() -> Dataset:
    """
    Create a sample dataset for testing and demonstration.
    
    Returns:
        Hugging Face Dataset with sample texts and summaries
    """
    sample_data = {
        "text": [
            """
            Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
            """,
            """
            Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.
            """,
            """
            Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them.
            """,
            """
            Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, graph neural networks, have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs.
            """,
            """
            Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information.
            """
        ],
        "reference_summary": [
            "AI is machine intelligence that perceives environments and takes actions to achieve goals, mimicking human cognitive functions like learning and problem-solving.",
            "Machine learning is an AI subset that enables systems to automatically learn and improve from experience without explicit programming, focusing on pattern recognition in data.",
            "NLP is a field combining linguistics, computer science, and AI to enable computers to process, analyze, and understand human language and its contextual nuances.",
            "Deep learning uses artificial neural networks for representation learning, applied across fields like computer vision, speech recognition, NLP, and medical analysis.",
            "Computer vision is an interdisciplinary field enabling computers to gain high-level understanding from digital images and videos, automating human visual system tasks."
        ]
    }
    
    return Dataset.from_dict(sample_data)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize summarizer
    config = SummarizationConfig(
        model_name="facebook/bart-large-cnn",
        max_length=50,
        min_length=25
    )
    
    summarizer = AbstractiveSummarizer(config)
    
    # Sample text
    sample_text = """
    Abstractive text summarization is an NLP task where the goal is to generate a summary that is a paraphrased version of the original text. 
    This is different from extractive summarization, where the summary is made up of direct excerpts from the original text. 
    Abstractive summarization models, like BART and T5, rely on transformer architectures to understand the context and generate coherent summaries. 
    These models are trained on large datasets and can summarize long pieces of text into concise, coherent summaries that capture the key ideas.
    """
    
    # Generate summary
    result = summarizer.summarize(sample_text)
    
    print(f"Original text length: {result.original_length} words")
    print(f"Summary length: {result.summary_length} words")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Summary: {result.summary}")

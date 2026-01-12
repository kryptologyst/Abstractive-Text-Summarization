"""
Streamlit web interface for abstractive text summarization.
"""

import streamlit as st
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from summarizer import AbstractiveSummarizer, SummarizationConfig, SummarizationResult, create_sample_dataset
from config import ConfigManager, AppConfig


def setup_logging(config: AppConfig) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        filename=config.logging.file_path if config.logging.file_path else None
    )


def display_model_info(summarizer: AbstractiveSummarizer) -> None:
    """Display model information in sidebar."""
    with st.sidebar:
        st.header("Model Information")
        model_info = summarizer.get_model_info()
        
        st.metric("Model", model_info.get("model_name", "Unknown"))
        st.metric("Device", model_info.get("device", "Unknown"))
        st.metric("Parameters", f"{model_info.get('num_parameters', 0):,}")
        st.metric("Vocabulary Size", f"{model_info.get('vocab_size', 0):,}")


def display_evaluation_metrics(result: SummarizationResult, reference: str = None) -> None:
    """Display evaluation metrics if reference summary is provided."""
    if reference and reference.strip():
        try:
            metrics = summarizer.evaluate_summary(reference, result.summary)
            
            st.subheader("Evaluation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ROUGE-1", f"{metrics.get('rouge-1', 0):.3f}")
            with col2:
                st.metric("ROUGE-2", f"{metrics.get('rouge-2', 0):.3f}")
            with col3:
                st.metric("ROUGE-L", f"{metrics.get('rouge-l', 0):.3f}")
            with col4:
                st.metric("BLEU", f"{metrics.get('bleu', 0):.3f}")
                
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")


def main():
    """Main Streamlit application."""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Setup logging
    setup_logging(config)
    
    # Page configuration
    st.set_page_config(
        page_title=config.ui.title,
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title(config.ui.title)
    st.markdown(config.ui.description)
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        with st.spinner("Loading model..."):
            summarizer_config = SummarizationConfig(
                model_name=config.model.name,
                max_length=config.model.max_length,
                min_length=config.model.min_length,
                do_sample=config.model.do_sample,
                temperature=config.model.temperature,
                top_p=config.model.top_p,
                device=config.model.device
            )
            st.session_state.summarizer = AbstractiveSummarizer(summarizer_config)
    
    summarizer = st.session_state.summarizer
    
    # Display model info
    display_model_info(summarizer)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Processing", "Sample Dataset"])
    
    with tab1:
        st.header("Single Text Summarization")
        
        # Text input
        text = st.text_area(
            "Enter text to summarize:",
            height=200,
            max_chars=config.ui.max_text_length,
            help=f"Maximum {config.ui.max_text_length} characters"
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.slider(
                "Maximum summary length",
                min_value=10,
                max_value=200,
                value=config.ui.default_max_summary_length,
                help="Maximum number of words in the summary"
            )
            
            min_length = st.slider(
                "Minimum summary length",
                min_value=5,
                max_value=50,
                value=config.ui.default_min_summary_length,
                help="Minimum number of words in the summary"
            )
        
        with col2:
            do_sample = st.checkbox(
                "Enable sampling",
                value=config.model.do_sample,
                help="Use sampling for more diverse outputs"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=config.model.temperature,
                step=0.1,
                help="Controls randomness in generation"
            )
        
        # Reference summary for evaluation
        reference = st.text_area(
            "Reference summary (optional, for evaluation):",
            height=100,
            help="Provide a reference summary to evaluate the generated summary"
        )
        
        # Summarize button
        if st.button("Generate Summary", type="primary"):
            if text.strip():
                with st.spinner("Generating summary..."):
                    try:
                        result = summarizer.summarize(
                            text,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=do_sample
                        )
                        
                        # Display results
                        st.subheader("Summary")
                        st.write(result.summary)
                        
                        # Display metadata
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Original Length", f"{result.original_length} words")
                        with col2:
                            st.metric("Summary Length", f"{result.summary_length} words")
                        with col3:
                            st.metric("Compression Ratio", f"{result.compression_ratio:.2f}")
                        with col4:
                            st.metric("Processing Time", f"{result.processing_time:.2f}s")
                        
                        # Display evaluation metrics if reference provided
                        if config.ui.show_evaluation_metrics:
                            display_evaluation_metrics(result, reference)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.warning("Please enter some text to summarize.")
    
    with tab2:
        st.header("Batch Processing")
        
        if config.ui.enable_batch_processing:
            # Batch text input
            batch_texts = st.text_area(
                "Enter multiple texts (one per line):",
                height=300,
                help="Enter each text on a separate line"
            )
            
            if st.button("Process Batch", type="primary"):
                if batch_texts.strip():
                    texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                    
                    if texts:
                        with st.spinner(f"Processing {len(texts)} texts..."):
                            try:
                                results = summarizer.summarize_batch(texts)
                                
                                # Display results
                                for i, result in enumerate(results):
                                    with st.expander(f"Text {i+1} Summary"):
                                        st.write("**Summary:**", result.summary)
                                        st.write(f"**Length:** {result.summary_length} words")
                                        st.write(f"**Compression:** {result.compression_ratio:.2f}")
                                        st.write(f"**Time:** {result.processing_time:.2f}s")
                                
                                # Summary statistics
                                st.subheader("Batch Statistics")
                                avg_length = sum(r.summary_length for r in results) / len(results)
                                avg_compression = sum(r.compression_ratio for r in results) / len(results)
                                total_time = sum(r.processing_time for r in results)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Texts", len(results))
                                with col2:
                                    st.metric("Avg Summary Length", f"{avg_length:.1f} words")
                                with col3:
                                    st.metric("Avg Compression", f"{avg_compression:.2f}")
                                with col4:
                                    st.metric("Total Time", f"{total_time:.2f}s")
                                
                            except Exception as e:
                                st.error(f"Error processing batch: {e}")
                    else:
                        st.warning("Please enter at least one text.")
                else:
                    st.warning("Please enter some texts to process.")
        else:
            st.info("Batch processing is disabled in the current configuration.")
    
    with tab3:
        st.header("Sample Dataset")
        
        st.markdown("Try the summarizer with our sample dataset:")
        
        if st.button("Load Sample Dataset"):
            with st.spinner("Loading sample dataset..."):
                try:
                    dataset = create_sample_dataset()
                    
                    # Display dataset info
                    st.info(f"Loaded {len(dataset)} sample texts")
                    
                    # Let user select a text
                    selected_idx = st.selectbox(
                        "Select a text to summarize:",
                        range(len(dataset)),
                        format_func=lambda x: f"Text {x+1}"
                    )
                    
                    if selected_idx is not None:
                        text = dataset[selected_idx]["text"]
                        reference = dataset[selected_idx]["reference_summary"]
                        
                        # Display original text
                        st.subheader("Original Text")
                        st.write(text)
                        
                        # Generate summary
                        if st.button("Generate Summary for Selected Text"):
                            with st.spinner("Generating summary..."):
                                try:
                                    result = summarizer.summarize(text)
                                    
                                    # Display summary
                                    st.subheader("Generated Summary")
                                    st.write(result.summary)
                                    
                                    # Display reference
                                    st.subheader("Reference Summary")
                                    st.write(reference)
                                    
                                    # Display evaluation metrics
                                    if config.ui.show_evaluation_metrics:
                                        display_evaluation_metrics(result, reference)
                                    
                                except Exception as e:
                                    st.error(f"Error generating summary: {e}")
                
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers/)"
    )


if __name__ == "__main__":
    main()

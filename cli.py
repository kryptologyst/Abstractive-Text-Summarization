"""
Command-line interface for abstractive text summarization.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from summarizer import AbstractiveSummarizer, SummarizationConfig, SummarizationResult, create_sample_dataset
from config import ConfigManager, AppConfig


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def print_result(result: SummarizationResult, show_metrics: bool = False) -> None:
    """Print summarization result in a formatted way."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(result.summary)
    print("\n" + "-"*60)
    print("METADATA")
    print("-"*60)
    print(f"Original length: {result.original_length} words")
    print(f"Summary length: {result.summary_length} words")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Model used: {result.model_used}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    
    if show_metrics:
        print("\n" + "-"*60)
        print("EVALUATION METRICS")
        print("-"*60)
        # Note: CLI doesn't have reference summaries, so metrics would be empty
        print("Evaluation metrics require reference summaries (use web interface)")


def summarize_text_file(
    file_path: Path,
    config: SummarizationConfig,
    output_file: Optional[Path] = None,
    show_metrics: bool = False
) -> None:
    """Summarize text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            print(f"Error: File {file_path} is empty")
            return
        
        summarizer = AbstractiveSummarizer(config)
        result = summarizer.summarize(text)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': result.summary,
                    'original_length': result.original_length,
                    'summary_length': result.summary_length,
                    'compression_ratio': result.compression_ratio,
                    'model_used': result.model_used,
                    'processing_time': result.processing_time
                }, f, indent=2)
            print(f"Summary saved to: {output_file}")
        else:
            print_result(result, show_metrics)
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error processing file: {e}")


def summarize_text_input(
    text: str,
    config: SummarizationConfig,
    show_metrics: bool = False
) -> None:
    """Summarize text from command line input."""
    if not text.strip():
        print("Error: No text provided")
        return
    
    try:
        summarizer = AbstractiveSummarizer(config)
        result = summarizer.summarize(text)
        print_result(result, show_metrics)
    except Exception as e:
        print(f"Error processing text: {e}")


def batch_summarize(
    file_paths: List[Path],
    config: SummarizationConfig,
    output_dir: Optional[Path] = None,
    show_metrics: bool = False
) -> None:
    """Summarize multiple files in batch."""
    summarizer = AbstractiveSummarizer(config)
    results = []
    
    print(f"Processing {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\nProcessing file {i}/{len(file_paths)}: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                print(f"  Warning: File {file_path} is empty, skipping")
                continue
            
            result = summarizer.summarize(text)
            results.append((file_path, result))
            
            if output_dir:
                output_file = output_dir / f"{file_path.stem}_summary.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'file': str(file_path),
                        'summary': result.summary,
                        'original_length': result.original_length,
                        'summary_length': result.summary_length,
                        'compression_ratio': result.compression_ratio,
                        'model_used': result.model_used,
                        'processing_time': result.processing_time
                    }, f, indent=2)
                print(f"  Summary saved to: {output_file}")
            else:
                print(f"  Summary: {result.summary[:100]}{'...' if len(result.summary) > 100 else ''}")
                print(f"  Length: {result.summary_length} words, Compression: {result.compression_ratio:.2f}")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    # Print batch statistics
    if results:
        print("\n" + "="*60)
        print("BATCH STATISTICS")
        print("="*60)
        total_time = sum(r.processing_time for _, r in results)
        avg_length = sum(r.summary_length for _, r in results) / len(results)
        avg_compression = sum(r.compression_ratio for _, r in results) / len(results)
        
        print(f"Total files processed: {len(results)}")
        print(f"Average summary length: {avg_length:.1f} words")
        print(f"Average compression ratio: {avg_compression:.2f}")
        print(f"Total processing time: {total_time:.2f} seconds")


def demo_sample_dataset(config: SummarizationConfig) -> None:
    """Demonstrate summarization with sample dataset."""
    print("Loading sample dataset...")
    
    try:
        dataset = create_sample_dataset()
        summarizer = AbstractiveSummarizer(config)
        
        print(f"Loaded {len(dataset)} sample texts")
        print("\nDemonstrating summarization on sample texts:\n")
        
        for i in range(min(3, len(dataset))):  # Show first 3 examples
            text = dataset[i]["text"]
            reference = dataset[i]["reference_summary"]
            
            print(f"Example {i+1}:")
            print(f"Original text: {text[:200]}{'...' if len(text) > 200 else ''}")
            
            result = summarizer.summarize(text)
            
            print(f"Generated summary: {result.summary}")
            print(f"Reference summary: {reference}")
            print(f"Compression ratio: {result.compression_ratio:.2f}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error loading sample dataset: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Abstractive Text Summarization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize text from command line
  python cli.py --text "Your text here"
  
  # Summarize text from file
  python cli.py --file input.txt
  
  # Batch process multiple files
  python cli.py --batch file1.txt file2.txt file3.txt
  
  # Use different model
  python cli.py --text "Your text" --model "t5-small"
  
  # Save output to file
  python cli.py --file input.txt --output summary.json
  
  # Demo with sample dataset
  python cli.py --demo
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        help="Text to summarize"
    )
    input_group.add_argument(
        "--file", "-f",
        type=Path,
        help="File containing text to summarize"
    )
    input_group.add_argument(
        "--batch", "-b",
        nargs="+",
        type=Path,
        help="Multiple files to process in batch"
    )
    input_group.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demo with sample dataset"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        default="facebook/bart-large-cnn",
        help="Model to use for summarization (default: facebook/bart-large-cnn)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum length of summary (default: 50)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=25,
        help="Minimum length of summary (default: 25)"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda, mps) (default: auto)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing"
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show evaluation metrics (requires reference summaries)"
    )
    
    # Other options
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    app_config = config_manager.load_config()
    
    # Create summarization config
    config = SummarizationConfig(
        model_name=args.model,
        max_length=args.max_length,
        min_length=args.min_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        device=args.device
    )
    
    # Execute based on input type
    try:
        if args.text:
            summarize_text_input(args.text, config, args.show_metrics)
        elif args.file:
            summarize_text_file(args.file, config, args.output, args.show_metrics)
        elif args.batch:
            batch_summarize(args.batch, config, args.output_dir, args.show_metrics)
        elif args.demo:
            demo_sample_dataset(config)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

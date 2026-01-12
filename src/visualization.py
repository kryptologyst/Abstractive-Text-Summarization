"""
Visualization and explainability module for text summarization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from summarizer import SummarizationResult


class SummarizationVisualizer:
    """
    Visualization tools for summarization results and analysis.
    """
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_summary_length_distribution(
        self, 
        results: List[SummarizationResult],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot distribution of summary lengths.
        
        Args:
            results: List of SummarizationResult objects
            save_path: Optional path to save the plot
        """
        lengths = [r.summary_length for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Summary Length (words)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Summary Lengths')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(lengths, vert=True)
        ax2.set_ylabel('Summary Length (words)')
        ax2.set_title('Summary Length Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_compression_ratio_analysis(
        self, 
        results: List[SummarizationResult],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot compression ratio analysis.
        
        Args:
            results: List of SummarizationResult objects
            save_path: Optional path to save the plot
        """
        compression_ratios = [r.compression_ratio for r in results]
        original_lengths = [r.original_length for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot: Original length vs Compression ratio
        ax1.scatter(original_lengths, compression_ratios, alpha=0.6)
        ax1.set_xlabel('Original Text Length (words)')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('Original Length vs Compression Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(original_lengths, compression_ratios, 1)
        p = np.poly1d(z)
        ax1.plot(original_lengths, p(original_lengths), "r--", alpha=0.8)
        
        # Histogram of compression ratios
        ax2.hist(compression_ratios, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Compression Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Compression Ratios')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_processing_time_analysis(
        self, 
        results: List[SummarizationResult],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot processing time analysis.
        
        Args:
            results: List of SummarizationResult objects
            save_path: Optional path to save the plot
        """
        processing_times = [r.processing_time for r in results]
        text_lengths = [r.original_length for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot: Text length vs Processing time
        ax1.scatter(text_lengths, processing_times, alpha=0.6)
        ax1.set_xlabel('Text Length (words)')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Text Length vs Processing Time')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(text_lengths, processing_times, 1)
        p = np.poly1d(z)
        ax1.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)
        
        # Histogram of processing times
        ax2.hist(processing_times, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Processing Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Processing Times')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(
        self, 
        results: List[SummarizationResult],
        title: str = "Summarization Analysis Dashboard"
    ) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            results: List of SummarizationResult objects
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Prepare data
        df = pd.DataFrame([
            {
                'original_length': r.original_length,
                'summary_length': r.summary_length,
                'compression_ratio': r.compression_ratio,
                'processing_time': r.processing_time,
                'model': r.model_used
            }
            for r in results
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Summary Length Distribution',
                'Compression Ratio Analysis',
                'Processing Time vs Text Length',
                'Model Performance Overview'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Summary length histogram
        fig.add_trace(
            go.Histogram(x=df['summary_length'], name='Summary Length'),
            row=1, col=1
        )
        
        # Compression ratio scatter
        fig.add_trace(
            go.Scatter(
                x=df['original_length'],
                y=df['compression_ratio'],
                mode='markers',
                name='Compression Ratio',
                text=df['model'],
                hovertemplate='Original: %{x}<br>Compression: %{y}<br>Model: %{text}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Processing time scatter
        fig.add_trace(
            go.Scatter(
                x=df['original_length'],
                y=df['processing_time'],
                mode='markers',
                name='Processing Time',
                text=df['model'],
                hovertemplate='Length: %{x}<br>Time: %{y}s<br>Model: %{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Model performance bar chart
        model_stats = df.groupby('model').agg({
            'processing_time': 'mean',
            'compression_ratio': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=model_stats['model'],
                y=model_stats['processing_time'],
                name='Avg Processing Time',
                yaxis='y4'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Summary Length (words)", row=1, col=1)
        fig.update_xaxes(title_text="Original Length (words)", row=1, col=2)
        fig.update_xaxes(title_text="Original Length (words)", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Compression Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Processing Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Avg Processing Time (s)", row=2, col=2)
        
        return fig
    
    def generate_summary_report(
        self, 
        results: List[SummarizationResult],
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Args:
            results: List of SummarizationResult objects
            save_path: Optional path to save the report
            
        Returns:
            Dictionary containing report statistics
        """
        if not results:
            return {}
        
        # Calculate statistics
        stats = {
            'total_summaries': len(results),
            'avg_summary_length': np.mean([r.summary_length for r in results]),
            'avg_original_length': np.mean([r.original_length for r in results]),
            'avg_compression_ratio': np.mean([r.compression_ratio for r in results]),
            'avg_processing_time': np.mean([r.processing_time for r in results]),
            'min_summary_length': min([r.summary_length for r in results]),
            'max_summary_length': max([r.summary_length for r in results]),
            'min_compression_ratio': min([r.compression_ratio for r in results]),
            'max_compression_ratio': max([r.compression_ratio for r in results]),
            'total_processing_time': sum([r.processing_time for r in results]),
            'models_used': list(set([r.model_used for r in results]))
        }
        
        # Create report text
        report_text = f"""
# Summarization Analysis Report

## Overview
- **Total Summaries**: {stats['total_summaries']}
- **Models Used**: {', '.join(stats['models_used'])}
- **Total Processing Time**: {stats['total_processing_time']:.2f} seconds

## Summary Length Statistics
- **Average Length**: {stats['avg_summary_length']:.1f} words
- **Range**: {stats['min_summary_length']} - {stats['max_summary_length']} words

## Compression Analysis
- **Average Compression Ratio**: {stats['avg_compression_ratio']:.2f}
- **Compression Range**: {stats['min_compression_ratio']:.2f} - {stats['max_compression_ratio']:.2f}

## Performance Metrics
- **Average Processing Time**: {stats['avg_processing_time']:.2f} seconds
- **Average Original Length**: {stats['avg_original_length']:.1f} words

## Efficiency Analysis
- **Words per Second**: {stats['avg_original_length'] / stats['avg_processing_time']:.1f}
- **Compression Efficiency**: {stats['avg_compression_ratio']:.2f} (lower is better)
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return stats
    
    def plot_model_comparison(
        self, 
        results_by_model: Dict[str, List[SummarizationResult]],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Compare performance across different models.
        
        Args:
            results_by_model: Dictionary mapping model names to result lists
            save_path: Optional path to save the plot
        """
        models = list(results_by_model.keys())
        
        # Calculate metrics for each model
        metrics = {
            'avg_processing_time': [],
            'avg_compression_ratio': [],
            'avg_summary_length': [],
            'total_summaries': []
        }
        
        for model in models:
            results = results_by_model[model]
            metrics['avg_processing_time'].append(np.mean([r.processing_time for r in results]))
            metrics['avg_compression_ratio'].append(np.mean([r.compression_ratio for r in results]))
            metrics['avg_summary_length'].append(np.mean([r.summary_length for r in results]))
            metrics['total_summaries'].append(len(results))
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Processing time comparison
        axes[0, 0].bar(models, metrics['avg_processing_time'])
        axes[0, 0].set_title('Average Processing Time by Model')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Compression ratio comparison
        axes[0, 1].bar(models, metrics['avg_compression_ratio'])
        axes[0, 1].set_title('Average Compression Ratio by Model')
        axes[0, 1].set_ylabel('Compression Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Summary length comparison
        axes[1, 0].bar(models, metrics['avg_summary_length'])
        axes[1, 0].set_title('Average Summary Length by Model')
        axes[1, 0].set_ylabel('Length (words)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Total summaries comparison
        axes[1, 1].bar(models, metrics['total_summaries'])
        axes[1, 1].set_title('Total Summaries by Model')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_sample_visualization_data() -> List[SummarizationResult]:
    """
    Create sample data for visualization testing.
    
    Returns:
        List of SummarizationResult objects
    """
    from summarizer import SummarizationResult
    
    sample_results = [
        SummarizationResult(
            summary="AI is transforming industries through automation and intelligent decision-making.",
            original_length=150,
            summary_length=12,
            compression_ratio=0.08,
            model_used="facebook/bart-large-cnn",
            processing_time=2.3
        ),
        SummarizationResult(
            summary="Machine learning enables computers to learn from data without explicit programming.",
            original_length=200,
            summary_length=15,
            compression_ratio=0.075,
            model_used="facebook/bart-large-cnn",
            processing_time=2.8
        ),
        SummarizationResult(
            summary="Natural language processing helps computers understand and generate human language.",
            original_length=180,
            summary_length=13,
            compression_ratio=0.072,
            model_used="t5-small",
            processing_time=1.5
        ),
        SummarizationResult(
            summary="Deep learning uses neural networks to solve complex problems in various domains.",
            original_length=220,
            summary_length=16,
            compression_ratio=0.073,
            model_used="t5-small",
            processing_time=1.8
        ),
        SummarizationResult(
            summary="Computer vision enables machines to interpret and understand visual information.",
            original_length=190,
            summary_length=14,
            compression_ratio=0.074,
            model_used="facebook/bart-large-cnn",
            processing_time=2.1
        )
    ]
    
    return sample_results


if __name__ == "__main__":
    # Example usage
    visualizer = SummarizationVisualizer()
    sample_data = create_sample_visualization_data()
    
    # Generate visualizations
    visualizer.plot_summary_length_distribution(sample_data)
    visualizer.plot_compression_ratio_analysis(sample_data)
    visualizer.plot_processing_time_analysis(sample_data)
    
    # Generate report
    stats = visualizer.generate_summary_report(sample_data)
    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create interactive dashboard
    fig = visualizer.create_interactive_dashboard(sample_data)
    fig.show()

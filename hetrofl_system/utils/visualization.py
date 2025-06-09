import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PlotGenerator:
    """Generates various plots and visualizations for the federated learning system."""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_metrics_over_rounds(self, metrics_data: Dict[str, List], 
                                title: str = "Model Performance Over Rounds",
                                save_name: Optional[str] = None) -> go.Figure:
        """Create interactive plot showing metrics over rounds."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for metric, (row, col) in zip(metrics, positions):
                if metric in metrics_data:
                    rounds = list(range(1, len(metrics_data[metric]) + 1))
                    fig.add_trace(
                        go.Scatter(
                            x=rounds,
                            y=metrics_data[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title(),
                            line=dict(width=3),
                            marker=dict(size=8)
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=title,
                height=600,
                showlegend=False,
                template="plotly_white"
            )
            
            # Update axes
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(title_text="Round", row=i, col=j)
                    fig.update_yaxes(title_text="Score", row=i, col=j)
            
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Plot saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating metrics plot: {e}")
            return go.Figure()
    
    def plot_comparison_chart(self, global_metrics: Dict, local_metrics: Dict,
                             metric_name: str = 'accuracy',
                             save_name: Optional[str] = None) -> go.Figure:
        """Create comparison chart between global and local models."""
        try:
            fig = go.Figure()
            
            # Global model
            if metric_name in global_metrics:
                rounds = list(range(1, len(global_metrics[metric_name]) + 1))
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=global_metrics[metric_name],
                    mode='lines+markers',
                    name='Global Model',
                    line=dict(width=4, color='red'),
                    marker=dict(size=10)
                ))
            
            # Local models
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            for i, (client_name, client_metrics) in enumerate(local_metrics.items()):
                if metric_name in client_metrics:
                    rounds = list(range(1, len(client_metrics[metric_name]) + 1))
                    fig.add_trace(go.Scatter(
                        x=rounds,
                        y=client_metrics[metric_name],
                        mode='lines+markers',
                        name=f'Local: {client_name}',
                        line=dict(width=3, color=colors[i % len(colors)]),
                        marker=dict(size=8)
                    ))
            
            fig.update_layout(
                title=f'{metric_name.replace("_", " ").title()} Comparison: Global vs Local Models',
                xaxis_title='Round',
                yaxis_title=metric_name.replace('_', ' ').title(),
                height=500,
                template="plotly_white",
                legend=dict(x=0.02, y=0.98)
            )
            
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Comparison plot saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return go.Figure()
    
    def plot_improvement_percentages(self, improvement_data: Dict[str, Dict],
                                   save_name: Optional[str] = None) -> go.Figure:
        """Create bar chart showing improvement percentages."""
        try:
            models = []
            metrics = []
            improvements = []
            
            for model_name, model_improvements in improvement_data.items():
                for metric, improvement in model_improvements.items():
                    if 'improvement' in metric:
                        models.append(model_name)
                        metrics.append(metric.replace('_improvement', '').replace('_', ' ').title())
                        improvements.append(improvement)
            
            df = pd.DataFrame({
                'Model': models,
                'Metric': metrics,
                'Improvement (%)': improvements
            })
            
            fig = px.bar(df, x='Model', y='Improvement (%)', color='Metric',
                        title='Model Improvement Percentages',
                        barmode='group',
                        template="plotly_white")
            
            fig.update_layout(height=500)
            
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Improvement plot saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating improvement plot: {e}")
            return go.Figure()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str],
                             title: str = "Confusion Matrix",
                             save_name: Optional[str] = None) -> go.Figure:
        """Create interactive confusion matrix heatmap."""
        try:
            # Normalize confusion matrix
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_normalized,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=confusion_matrix,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=500,
                template="plotly_white"
            )
            
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Confusion matrix saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
            return go.Figure()
    
    def plot_training_progress(self, training_data: Dict[str, List],
                              save_name: Optional[str] = None) -> go.Figure:
        """Create real-time training progress plot."""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss Over Time', 'Training Time per Round')
            )
            
            if 'loss' in training_data:
                rounds = list(range(1, len(training_data['loss']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=rounds,
                        y=training_data['loss'],
                        mode='lines+markers',
                        name='Loss',
                        line=dict(color='red', width=3)
                    ),
                    row=1, col=1
                )
            
            if 'training_time' in training_data:
                rounds = list(range(1, len(training_data['training_time']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=rounds,
                        y=training_data['training_time'],
                        mode='lines+markers',
                        name='Training Time',
                        line=dict(color='blue', width=3)
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title='Training Progress',
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Round", row=1, col=1)
            fig.update_xaxes(title_text="Round", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
            
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Training progress plot saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating training progress plot: {e}")
            return go.Figure()
    
    def create_dashboard_summary(self, summary_stats: Dict[str, Any],
                                save_name: Optional[str] = None) -> go.Figure:
        """Create a summary dashboard with key metrics."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Global Model Performance', 'Local Models Comparison',
                              'Improvement Trends', 'System Statistics'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # Global model performance indicator
            if 'global' in summary_stats and 'accuracy' in summary_stats['global']:
                latest_accuracy = summary_stats['global']['accuracy']['latest']
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=latest_accuracy * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Global Accuracy (%)"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"},
                                        {'range': [80, 100], 'color': "lightgreen"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}
                    ),
                    row=1, col=1
                )
            
            # Local models comparison
            if 'local' in summary_stats:
                models = list(summary_stats['local'].keys())
                accuracies = [summary_stats['local'][model]['accuracy']['latest'] * 100 
                             for model in models if 'accuracy' in summary_stats['local'][model]]
                
                fig.add_trace(
                    go.Bar(x=models, y=accuracies, name='Local Models'),
                    row=1, col=2
                )
            
            # System statistics table
            if 'overall' in summary_stats:
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric', 'Value']),
                        cells=dict(values=[
                            ['Total Rounds', 'Total Clients', 'Last Updated'],
                            [summary_stats['overall']['total_rounds'],
                             summary_stats['overall']['total_clients'],
                             summary_stats['overall']['last_updated'][:19]]
                        ])
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='HETROFL System Dashboard',
                height=800,
                template="plotly_white"
            )
            
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Dashboard saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return go.Figure()
    
    def save_static_plot(self, fig, filename: str, format: str = 'png'):
        """Save plotly figure as static image."""
        try:
            filepath = os.path.join(self.save_dir, f"{filename}.{format}")
            fig.write_image(filepath)
            logger.info(f"Static plot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving static plot: {e}")
    
    def create_model_evaluation_plots(self, model_name: str, balanced_metrics: Dict[str, float], 
                                     imbalanced_metrics: Dict[str, float], 
                                     training_history: List[Dict[str, float]] = None,
                                     save_plots: bool = True) -> Dict[str, go.Figure]:
        """Create comprehensive evaluation plots for a specific model."""
        try:
            plots = {}
            
            # 1. Balanced vs Imbalanced Performance Comparison
            comparison_fig = self._create_balanced_vs_imbalanced_plot(
                model_name, balanced_metrics, imbalanced_metrics
            )
            plots['balanced_comparison'] = comparison_fig
            
            # 2. Training Progress Plot (if history available)
            if training_history:
                progress_fig = self._create_training_progress_plot(model_name, training_history)
                plots['training_progress'] = progress_fig
            
            # 3. Metrics Radar Chart
            radar_fig = self._create_metrics_radar_chart(model_name, balanced_metrics)
            plots['metrics_radar'] = radar_fig
            
            # 4. Performance Summary Dashboard
            dashboard_fig = self._create_model_dashboard(
                model_name, balanced_metrics, imbalanced_metrics, training_history
            )
            plots['dashboard'] = dashboard_fig
            
            # Save plots if requested
            if save_plots:
                self._save_model_plots(model_name, plots)
            
            logger.info(f"Created {len(plots)} evaluation plots for {model_name}")
            return plots
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots for {model_name}: {e}")
            return {}
    
    def _create_balanced_vs_imbalanced_plot(self, model_name: str, balanced_metrics: Dict[str, float], 
                                          imbalanced_metrics: Dict[str, float]) -> go.Figure:
        """Create comparison plot between balanced and imbalanced performance."""
        try:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            balanced_values = [balanced_metrics.get(metric, 0.0) for metric in metrics]
            imbalanced_values = [imbalanced_metrics.get(metric, 0.0) for metric in metrics]
            
            fig = go.Figure()
            
            # Add balanced performance
            fig.add_trace(go.Bar(
                name='Balanced Dataset',
                x=[m.replace('_', ' ').title() for m in metrics],
                y=balanced_values,
                marker_color='#2ecc71',
                text=[f'{v:.3f}' for v in balanced_values],
                textposition='auto'
            ))
            
            # Add imbalanced performance
            fig.add_trace(go.Bar(
                name='Imbalanced Dataset',
                x=[m.replace('_', ' ').title() for m in metrics],
                y=imbalanced_values,
                marker_color='#e74c3c',
                text=[f'{v:.3f}' for v in imbalanced_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f'{model_name} Performance: Balanced vs Imbalanced Data',
                xaxis_title='Metrics',
                yaxis_title='Score',
                barmode='group',
                template="plotly_white",
                height=500,
                legend=dict(x=0.02, y=0.98)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating balanced vs imbalanced plot: {e}")
            return go.Figure()
    
    def _create_training_progress_plot(self, model_name: str, training_history: List[Dict[str, float]]) -> go.Figure:
        """Create training progress plot showing improvement over rounds."""
        try:
            rounds = list(range(1, len(training_history) + 1))
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            for metric, (row, col), color in zip(metrics, positions, colors):
                values = [entry.get(metric, 0.0) for entry in training_history]
                
                fig.add_trace(
                    go.Scatter(
                        x=rounds,
                        y=values,
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=3, color=color),
                        marker=dict(size=8)
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title=f'{model_name} Training Progress Over Federated Rounds',
                height=600,
                showlegend=False,
                template="plotly_white"
            )
            
            # Update axes
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(title_text="Round", row=i, col=j)
                    fig.update_yaxes(title_text="Score", row=i, col=j)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating training progress plot: {e}")
            return go.Figure()
    
    def _create_metrics_radar_chart(self, model_name: str, metrics: Dict[str, float]) -> go.Figure:
        """Create radar chart showing all metrics for the model."""
        try:
            metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            metric_keys = ['accuracy', 'f1_score', 'precision', 'recall']
            values = [metrics.get(key, 0.0) for key in metric_keys]
            
            # Close the radar chart
            values.append(values[0])
            metric_names.append(metric_names[0])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names,
                fill='toself',
                name=model_name,
                line_color='#3498db',
                fillcolor='rgba(52, 152, 219, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title=f'{model_name} Performance Radar Chart',
                template="plotly_white",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return go.Figure()
    
    def _create_model_dashboard(self, model_name: str, balanced_metrics: Dict[str, float], 
                              imbalanced_metrics: Dict[str, float], 
                              training_history: List[Dict[str, float]] = None) -> go.Figure:
        """Create comprehensive dashboard for the model."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Performance Summary', 
                    'Balanced vs Imbalanced',
                    'Improvement Tracking',
                    'Model Statistics'
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "table"}]
                ]
            )
            
            # 1. Performance indicator (balanced accuracy)
            balanced_accuracy = balanced_metrics.get('accuracy', 0.0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=balanced_accuracy * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{model_name} Accuracy (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
            
            # 2. Balanced vs Imbalanced comparison
            metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            metric_keys = ['accuracy', 'f1_score', 'precision', 'recall']
            
            balanced_vals = [balanced_metrics.get(key, 0.0) for key in metric_keys]
            imbalanced_vals = [imbalanced_metrics.get(key, 0.0) for key in metric_keys]
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=balanced_vals,
                    name='Balanced',
                    marker_color='#2ecc71'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=imbalanced_vals,
                    name='Imbalanced',
                    marker_color='#e74c3c'
                ),
                row=1, col=2
            )
            
            # 3. Improvement tracking (if training history available)
            if training_history and len(training_history) > 1:
                rounds = list(range(1, len(training_history) + 1))
                accuracies = [entry.get('accuracy', 0.0) for entry in training_history]
                
                fig.add_trace(
                    go.Scatter(
                        x=rounds,
                        y=accuracies,
                        mode='lines+markers',
                        name='Training Progress',
                        line=dict(color='#3498db', width=3)
                    ),
                    row=2, col=1
                )
            
            # 4. Model statistics table
            stats_data = [
                ['Balanced Accuracy', f"{balanced_accuracy:.3f}"],
                ['Imbalanced Accuracy', f"{imbalanced_metrics.get('accuracy', 0.0):.3f}"],
                ['Balanced F1 Score', f"{balanced_metrics.get('f1_score', 0.0):.3f}"],
                ['Training Rounds', str(len(training_history) if training_history else 0)],
                ['Model Type', model_name.replace('_', ' ').title()]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                    cells=dict(values=list(zip(*stats_data)), fill_color='white')
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{model_name} Comprehensive Performance Dashboard',
                height=800,
                template="plotly_white",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model dashboard: {e}")
            return go.Figure()
    
    def _save_model_plots(self, model_name: str, plots: Dict[str, go.Figure]):
        """Save all plots for a specific model."""
        try:
            model_dir = os.path.join(self.save_dir, f"{model_name}_plots")
            os.makedirs(model_dir, exist_ok=True)
            
            for plot_name, fig in plots.items():
                # Save as HTML
                html_path = os.path.join(model_dir, f"{plot_name}.html")
                fig.write_html(html_path)
                
                # Save as PNG (if possible)
                try:
                    png_path = os.path.join(model_dir, f"{plot_name}.png")
                    fig.write_image(png_path, width=1200, height=800)
                except Exception as e:
                    logger.warning(f"Could not save PNG for {plot_name}: {e}")
            
            logger.info(f"Saved {len(plots)} plots for {model_name} in {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving plots for {model_name}: {e}")
    
    def create_federated_learning_summary(self, all_model_results: Dict[str, Dict], 
                                        save_name: str = "federated_learning_summary") -> go.Figure:
        """Create comprehensive summary of all models' federated learning performance."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Model Accuracy Comparison (Balanced Data)',
                    'Federated Learning Improvements',
                    'Performance Distribution',
                    'Training Efficiency'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "box"}, {"type": "scatter"}]
                ]
            )
            
            model_names = list(all_model_results.keys())
            
            # 1. Accuracy comparison on balanced data
            balanced_accuracies = [
                all_model_results[model].get('balanced_metrics', {}).get('accuracy', 0.0)
                for model in model_names
            ]
            
            fig.add_trace(
                go.Bar(
                    x=[name.replace('_', ' ').title() for name in model_names],
                    y=balanced_accuracies,
                    name='Balanced Accuracy',
                    marker_color='#2ecc71',
                    text=[f'{acc:.3f}' for acc in balanced_accuracies],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # 2. Federated learning improvements
            improvements = []
            for model in model_names:
                history = all_model_results[model].get('training_history', [])
                if len(history) > 1:
                    initial_acc = history[0].get('accuracy', 0.0)
                    final_acc = history[-1].get('accuracy', 0.0)
                    improvement = final_acc - initial_acc
                else:
                    improvement = 0.0
                improvements.append(improvement)
            
            fig.add_trace(
                go.Bar(
                    x=[name.replace('_', ' ').title() for name in model_names],
                    y=improvements,
                    name='Accuracy Improvement',
                    marker_color='#3498db',
                    text=[f'{imp:.3f}' for imp in improvements],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Performance distribution (box plot)
            for i, model in enumerate(model_names):
                history = all_model_results[model].get('training_history', [])
                accuracies = [entry.get('accuracy', 0.0) for entry in history]
                
                if accuracies:
                    fig.add_trace(
                        go.Box(
                            y=accuracies,
                            name=model.replace('_', ' ').title(),
                            boxpoints='all'
                        ),
                        row=2, col=1
                    )
            
            # 4. Training efficiency (rounds vs final accuracy)
            final_accuracies = []
            training_rounds = []
            
            for model in model_names:
                history = all_model_results[model].get('training_history', [])
                if history:
                    final_accuracies.append(history[-1].get('accuracy', 0.0))
                    training_rounds.append(len(history))
                else:
                    final_accuracies.append(0.0)
                    training_rounds.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=training_rounds,
                    y=final_accuracies,
                    mode='markers+text',
                    text=[name.replace('_', ' ').title() for name in model_names],
                    textposition='top center',
                    marker=dict(size=12, color='#e74c3c'),
                    name='Training Efficiency'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Federated Learning System Performance Summary',
                height=800,
                template="plotly_white",
                showlegend=True
            )
            
            # Save the summary
            if save_name:
                filepath = os.path.join(self.save_dir, f"{save_name}.html")
                fig.write_html(filepath)
                logger.info(f"Federated learning summary saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating federated learning summary: {e}")
            return go.Figure() 
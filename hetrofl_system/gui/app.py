from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetrofl_system.config import *
from hetrofl_system.models.federated_coordinator import FederatedCoordinator
from hetrofl_system.utils.data_loader import DataLoader
from hetrofl_system.utils.metrics import MetricsTracker
from hetrofl_system.utils.visualization import PlotGenerator
from hetrofl_system.utils.state_manager import SystemStateManager
from hetrofl_system.models.global_model import GlobalMLPModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sanitize_for_json(data):
    """Sanitize data to ensure safe JSON serialization."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data):
            return 0.0
        elif math.isinf(data):
            return 0.0
        return data
    elif isinstance(data, np.floating):
        if np.isnan(data):
            return 0.0
        elif np.isinf(data):
            return 0.0
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    return data

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hetrofl_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Add custom Jinja2 filters
@app.template_filter('strftime')
def _jinja2_filter_strftime(date, fmt=None):
    """Convert a datetime to a formatted string."""
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    try:
        return date.strftime(fmt)
    except:
        return str(date)

# Add context processor for common template functions
@app.context_processor
def utility_processor():
    def time():
        return datetime.now()
    return {'time': time}

# Global variables
coordinator: Optional[FederatedCoordinator] = None
metrics_tracker: Optional[MetricsTracker] = None
plot_generator: Optional[PlotGenerator] = None
data_loader: Optional[DataLoader] = None
state_manager: Optional[SystemStateManager] = None

def initialize_system():
    """Initialize the HETROFL system."""
    global coordinator, metrics_tracker, plot_generator, data_loader, state_manager
    
    try:
        logger.info("Initializing HETROFL system...")
        
        # Initialize state manager first
        state_manager = SystemStateManager(str(RESULTS_DIR))
        
        # Initialize data loader
        data_loader = DataLoader(
            dataset_path=MAIN_DATASET_PATH,
            target_column=TARGET_COLUMN,
            columns_to_drop=COLUMNS_TO_DROP
        )
        
        # Initialize metrics tracker with save_dir
        metrics_tracker = MetricsTracker(save_dir=str(RESULTS_DIR))
        
        # Initialize plot generator
        plot_generator = PlotGenerator(save_dir=str(PLOTS_DIR))
        
        # Initialize federated coordinator
        coordinator = FederatedCoordinator(
            config=FL_CONFIG,
            local_models_config=LOCAL_MODELS,
            global_model_config=GLOBAL_MODEL_CONFIG,
            data_loader=data_loader
        )
        
        # Initialize local models
        logger.info("Loading local models...")
        coordinator._initialize_local_models()
        
        # Ensure baseline metrics exist for visualization
        logger.info("Creating baseline metrics...")
        metrics_tracker.ensure_baseline_metrics()
        
        # Mark system as successfully initialized
        state_manager.set_initialized(True)
        state_manager.clear_error()
        
        logger.info("System initialization completed successfully - Advanced Analytics disabled")
        return True
        
    except Exception as e:
        error_msg = f"Error initializing system: {e}"
        logger.error(error_msg, exc_info=True)
        if state_manager:
            state_manager.set_error(error_msg)
        else:
            # Fallback if state manager failed to initialize
            logger.error("State manager failed to initialize")
        return False

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/models')
def models():
    """Models overview page."""
    return render_template('models.html')

@app.route('/training')
def training():
    """Training control page."""
    return render_template('training.html')

@app.route('/map')
def device_map():
    """Interactive map showing local model locations."""
    return render_template('map.html')

@app.route('/about')
def about():
    """About page with animated FL diagram."""
    return render_template('about.html')

@app.route('/help')
def help_page():
    """Help and documentation page."""
    return render_template('help.html')

@app.route('/performance')
def performance_analytics():
    """Advanced performance analytics page."""
    return render_template('performance.html')

@app.route('/flow')
def flow_diagram():
    """Animated HETROFL workflow diagram page."""
    return render_template('flow.html')

@app.route('/api/status')
def get_status():
    """Get system status."""
    global coordinator, state_manager, metrics_tracker
    
    if not state_manager:
        return jsonify({'error': 'System not initialized'})
    
    # Get base status from state manager
    status = state_manager.get_system_status()
    
    # Add training status
    training_status = state_manager.get_training_status()
    status.update(training_status)
    
    # Add coordinator status if available
    if coordinator:
        try:
            coordinator_status = coordinator.get_training_status()
            status.update(coordinator_status)
        except Exception as e:
            logger.warning(f"Error getting coordinator status: {e}")
    
    # Add metrics info
    if metrics_tracker:
        try:
            status['has_metrics'] = metrics_tracker.has_data()
            status['current_round'] = metrics_tracker.get_current_round()
        except Exception as e:
            logger.warning(f"Error getting metrics info: {e}")
    
    # Add last update time
    status['last_update'] = datetime.now().isoformat()
    
    return jsonify(status)

@app.route('/api/models/info')
def get_models_info():
    """Get information about all models."""
    global coordinator
    
    if not coordinator:
        return jsonify({'error': 'System not initialized'})
    
    try:
        models_info = {
            'local_models': {},
            'global_model': {}
        }
        
        # Local models info
        if coordinator.local_models:
            for model_name, model_adapter in coordinator.local_models.items():
                try:
                    models_info['local_models'][model_name] = model_adapter.get_model_info()
                except Exception as e:
                    logger.warning(f"Error getting info for {model_name}: {e}")
                    models_info['local_models'][model_name] = {
                        'type': 'unknown',
                        'is_loaded': False,
                        'error': str(e)
                    }
        
        # Global model info
        if coordinator.global_model:
            try:
                models_info['global_model'] = coordinator.global_model.get_model_summary()
            except Exception as e:
                logger.warning(f"Error getting global model info: {e}")
                models_info['global_model'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            models_info['global_model'] = {
                'status': 'not_initialized',
                'message': 'Global model not yet created'
            }
        
        return jsonify(models_info)
        
    except Exception as e:
        logger.error(f"Error in get_models_info: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/metrics/latest')
def get_latest_metrics():
    """Get latest metrics for all models."""
    global metrics_tracker
    
    if not metrics_tracker:
        return jsonify({'error': 'Metrics tracker not initialized'})
    
    try:
        latest_metrics = {
            'global': {},
            'local': {}
        }
        
        # Get global metrics
        try:
            global_metrics = metrics_tracker.get_latest_metrics()
            if global_metrics and len(global_metrics) > 0:
                latest_metrics['global'] = global_metrics
            else:
                # Provide default structure
                latest_metrics['global'] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'loss': 1.0,
                    'training_time': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"Error getting global metrics: {e}")
            latest_metrics['global'] = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'loss': 1.0,
                'training_time': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Get latest metrics for each local model - use actual model names from existing data
        local_model_names = ['xgboost', 'random_forest', 'catboost']
        for model_name in local_model_names:
            try:
                local_metrics = metrics_tracker.get_latest_metrics(model_name)
                if local_metrics and len(local_metrics) > 0:
                    latest_metrics['local'][model_name] = local_metrics
                else:
                    # Provide default structure
                    latest_metrics['local'][model_name] = {
                        'accuracy': 0.0,
                        'f1_score': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'loss': 1.0,
                        'training_time': 0.0,
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Error getting metrics for {model_name}: {e}")
                latest_metrics['local'][model_name] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'loss': 1.0,
                    'training_time': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
        
        return jsonify(sanitize_for_json(latest_metrics))
        
    except Exception as e:
        logger.error(f"Error in get_latest_metrics: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/metrics/history')
def get_metrics_history():
    """Get metrics history for visualization."""
    global metrics_tracker, state_manager
    
    if not metrics_tracker:
        return jsonify({'error': 'Metrics tracker not initialized'})
    
    try:
        # Check if system is initialized first
        if not state_manager or not state_manager.is_initialized():
            return jsonify({
                'error': 'System initialization in progress',
                'status': 'initializing',
                'message': 'Please wait while the system is initializing'
            })
            
        history = {
            'global': [],
            'local': {}
        }
        
        # Get global history
        try:
            global_df = metrics_tracker.get_metrics_dataframe()
            if not global_df.empty:
                # Convert DataFrame to records and ensure proper formatting
                records = global_df.to_dict('records')
                for record in records:
                    # Ensure all required fields exist
                    if 'timestamp' not in record:
                        record['timestamp'] = datetime.now().isoformat()
                    if 'round' not in record:
                        record['round'] = len(history['global']) + 1
                history['global'] = records
            else:
                # Provide sample data point for visualization
                history['global'] = [{
                    'round': 0,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'loss': 1.0,
                    'training_time': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'is_sample_data': True
                }]
                logger.info("No global metrics data found, providing sample data")
        except Exception as e:
            logger.warning(f"Error getting global metrics history: {e}")
            history['global'] = [{
                'round': 0,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'loss': 1.0,
                'training_time': 0.0,
                'timestamp': datetime.now().isoformat(),
                'is_sample_data': True,
                'error_detail': str(e)
            }]
        
        # Get history for each local model - use actual model names
        local_model_names = ['xgboost', 'random_forest', 'catboost']
        for model_name in local_model_names:
            try:
                df = metrics_tracker.get_metrics_dataframe(model_name)
                if not df.empty:
                    # Convert DataFrame to records and ensure proper formatting
                    records = df.to_dict('records')
                    for record in records:
                        # Ensure all required fields exist
                        if 'timestamp' not in record:
                            record['timestamp'] = datetime.now().isoformat()
                        if 'round' not in record:
                            record['round'] = len(history['local'].get(model_name, [])) + 1
                    history['local'][model_name] = records
                else:
                    # Provide sample data point for visualization
                    history['local'][model_name] = [{
                        'round': 0,
                        'accuracy': 0.0,
                        'f1_score': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'loss': 1.0,
                        'training_time': 0.0,
                        'timestamp': datetime.now().isoformat(),
                        'is_sample_data': True
                    }]
            except Exception as e:
                logger.warning(f"Error getting metrics history for {model_name}: {e}")
                history['local'][model_name] = [{
                    'round': 0,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'loss': 1.0,
                    'training_time': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'is_sample_data': True,
                    'error_detail': str(e)
                }]
        
        # Add system status info
        history['system_status'] = {
            'initialized': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(sanitize_for_json(history))
        
    except Exception as e:
        error_msg = f"Error retrieving metrics history: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': error_msg,
            'status': 'error',
            'system_status': {
                'initialized': state_manager.is_initialized() if state_manager else False,
                'timestamp': datetime.now().isoformat()
            }
        })

@app.route('/api/metrics/improvements')
def get_improvements():
    """Get improvement percentages for all models."""
    global metrics_tracker
    
    if not metrics_tracker:
        return jsonify({'error': 'Metrics tracker not initialized'})
    
    try:
        improvements = {
            'global': {},
            'local': {}
        }
        
        # Get global improvements
        try:
            global_improvements = metrics_tracker.calculate_improvement_percentage()
            if global_improvements:
                improvements['global'] = global_improvements
            else:
                # Provide default structure
                improvements['global'] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                }
        except Exception as e:
            logger.warning(f"Error calculating global improvements: {e}")
            improvements['global'] = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        # Get improvements for each local model
        for model_name in LOCAL_MODELS.keys():
            try:
                local_improvements = metrics_tracker.calculate_improvement_percentage(model_name)
                if local_improvements:
                    improvements['local'][model_name] = local_improvements
                else:
                    improvements['local'][model_name] = {
                        'accuracy': 0.0,
                        'f1_score': 0.0,
                        'precision': 0.0,
                        'recall': 0.0
                    }
            except Exception as e:
                logger.warning(f"Error calculating improvements for {model_name}: {e}")
                improvements['local'][model_name] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                }
        
        return jsonify(sanitize_for_json(improvements))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start federated training."""
    global coordinator, metrics_tracker, plot_generator, data_loader, state_manager
    
    if not coordinator or not state_manager or not state_manager.is_initialized():
        return jsonify({'error': 'System not initialized'})
    
    if state_manager.is_training():
        return jsonify({'error': 'Training already in progress'})
    
    try:
        # Get training parameters
        data = request.get_json() or {}
        sample_size = data.get('sample_size', 50000)
        max_rounds = data.get('max_rounds', 50)
        
        # Update state manager
        state_manager.start_training(max_rounds=max_rounds, sample_size=sample_size)
        
        # Start training in background thread
        def training_worker():
            try:
                logger.info("Starting background training worker")
                
                # First train global model if not already trained
                if not coordinator.global_model or not coordinator.global_model.is_trained:
                    logger.info("Training global model initially...")
                    initial_metrics = coordinator.train_global_model_initial(sample_size=sample_size)
                    
                    if not initial_metrics:
                        logger.error("Failed to train global model")
                        state_manager.stop_training()
                        return
                
                # Load test data for evaluation
                logger.info("Loading test data...")
                df_test = data_loader.load_data(sample_size=sample_size // 5)  # Smaller test set
                if df_test is None:
                    logger.error("Failed to load test data")
                    state_manager.stop_training()
                    return
                
                X_test, y_test = data_loader.preprocess_data(df_test, fit_transformers=False)
                
                # Load distillation data (subset of main dataset)
                df_distill = data_loader.load_data(sample_size=sample_size // 10)  # Even smaller for distillation
                if df_distill is not None:
                    X_distill, y_distill = data_loader.preprocess_data(df_distill, fit_transformers=False)
                else:
                    X_distill, y_distill = None, None
                
                # Start federated training
                success = coordinator.start_federated_training(
                    X_test, y_test,
                    X_distill, y_distill,
                    metrics_tracker, plot_generator
                )
                
                if not success:
                    logger.error("Federated training failed")
                else:
                    logger.info("Federated training completed successfully")
                    
                    # After training completes, run comprehensive evaluation
                    try:
                        logger.info("Running post-training comprehensive evaluation...")
                        evaluation_results = coordinator.evaluate_models_on_balanced_data()
                        
                        if evaluation_results:
                            logger.info(f"Comprehensive evaluation completed for {len(evaluation_results)} models")
                            
                            # Log summary of results
                            for model_name, results in evaluation_results.items():
                                balanced_acc = results.get('balanced_metrics', {}).get('accuracy', 0.0)
                                imbalanced_acc = results.get('imbalanced_metrics', {}).get('accuracy', 0.0)
                                logger.info(f"{model_name}: Balanced={balanced_acc:.3f}, Imbalanced={imbalanced_acc:.3f}")
                        else:
                            logger.warning("Comprehensive evaluation returned no results")
                            
                    except Exception as e:
                        logger.error(f"Error in post-training evaluation: {e}")
                    
            except Exception as e:
                logger.error(f"Error in training worker: {e}")
            finally:
                state_manager.stop_training()
                logger.info("Training worker finished")
        
        # Start training in background thread
        training_thread = threading.Thread(target=training_worker, daemon=True)
        training_thread.start()
        
        return jsonify({'success': True, 'message': 'Federated training started in background'})
        
    except Exception as e:
        error_msg = f"Error starting training: {e}"
        logger.error(error_msg)
        state_manager.stop_training()
        return jsonify({'error': error_msg})

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop federated training."""
    global coordinator, state_manager
    
    if not coordinator or not state_manager:
        return jsonify({'error': 'System not initialized'})
    
    try:
        coordinator.stop_federated_training()
        state_manager.stop_training()
        return jsonify({'success': True, 'message': 'Federated training stopped'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/evaluation/comprehensive', methods=['POST'])
def run_comprehensive_evaluation():
    """Run comprehensive evaluation on balanced data and generate plots."""
    global coordinator, metrics_tracker, plot_generator
    
    if not coordinator:
        return jsonify({'error': 'System not initialized'})
    
    try:
        logger.info("Starting comprehensive evaluation via API...")
        
        # Run comprehensive evaluation
        evaluation_results = coordinator.evaluate_models_on_balanced_data()
        
        if not evaluation_results:
            return jsonify({'error': 'No evaluation results generated'})
        
        # Prepare response data
        response_data = {
            'success': True,
            'message': f'Comprehensive evaluation completed for {len(evaluation_results)} models',
            'results': {}
        }
        
        # Add summary of results
        for model_name, results in evaluation_results.items():
            balanced_metrics = results.get('balanced_metrics', {})
            imbalanced_metrics = results.get('imbalanced_metrics', {})
            training_history = results.get('training_history', [])
            
            response_data['results'][model_name] = {
                'balanced_accuracy': balanced_metrics.get('accuracy', 0.0),
                'imbalanced_accuracy': imbalanced_metrics.get('accuracy', 0.0),
                'balanced_f1': balanced_metrics.get('f1_score', 0.0),
                'imbalanced_f1': imbalanced_metrics.get('f1_score', 0.0),
                'training_rounds': len(training_history),
                'improvement': (
                    training_history[-1].get('accuracy', 0.0) - training_history[0].get('accuracy', 0.0)
                    if len(training_history) > 1 else 0.0
                )
            }
        
        logger.info(f"Comprehensive evaluation API completed: {response_data['message']}")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error in comprehensive evaluation: {e}"
        logger.error(error_msg)
        return jsonify({'error': error_msg})

@app.route('/api/results/latest')
def get_latest_results():
    """Get latest training results."""
    global coordinator
    
    if not coordinator:
        return jsonify({'error': 'System not initialized'})
    
    try:
        latest_results = coordinator.get_latest_results()
        return jsonify(latest_results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/results/all')
def get_all_results():
    """Get all training results."""
    global coordinator
    
    if not coordinator:
        return jsonify({'error': 'System not initialized'})
    
    try:
        all_results = coordinator.get_all_results()
        return jsonify(all_results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/plots/<plot_type>')
def get_plot(plot_type):
    """Get specific plot data."""
    global plot_generator, metrics_tracker
    
    if not plot_generator or not metrics_tracker:
        return jsonify({'error': 'Visualization components not initialized'})
    
    try:
        if plot_type == 'metrics_comparison':
            # Get metrics data
            global_df = metrics_tracker.get_metrics_dataframe()
            local_metrics = {}
            
            for model_name in LOCAL_MODELS.keys():
                try:
                    local_df = metrics_tracker.get_metrics_dataframe(model_name)
                    if not local_df.empty:
                        local_metrics[model_name] = {
                            'accuracy': local_df['accuracy'].tolist(),
                            'f1_score': local_df['f1_score'].tolist(),
                            'precision': local_df['precision'].tolist(),
                            'recall': local_df['recall'].tolist()
                        }
                    else:
                        # Provide default data for empty models
                        local_metrics[model_name] = {
                            'accuracy': [0.0],
                            'f1_score': [0.0],
                            'precision': [0.0],
                            'recall': [0.0]
                        }
                except Exception as e:
                    logger.warning(f"Error getting metrics for {model_name}: {e}")
                    local_metrics[model_name] = {
                        'accuracy': [0.0],
                        'f1_score': [0.0],
                        'precision': [0.0],
                        'recall': [0.0]
                    }
            
            if not global_df.empty:
                global_metrics = {
                    'accuracy': global_df['accuracy'].tolist(),
                    'f1_score': global_df['f1_score'].tolist(),
                    'precision': global_df['precision'].tolist(),
                    'recall': global_df['recall'].tolist()
                }
            else:
                # Provide default global metrics
                global_metrics = {
                    'accuracy': [0.0],
                    'f1_score': [0.0],
                    'precision': [0.0],
                    'recall': [0.0]
                }
                
            fig = plot_generator.plot_comparison_chart(
                global_metrics, local_metrics, 'accuracy'
            )
            return jsonify({'plot_json': fig.to_json()})
        
        elif plot_type == 'improvements':
            # Get improvement data
            try:
                global_improvements = metrics_tracker.calculate_improvement_percentage()
                if not global_improvements:
                    global_improvements = {
                        'accuracy_improvement': 0.0,
                        'f1_score_improvement': 0.0,
                        'precision_improvement': 0.0,
                        'recall_improvement': 0.0
                    }
            except Exception as e:
                logger.warning(f"Error calculating global improvements: {e}")
                global_improvements = {
                    'accuracy_improvement': 0.0,
                    'f1_score_improvement': 0.0,
                    'precision_improvement': 0.0,
                    'recall_improvement': 0.0
                }
            
            improvements = {'global': global_improvements}
            
            for model_name in LOCAL_MODELS.keys():
                try:
                    local_improvements = metrics_tracker.calculate_improvement_percentage(model_name)
                    if not local_improvements:
                        local_improvements = {
                            'accuracy_improvement': 0.0,
                            'f1_score_improvement': 0.0,
                            'precision_improvement': 0.0,
                            'recall_improvement': 0.0
                        }
                    improvements[model_name] = local_improvements
                except Exception as e:
                    logger.warning(f"Error calculating improvements for {model_name}: {e}")
                    improvements[model_name] = {
                        'accuracy_improvement': 0.0,
                        'f1_score_improvement': 0.0,
                        'precision_improvement': 0.0,
                        'recall_improvement': 0.0
                    }
            
            fig = plot_generator.plot_improvement_percentages(improvements)
            return jsonify({'plot_json': fig.to_json()})
        
        elif plot_type == 'training_progress':
            # Get training progress data
            global_df = metrics_tracker.get_metrics_dataframe()
            
            if not global_df.empty:
                training_data = {
                    'loss': global_df['loss'].tolist() if 'loss' in global_df.columns else [1.0],
                    'training_time': global_df['training_time'].tolist() if 'training_time' in global_df.columns else [0.0]
                }
            else:
                # Provide default training data
                training_data = {
                    'loss': [1.0],
                    'training_time': [0.0]
                }
                
            fig = plot_generator.plot_training_progress(training_data)
            return jsonify({'plot_json': fig.to_json()})
        
        elif plot_type == 'model_accuracies':
            # Create a simple bar chart of latest model accuracies
            try:
                latest_metrics = get_latest_metrics().get_json()
                
                model_names = []
                accuracies = []
                
                # Global model
                if 'global' in latest_metrics and 'accuracy' in latest_metrics['global']:
                    model_names.append('Global MLP')
                    accuracies.append(latest_metrics['global']['accuracy'])
                
                # Local models
                if 'local' in latest_metrics:
                    for model_name, metrics in latest_metrics['local'].items():
                        if 'accuracy' in metrics:
                            model_names.append(model_name.replace('_', ' ').title())
                            accuracies.append(metrics['accuracy'])
                
                # Create simple bar chart
                import plotly.graph_objs as go
                fig = go.Figure(data=[
                    go.Bar(
                        x=model_names,
                        y=accuracies,
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)]
                    )
                ])
                
                fig.update_layout(
                    title='Latest Model Accuracies',
                    xaxis_title='Model',
                    yaxis_title='Accuracy',
                    template="plotly_white",
                    height=400
                )
                
                return jsonify({'plot_json': fig.to_json()})
                
            except Exception as e:
                logger.error(f"Error creating model accuracies plot: {e}")
                return jsonify({'error': f'Error creating plot: {str(e)}'})
        
        return jsonify({'error': f'Unknown plot type: {plot_type}'})
        
    except Exception as e:
        logger.error(f"Error in get_plot for {plot_type}: {e}")
        return jsonify({'error': f'Error generating plot: {str(e)}'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('status', get_status().get_json())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle real-time update requests."""
    try:
        # Send current status
        emit('status_update', get_status().get_json())
        
        # Send latest metrics
        if metrics_tracker:
            latest_metrics = get_latest_metrics().get_json()
            emit('metrics_update', latest_metrics)
        
        # Send latest results
        if coordinator:
            latest_results = get_latest_results().get_json()
            emit('results_update', latest_results)
            
    except Exception as e:
        emit('error', {'message': str(e)})

def background_updates():
    """Send periodic updates to connected clients."""
    while True:
        try:
            with app.app_context():
                if state_manager and state_manager.is_training():
                    socketio.emit('status_update', get_status().get_json())
                    
                    if metrics_tracker:
                        latest_metrics = get_latest_metrics().get_json()
                        socketio.emit('metrics_update', latest_metrics)
                    
                    if coordinator:
                        latest_results = get_latest_results().get_json()
                        socketio.emit('results_update', latest_results)
            
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in background updates: {e}")
            time.sleep(10)

if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Start background update thread
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    # Run Flask app
    logger.info(f"Starting HETROFL GUI on {GUI_CONFIG['host']}:{GUI_CONFIG['port']}")
    socketio.run(
        app,
        host=GUI_CONFIG['host'],
        port=GUI_CONFIG['port'],
        debug=GUI_CONFIG['debug']
    ) 

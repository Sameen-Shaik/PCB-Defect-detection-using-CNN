"""Flask application factory."""
import os
from flask import Flask
from .config import config


def create_app(config_name=None):
    """Application factory pattern.
    
    Args:
        config_name: Configuration name (development, production, testing)
    
    Returns:
        Configured Flask application
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), '..', 'web', 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), '..', 'web', 'static'))
    
    # Load configuration
    app.config.from_object(config.get(config_name, config['default']))
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from .routes import main_bp
    from .api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    
    # Preload model on startup (optional, can be lazy-loaded)
    with app.app_context():
        from .models import model_manager
        try:
            model_manager.load_model(app.config['MODEL_PATH'])
            app.logger.info(f"Model loaded successfully from {app.config['MODEL_PATH']}")
        except Exception as e:
            app.logger.warning(f"Could not preload model: {e}")
    
    return app

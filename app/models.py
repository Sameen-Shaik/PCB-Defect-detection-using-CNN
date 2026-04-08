"""Machine Learning model management."""
import os
import torch
from ultralytics import YOLO
from flask import current_app


class ModelManager:
    """Singleton pattern for managing ML models."""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path=None):
        """Load the YOLO model."""
        if self._model is None:
            if model_path is None:
                model_path = current_app.config.get('MODEL_PATH', 
                    'runs/detect/train9/weights/best.pt')
            
            # Ensure model path exists, fallback to pretrained if needed
            if not os.path.exists(model_path):
                # Try alternate paths
                alt_paths = [
                    'yolov8m.pt',
                    'yolo26n.pt',
                    os.path.join(current_app.config['BASE_DIR'], 'yolov8m.pt'),
                    os.path.join(current_app.config['BASE_DIR'], 'yolo26n.pt'),
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
            
            self._model = YOLO(model_path)
        return self._model
    
    def get_model(self):
        """Get the loaded model."""
        if self._model is None:
            return self.load_model()
        return self._model
    
    @property
    def model(self):
        """Property accessor for the model."""
        return self.get_model()


# Global model manager instance
model_manager = ModelManager()

"""API routes for the Flask application."""
import os
import cv2
import base64
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from .models import model_manager

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@api_bp.route("/predict", methods=["POST"])
def predict():
    """API endpoint for PCB defect detection.
    
    Accepts:
        - multipart/form-data with 'image' file
        - OR JSON with 'image_base64' field
    
    Returns:
        JSON with detections list and annotated image base64
    """
    # Handle multipart form data (file upload)
    if request.files and "image" in request.files:
        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"error": "No image file provided"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        filename = secure_filename(file.filename)
        image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(image_path)
    
    # Handle base64 encoded image
    elif request.is_json and "image_base64" in request.json:
        import base64
        import io
        from PIL import Image
        
        base64_data = request.json["image_base64"]
        try:
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            
            filename = "upload_base64.jpg"
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            image.save(image_path)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
    else:
        return jsonify({"error": "No image file or image_base64 provided"}), 400
    
    # Run inference
    try:
        model = model_manager.model
        results = model(image_path)
        result = results[0]
        
        img = cv2.imread(image_path)
        detections = []
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detections.append({
                "class": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{model.names[cls_id]} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Encode image to base64
        _, buffer = cv2.imencode(".jpg", img)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        
        return jsonify({
            "detections": detections,
            "image_base64": image_base64,
            "count": len(detections)
        })
        
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager._model is not None
    })

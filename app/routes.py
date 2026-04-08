"""Web UI routes for the Flask application."""
import os
import cv2
from flask import Blueprint, render_template, request, current_app
from werkzeug.utils import secure_filename
from .models import model_manager

# Create blueprint
main_bp = Blueprint('main', __name__, template_folder='../web/templates')


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@main_bp.route("/", methods=["GET", "POST"])
def index():
    """Main route for PCB defect detection web interface."""
    image_url = None
    detections = []
    
    if request.method == "POST":
        file = request.files.get("image")
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(current_app.config['RESULT_FOLDER'], filename)
            
            # Ensure directories exist
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(current_app.config['RESULT_FOLDER'], exist_ok=True)
            
            # Save uploaded image
            file.save(input_path)
            
            # Get model and run inference
            model = model_manager.model
            results = model(input_path)
            result = results[0]
            
            # Read image for annotation
            img = cv2.imread(input_path)
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = f"{model.names[cls_id]} ({conf:.2f})"
                detections.append(label)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            # Save output image
            cv2.imwrite(output_path, img)
            
            image_url = f"/static/results/{filename}"
    
    return render_template(
        "index.html",
        image_url=image_url,
        detections=detections
    )

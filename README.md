# PCB Defect Detection using CNN and YOLO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv8-v8.0+-green.svg)](https://ultralytics.com)

An end-to-end deep learning solution for detecting defects in Printed Circuit Boards (PCBs) using Convolutional Neural Networks (CNN) and YOLO (You Only Look Once) object detection. The project includes multiple model architectures, training pipelines, and deployment options via Flask web applications and REST APIs.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Deployment](#deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project addresses the critical need for automated quality control in electronics manufacturing. PCB defects can lead to device failures, and manual inspection is time-consuming and error-prone. This solution leverages deep learning to automatically detect and classify 6 common types of PCB defects with high accuracy.

### Supported Defect Types

| Defect Class | Description |
|--------------|-------------|
| **Missing_hole** | Absence of a required drilled hole in the PCB |
| **Mouse_bite** | Small irregular copper residues along the edge of a pad or trace |
| **Open_circuit** | Broken or disconnected copper traces |
| **Short** | Unintended electrical connections between traces |
| **Spur** | Thin unwanted copper projections from traces |
| **Spurious_copper** | Unintended copper patches or residues |

---

## Features

- **Multi-Model Support**: Implements both custom CNN and state-of-the-art YOLOv8
- **Data Augmentation**: Random rotation and Gaussian blur for improved generalization
- **Web Interface**: User-friendly Flask web application for image upload and visualization
- **REST API**: JSON API for integration with external systems
- **Cloud-Ready**: Frontend supports AWS Lambda and API Gateway deployment
- **GPU Acceleration**: CUDA support for faster training and inference
- **Modular Design**: Reusable helper functions for training and evaluation

---

## Dataset

The project uses the [PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) from Kaggle, which contains:
- **693 images** of PCB boards
- **6 defect classes** with balanced distribution
- Images resized to 224×224 or 330×330 pixels for processing

### Dataset Download

```bash
python "download data.py"
```

Or manually download from Kaggle and extract to `Data/1/PCB_DATASET/images/`

---

## Project Structure

```
PCB-Defect-detection-using-CNN/
├── app/                          # Flask application package
│   ├── __init__.py              # Application factory
│   ├── config.py                # Configuration settings
│   ├── models.py                # ML model management
│   ├── routes.py                # Web UI routes (from app.py)
│   └── api.py                   # API routes (from api_app.py)
├── ml/                           # Machine learning code
│   ├── __init__.py
│   ├── helper_functions.py      # Training utilities
│   ├── train.py                 # Production training script
│   ├── train_copy.py            # Alternative configuration
│   ├── train_copy_2.py          # Experimental variant
│   └── models/                  # Model weights
│       ├── yolo26n.pt
│       └── yolov8m.pt
├── data/                         # Data scripts
│   └── download_data.py         # Dataset download
├── notebooks/                    # Jupyter notebooks
│   ├── main.ipynb               # Main experimentation
│   ├── prepare_annotations.ipynb
│   ├── Untitled.ipynb
│   └── Untitled1.ipynb
├── web/                          # Web assets
│   ├── static/                  # CSS, JS, results
│   │   ├── style.css
│   │   └── results/
│   └── templates/               # HTML templates
│       └── index.html
├── uploads/                      # Temporary upload storage
├── runs/                         # YOLO training outputs
├── tests/                        # Test files
├── Data/                         # Dataset (downloaded)
│   └── 1/PCB_DATASET/images/
├── run.py                        # Development entry point
├── wsgi.py                       # Production entry point
├── requirements.txt
├── LICENSE
└── README.md
```

**Legacy files (kept for reference):**
- `app.py` - Original Flask web app
- `api_app.py` - Original Flask API
- `index.html` - Standalone frontend

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sameen-Shaik/PCB-Defect-detection-using-CNN.git
cd PCB-Defect-detection-using-CNN
```

### Step 2: Create Virtual Environment

```bash
# Using conda
conda create -n pcb-defect python=3.10
conda activate pcb-defect

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- torch >= 2.0
- torchvision
- ultralytics (YOLO)
- flask
- opencv-python
- scikit-learn
- matplotlib
- tqdm
- kagglehub

### Step 4: Download Dataset

```bash
python "download data.py"
```

---

## Usage

### 1. Training a Custom CNN Model

```bash
cd ml
python train.py
```

Or from project root:
```bash
python -m ml.train
```

### 2. Running the Web Application

#### Development Server

```bash
python run.py
```

Access at: `http://localhost:5000`

#### Production Server (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

#### API Endpoints

The new structure provides unified API access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Web interface |
| `/api/predict` | POST | JSON API for predictions |
| `/api/health` | GET | Health check |

**Example API Request:**
```bash
curl -X POST -F "image=@test_pcb.jpg" http://localhost:5000/api/predict
```

### 3. Jupyter Notebook Exploration

```bash
jupyter notebook notebooks/main.ipynb
```

### 4. Download Dataset

```bash
python data/download_data.py
```

### Model Architectures

### Custom CNN Architecture

The custom CNN (`train.py`) consists of:

```
Input (3×330×330)
    ↓
Conv2d(3→32, 3×3) → ReLU → MaxPool2d(2×2)
    ↓
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2×2)
    ↓
Conv2d(64→128, 3×3) → ReLU → MaxPool2d(2×2)
    ↓
AdaptiveAvgPool2d(1×1) → Flatten
    ↓
Linear(128→6) → Output
```

**Features:**
- 3 convolutional blocks with max pooling
- Progressive feature extraction (32→64→128 channels)
- Adaptive pooling for input size flexibility
- Single fully-connected classification layer

### YOLOv8 Architecture

For object detection with bounding boxes:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8m.pt")

# Fine-tune on PCB defects
model.train(data="pcb_data.yaml", epochs=100, imgsz=640)
```

**Benefits:**
- Real-time inference capability
- Bounding box predictions
- Multi-scale detection
- High accuracy on small objects

---

## Training

### Data Preprocessing

**Training Transformations:**
```python
train_transforms = transforms.Compose([
    transforms.Resize((330, 330)),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
])
```

**Test Transformations:**
```python
test_transforms = transforms.Compose([
    transforms.Resize((330, 330)),
    transforms.ToTensor()
])
```

### Data Splitting

- **Training Set**: 80% (554 images)
- **Test Set**: 20% (139 images)
- **Stratified Split**: Maintains class distribution across splits

### Training Loop

The training process (`helper_functions.py`):

1. **Forward Pass**: Compute predictions
2. **Loss Calculation**: Cross-entropy loss
3. **Backpropagation**: Gradient computation
4. **Optimizer Step**: SGD parameter updates
5. **Evaluation**: Accuracy and loss metrics

### Monitoring

Training progress displays:
```
Epoch: 0
---------
Train Loss: 1.234 | Train Accuracy: 65.42 | Test Loss: 1.123 | Test Accuracy: 68.90
```

---

## Deployment

### Local Deployment

**Development Server:**
```bash
python run.py
```

**Production with Gunicorn:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

### Cloud Deployment (AWS)

The `web/templates/index.html` includes pre-configured AWS integration:

1. Deploy `app/api.py` logic to AWS Lambda
2. Configure API Gateway pointing to `/api/predict`
3. Update the API endpoint in `web/templates/index.html`:
   ```javascript
   const response = await fetch("https://YOUR_API_GATEWAY_URL/api/predict", {...})
   ```

**Static Website Hosting:**
- Upload `web/templates/index.html` to S3
- Enable static website hosting
- Configure CORS for API Gateway

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "run.py"]
```

Build and run:
```bash
docker build -t pcb-defect-detection .
docker run -p 5000:5000 pcb-defect-detection
```

---

## Results

### Model Performance

| Model | Architecture | Input Size | Accuracy | Training Time |
|-------|-------------|------------|----------|---------------|
| Custom CNN | 3 Conv Layers | 330×330 | ~95% | ~15 min (GPU) |
| YOLOv8-N | Ultralytics | 640×640 | ~92% | ~30 min (GPU) |
| YOLOv8-M | Ultralytics | 640×640 | ~97% | ~45 min (GPU) |

### Confusion Matrix

The model effectively distinguishes between all 6 defect types with minimal confusion between similar classes like `Spur` and `Spurious_copper`.

### Inference Speed

| Device | CNN (per image) | YOLO (per image) |
|--------|-----------------|------------------|
| RTX 4050 | ~5ms | ~12ms |
| CPU (i7) | ~120ms | ~200ms |

---

## API Reference

### Endpoints

#### POST `/api/predict`

Detect defects in an uploaded PCB image.

**Request (multipart/form-data):**
- Parameter: `image` (file)

**Request (JSON):**
```json
{
  "image_base64": "base64-encoded-image-string"
}
```

**Response:**
```json
{
  "detections": [
    {
      "class": "string",
      "confidence": 0.923,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "image_base64": "string",
  "count": 1
}
```

#### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET `/`

Renders the web interface (form-based upload).

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install pytest black flake8

# Run linting
black *.py
flake8 *.py

# Run tests
pytest tests/
```

---

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory  
**Solution**: Reduce `BATCH_SIZE` in `ml/train.py`

**Issue**: Dataset not found  
**Solution**: Run `python data/download_data.py` or check `Data/` directory structure

**Issue**: Model file not found  
**Solution**: Train the model first or download pretrained weights to `runs/detect/train9/weights/`

**Issue**: Port already in use  
**Solution**: Change port in `run.py`: `app.run(debug=True, port=5001)` or set `PORT` environment variable

---

## Future Enhancements

- [ ] Add support for more defect types
- [ ] Implement model quantization for edge deployment
- [ ] Add real-time video stream processing
- [ ] Integrate with industrial IoT systems
- [ ] Develop mobile application
- [ ] Add explainability features (Grad-CAM)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [PCB Defects Dataset by akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects) on Kaggle
- YOLO implementation: [Ultralytics](https://ultralytics.com)
- PyTorch framework: [PyTorch](https://pytorch.org)

---

## Contact

**Sameen Shaik** - [@Sameen-Shaik](https://github.com/Sameen-Shaik)

Project Link: [https://github.com/Sameen-Shaik/PCB-Defect-detection-using-CNN](https://github.com/Sameen-Shaik/PCB-Defect-detection-using-CNN)

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{pcb_defect_detection_cnn,
  author = {Shaik, Sameen},
  title = {PCB Defect Detection using CNN and YOLO},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Sameen-Shaik/PCB-Defect-detection-using-CNN}}
}
```

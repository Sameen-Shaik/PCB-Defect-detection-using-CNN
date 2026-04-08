from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Data/1/PCB_DATASET"
YOLO_DATA_DIR = PROJECT_ROOT / "Data/data_for_yolo"

ANNOTATIONS_DIR = DATA_DIR / "Annotations"
IMAGES_DIR = DATA_DIR / "images"
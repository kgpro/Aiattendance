from ultralytics import YOLO
import os
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

def load_yolo_model():
    # Define model paths
    model_dir = os.path.join(BASE_DIR, 'runs')
    model_path = os.path.join(model_dir, 'best.pt')

    # Load model
    model = YOLO(model_path)

    # Move to GPU if available
    if torch.cuda.is_available():
        model.to("cuda:0")
        print("YOLO model loaded on GPU:", torch.cuda.get_device_name(0))
        logger.info(f"YOLO model loaded on GPU: {torch.cuda.get_device_name(0)}")

    return model

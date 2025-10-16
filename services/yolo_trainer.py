"""
YOLOv11 Face Detection Training Script

This script provides a complete pipeline for training a YOLOv11 model on a custom face dataset,
such as WIDER FACE. It includes dataset validation, YAML configuration management,
model training, and final evaluation with GPU enforcement options.


"""

import os
import sys
import argparse
import yaml
import torch
from ultralytics import YOLO


def check_dataset_structure(dataset_root):
    """
    Validates the dataset directory structure and reports file counts.

    Args:
        dataset_root (str): The path to the root of the dataset.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    required_dirs = [
        os.path.join(dataset_root, "images", "train"),
        os.path.join(dataset_root, "images", "val"),
        os.path.join(dataset_root, "labels", "train"),
        os.path.join(dataset_root, "labels", "val")
    ]

    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]

    if missing_dirs:
        print("[ERROR] Missing required dataset directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        return False

    stats = {}
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_root, "images", split)
        lbl_dir = os.path.join(dataset_root, "labels", split)
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        stats[f'{split}_images'] = img_count
        stats[f'{split}_labels'] = lbl_count

    print("[INFO] Dataset Statistics:")
    for key, count in stats.items():
        print(f"  - {key.replace('_', ' ').capitalize()}: {count}")

    if stats['train_images'] != stats['train_labels']:
        print(f"[WARNING] Mismatch in training set: {stats['train_images']} images vs {stats['train_labels']} labels.")
    if stats['val_images'] != stats['val_labels']:
        print(f"[WARNING] Mismatch in validation set: {stats['val_images']} images vs {stats['val_labels']} labels.")

    total_images = stats['train_images'] + stats['val_images']
    if total_images == 0:
        print("[ERROR] No images found in the dataset.")
        return False

    print(f"[INFO] Dataset validation passed. Found {total_images} total images.")
    return True


def write_yaml_config(yaml_path, dataset_root):
    """
    Creates or updates the YAML configuration file for the dataset.

    Args:
        yaml_path (str): The full path to save the YAML file.
        dataset_root (str): The path to the root of the dataset.
    """
    config = {
        'path': os.path.abspath(dataset_root),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'face'},
    }

    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Successfully wrote YAML configuration to {yaml_path}")
    except IOError as e:
        print(f"[ERROR] Could not write YAML file: {e}")
        sys.exit(1)


def main(args):
    """
    Main function to orchestrate the training pipeline.
    """
    print("--- YOLO Face Detection Training Initialized ---")

    # --- 1. Validate Paths and Dataset ---
    if not os.path.isdir(args.data_path):
        print(f"[ERROR] Dataset directory not found at: {args.data_path}")
        sys.exit(1)

    if not check_dataset_structure(args.data_path):
        print("[ERROR] Halting due to invalid dataset structure.")
        sys.exit(1)

    yaml_path = os.path.join(args.data_path, "data.yaml")
    write_yaml_config(yaml_path, args.data_path)

    # --- 2. Setup Training Environment ---
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"[INFO] Auto-detected device: {device.upper()}")
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        else:
            if args.require_gpu:
                print("[ERROR] GPU required but CUDA is not available!")
                print("[INFO] To use GPU, ensure you have:")
                print("  - NVIDIA GPU with CUDA support")
                print("  - PyTorch with CUDA support installed")
                print("  - Compatible CUDA drivers")
                print(
                    "[INFO] Check GPU availability with: python -c \"import torch; print(torch.cuda.is_available())\"")
                print("[INFO] Your current PyTorch version:", torch.__version__)
                sys.exit(1)
            else:
                device = 'cpu'
                print(f"[INFO] CUDA not available. Using device: {device.upper()}")
                print("[WARNING] Training on CPU is significantly slower than on GPU.")
    else:
        device = args.device
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(f"[ERROR] CUDA device '{device}' requested but CUDA is not available!")
                print("[INFO] To use GPU, ensure you have:")
                print("  - NVIDIA GPU with CUDA support")
                print("  - PyTorch with CUDA support installed")
                print("  - Compatible CUDA drivers")
                print("[INFO] Your current PyTorch version:", torch.__version__)
                sys.exit(1)
            print(f"[INFO] Using specified device: {device.upper()}")
        else:
            print(f"[INFO] Using specified device: {device.upper()}")

    # --- 3. Initialize Model ---
    print(f"[INFO] Initializing model with pre-trained weights: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    # --- 4. Start Training ---
    print("[INFO] Starting model training...")
    print(f"[INFO] Training configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Device: {device}")
    #
    # try:
    #     model.train(
    #         data=yaml_path,
    #         epochs=args.epochs,
    #         batch=args.batch,
    #         imgsz=args.imgsz,
    #         name=args.name,
    #         device=device,
    #         patience=20,
    #         project=args.project_dir,
    #         exist_ok=True
    #     )
    #     print("\n--- Training Completed Successfully ---")
    try:
        if args.resume:
            print(f"[INFO] Resuming training from: {args.resume}")
            if not os.path.exists(args.resume):
                print(f"[ERROR] Resume checkpoint not found: {args.resume}")
                sys.exit(1)
            model = YOLO(args.resume)  # Load from checkpoint instead
            # Training will automatically resume from where it left off
            model.train(
                data=yaml_path,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                name=args.name,
                device=device,
                patience=20,
                project=args.project_dir,
                exist_ok=True
            )
        else:
            model.train(
                data=yaml_path,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                name=args.name,
                device=device,
                patience=20,
                project=args.project_dir,
                exist_ok=True
            )
        print("\n--- Training Completed Successfully ---")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        if "out of memory" in str(e).lower():
            print("[SUGGESTION] Try reducing batch size with --batch 4 or --batch 2")
            print("[SUGGESTION] Or try a smaller model like yolo11s.pt or yolo11m.pt")
        else:
            print("Troubleshooting: Try reducing batch size or checking dataset integrity.")
        sys.exit(1)

    # --- 5. Final Evaluation ---
    print("[INFO] Running final validation on the best model...")
    try:
        best_model_path = os.path.join(args.project_dir, args.name, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            metrics = best_model.val()
            print("\n--- Final Model Evaluation Metrics ---")
            print(f"  - mAP50-95 (Box): {metrics.box.map:.4f}")
            print(f"  - mAP50 (Box):    {metrics.box.map50:.4f}")
            print(f"  - Precision (Box): {metrics.box.mp:.4f}")
            print(f"  - Recall (Box):    {metrics.box.mr:.4f}")
            print(f"\n[INFO] Best model and results saved in: {os.path.join(args.project_dir, args.name)}")
            print(f"[INFO] Model weights: {best_model_path}")
        else:
            print("[WARNING] Could not find 'best.pt' for final validation.")
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLOv11 Face Detection Training Script")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g., runs/train/face_detection_run/weights/last.pt)")

    parser.add_argument('--data_path', type=str, required=True, help="Path to the root dataset directory.")
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                        help="Pre-trained model to use (e.g., yolo11s.pt, yolo11m.pt, yolo11x.pt).")
    parser.add_argument('--epochs', type=int, default=100, help="Total number of training epochs.")
    parser.add_argument('--batch', type=int, default=10, help="Batch size for training.")
    parser.add_argument('--imgsz', type=int, default=416, help="Image size for training.")
    parser.add_argument('--name', type=str, default='face_detection_run', help="Name of the training run/experiment.")
    parser.add_argument('--project_dir', type=str, default='runs/train', help="Directory to save training runs.")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use for training (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument('--require_gpu', action='store_true',
                        help="Fail if GPU is not available (don't fall back to CPU)")

    args = parser.parse_args()
    main(args)
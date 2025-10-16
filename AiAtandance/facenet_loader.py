import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


class DeepFaceLite:
    def __init__(self, model_name="Facenet512", device=None):
        if model_name.lower() not in ["facenet512", "facenet"]:
            raise ValueError("This Lite wrapper currently supports only Facenet512")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def represent(self, img_path=None, img_array=None, enforce_detection=False):
        """
        Mimics DeepFace.represent:
        - Accepts img_path or NumPy array
        - Returns list[dict] with embedding + metadata
        """

        if img_array is None and img_path is None:
            raise ValueError("Provide either img_path or img_array")

        # Handle NumPy array input
        if img_array is not None:
            if isinstance(img_array, np.ndarray):
                if img_array.shape[-1] == 3:
                    img_array = img_array[:, :, ::-1]  # BGR → RGB
                img = Image.fromarray(img_array.astype("uint8"))
                facial_area = {
                    "x": 0,
                    "y": 0,
                    "w": img_array.shape[1],
                    "h": img_array.shape[0],
                }
            else:
                raise ValueError("img_array must be a NumPy array")
        else:
            img = Image.open(img_path).convert("RGB")
            facial_area = None

        # Preprocess and infer
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor).cpu().numpy().flatten()

        # Match DeepFace’s return format
        return [{
            "embedding": embedding.tolist(),
            "facial_area": facial_area,
            "dominant_emotion": None,
            "age": None,
            "gender": None
        }]

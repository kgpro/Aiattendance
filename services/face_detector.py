
import cv2
import os
class FaceCropper():
    def __init__(self, model,conf_threshold=0.35):
        """
        Initialize the FaceCropper with YOLOv11 model.
        You can use a pretrained face model or your own trained weights.
        """
        self.model = model
        self.conf_threshhold = conf_threshold

    def crop_faces(self, image_path, save_dir="cropped_faces"):
        """
        Detects and crops faces from an image.

        Args:
            image_path (str): Path to the input image.
            save_dir (str): Directory to save cropped faces.

        Returns:
            List of cropped face images (numpy arrays).
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or cannot be opened.")

        # Run YOLOv11 inference
        results = self.model(img)

        cropped_faces = []
        os.makedirs(save_dir, exist_ok=True)



        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box[:4])
            face = img[y1:y2, x1:x2]
            conf = float(box.conf.cpu().numpy())

            if face.size == 0 or conf < self.conf_threshhold:
                continue  # skip invalid crops

            cropped_faces.append(face)

            # Save cropped face
            face_path = os.path.join(save_dir, f"face_{i+1}.jpg")
            cv2.imwrite(face_path, face)

        return cropped_faces


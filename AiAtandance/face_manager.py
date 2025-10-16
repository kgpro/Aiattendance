import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any

import requests
# from deepface import DeepFace
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction, models
from django.db.models import Count, Q  # Added Q import here
from django.utils import timezone
from .models import Person, FaceEmbedding, AttendanceLog
import logging
from .facenet_loader import DeepFaceLite as DeepFace


logger = logging.getLogger(__name__)


class FaceEmbeddingManager:

    def __init__(self,
                 username=None,
                 model_name: str = "Facenet512",
                 distance_metric: str = "cosine",
                 threshold: float = 0.6,
                 duplicate_time_window: int = 300):

        """
        Initialize the Django Face Embedding Manager.

        """
        self.user_name=username
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.duplicate_time_window = duplicate_time_window

        # Pre-load the DeepFace model to avoid on-demand downloading
        self.DeepFace=self._preload_deepface_model()

        # Cache for loaded embeddings
        self._embedding_cache = {}
        if  self.user_name:
            self._load_cache()






    def _preload_deepface_model(self):
        """Pre-load the DeepFace model to avoid on-demand downloading during recognition"""
        try:
            print(f"Pre-loading DeepFace model: {self.model_name}")
            # Create a dummy embedding to force model loading
            dummy_array = np.random.rand(100, 100, 3).astype(np.uint8)
            Deepface=DeepFace(
                device="cuda"
            )
            print(f"facenet512 loaded successfully")
            return Deepface
        except Exception as e:
            print(f"Error pre-loading DeepFace model: {str(e)}")

    def _load_cache(self):
        """Load all active embeddings into memory cache for faster recognition."""
        try:
            embeddings = FaceEmbedding.objects.select_related('person').filter(
                is_active=True,
                person__user_name=self.user_name,
                person__is_active=True
            )

            self._embedding_cache = {}
            for embedding_obj in embeddings:
                student_id = embedding_obj.person.student_id

                if student_id not in self._embedding_cache:
                    self._embedding_cache[student_id] = []

                # Store only essential data to reduce memory usage
                self._embedding_cache[student_id].append({
                    'id': embedding_obj.id,
                    'student_id': student_id,
                    'person_name': embedding_obj.person.name,
                    'embedding': embedding_obj.get_embedding(),
                })

            print(f"Loaded {len(embeddings)} embeddings into cache for {len(self._embedding_cache)} people")

        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            self._embedding_cache = {}

    def create_embedding_from_face_crop(self,face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Create face embedding from cropped face image (from YOLO detection).

        Args:
            face_crop: Cropped face image as numpy array (BGR format)

        Returns:
            Face embedding as numpy array or None if processing fails
        """


        try:
            if face_crop is None or face_crop.size == 0:
                return None

            # Ensure minimum size for face recognition
            if face_crop.shape[0] < 32 or face_crop.shape[1] < 32:
                face_crop = cv2.resize(face_crop, (64, 64))

            embedding = self.DeepFace.represent(
                img_array=face_crop,
                enforce_detection=False
            )

            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            else:
                return None

        except Exception as e:
            print(f"Error creating embedding from face crop: {str(e)}")
            return None

    def store_embedding(self,
                        _id,
                        embedding: np.ndarray,
                        image_path: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                       ) -> Optional[FaceEmbedding]:
        """
        Store face embedding in database.

        Args:
            person_name: Name of the person
            embedding: Face embedding as numpy array
            image_path: Optional path to source image
            metadata: Optional metadata dictionary
            **person_kwargs: Additional fields for Person model

        Returns:
            FaceEmbedding object if stored successfully, None otherwise
        """
        try:
            with transaction.atomic():
                person = Person.objects.get(
                id=_id
                )

                # Create embedding object
                face_embedding = FaceEmbedding(
                    person=person,
                    image_path=image_path,
                    metadata=metadata or {}
                )
                face_embedding.set_embedding(embedding)
                face_embedding.save()

                # Update cache
                if person_name not in self._embedding_cache:
                    self._embedding_cache[person_name] = []

                self._embedding_cache[person_name].append({
                    'id': face_embedding.id,
                    'person_id': person.id,
                    'embedding': embedding,
                })

                print(f"Stored embedding for {person.name} with ID {face_embedding.id}")
                return face_embedding

        except Exception as e:
            raise e
            return None



    def recognize_face(self, face_embedding: np.ndarray,
                       save_attendance: bool = True,
                       image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Recognize face by comparing with stored embeddings.

        Args:
            face_embedding: Face embedding to match
            save_attendance: Whether to save attendance log
            image_path: Optional path to recognition image

        Returns:
            Dictionary with match info or None if no match found
        """
        if not self._embedding_cache:
            return None

        best_match = None
        best_distance = float('inf')
        best_embedding_id = None

        try:
            for student_id, embeddings_list in self._embedding_cache.items():  # Iterate by ID
                for embedding_data in embeddings_list:
                    stored_embedding = embedding_data['embedding']

                    # Calculate distance
                    distance = self._calculate_distance(face_embedding, stored_embedding)

                    if distance < best_distance:
                        best_distance = distance
                        best_embedding_id = embedding_data['id']
                        best_match = {
                            'student_id': student_id,  # Use ID
                            'person_name': embedding_data['person_name'],  # Keep name for reference
                            'distance': distance,
                            'confidence': max(0, 1 - (distance / 2)),
                            'embedding_id': embedding_data['id']
                        }

            # Check if best match meets threshold
            if best_match and best_distance <= self.threshold:
                # Check for duplicate attendance
                if save_attendance and not self._is_duplicate_attendance(
                        best_match['student_id'],
                        self.duplicate_time_window
                ):
                    # Get the embedding object for saving attendance
                    try:
                        embedding_obj = FaceEmbedding.objects.get(id=best_embedding_id)
                        self._save_attendance_log(
                            best_match,
                            embedding_obj,
                            image_path
                        )
                    except ObjectDoesNotExist:
                        print(f"Embedding object {best_embedding_id} not found")

                return best_match
            else:
                return None

        except Exception as e:
            print(f"Error during recognition: {str(e)}")
            return None

    def _is_duplicate_attendance(self, student_id: int, time_window: int) -> bool:
        """Check if attendance was already marked within time window."""
        try:
            cutoff_time = timezone.now() - timedelta(seconds=time_window)
            recent_attendance = AttendanceLog.objects.filter(
                person__student_id=student_id,
                timestamp__gte=cutoff_time
            ).exists()
            return recent_attendance
        except Exception as e:
            print(f"Error checking duplicate attendance: {str(e)}")
            return False


    def _save_attendance_log(self, match_info: Dict,
                             embedding_obj: FaceEmbedding,
                             image_path: Optional[str]):
        """Save attendance log to database."""
        try:
            person = Person.objects.get(student_id=match_info['student_id'])
            AttendanceLog.objects.create(
                person=person,
                confidence=match_info['confidence'],
                distance=match_info['distance'],
                embedding_used=embedding_obj,
                image_path=image_path,
                metadata={
                    'model_name': self.model_name,
                    'threshold': self.threshold,
                    'distance_metric': self.distance_metric
                }
            )
            print(f"Attendance logged for {match_info['person_name']}")
        except Exception as e:
            print(f"Error saving attendance log: {str(e)}")

    def get_all_persons(self) :
        """Get all active persons."""
        return Person.objects.filter(is_active=True,user_name=self.user_name).prefetch_related('embeddings')

    def get_person_embeddings(self, _id):
        """Get all embeddings for a specific person."""
        try:
            person = Person.objects.get(id=_id, is_active=True)
            return person.embeddings.filter(is_active=True)
        except ObjectDoesNotExist:
            return []

    def delete_embedding(self, embedding_id: int) -> bool:
        """Delete embedding by ID (soft delete)."""
        try:
            embedding = FaceEmbedding.objects.get(id=embedding_id)
            embedding.is_active = False
            embedding.save()

            # Reload cache
            self._load_cache()
            return True

        except ObjectDoesNotExist:
            print(f"Embedding {embedding_id} not found")
            return False
        except Exception as e:
            print(f"Error deleting embedding: {str(e)}")
            return False

    def _calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        # Ensure numpy arrays
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)

        # Normalize embeddings (improves consistency across models)
        emb1 = emb1 / np.linalg.norm(emb1) if np.linalg.norm(emb1) > 0 else emb1
        emb2 = emb2 / np.linalg.norm(emb2) if np.linalg.norm(emb2) > 0 else emb2

        if self.distance_metric == "cosine":
            return 1 - np.dot(emb1, emb2)
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(emb1 - emb2)
        else:  # default fallback
            return 1 - np.dot(emb1, emb2)


    def update_threshold(self, new_threshold: float):
        """Update recognition threshold."""
        self.threshold = new_threshold
        print(f"Updated recognition threshold to {new_threshold}")


    def recognize_faces_batch(self, face_embeddings: List[np.ndarray],
                              save_attendance: bool = True) -> List[Optional[Dict[str, Any]]]:
        """
        Recognize multiple faces in batch for better performance.

        Args:
            face_embeddings: List of face embeddings to match
            save_attendance: Whether to save attendance logs

        Returns:
            List of match info dictionaries or None for no match
        """
        results = []
        for embedding in face_embeddings:
            results.append(self.recognize_face(embedding, save_attendance))


        return results




    def get_embedding_by_id(self, embedding_id: int) -> Optional[Dict]:
        """
        Get embedding data by ID from cache or database.
        """
        # First check cache
        for person_name, embeddings in self._embedding_cache.items():
            for embedding_data in embeddings:
                if embedding_data['id'] == embedding_id:
                    return embedding_data

        # If not in cache, try database
        try:
            embedding_obj = FaceEmbedding.objects.get(id=embedding_id, is_active=True)
            return {
                'id': embedding_obj.id,
                'person_id': embedding_obj.person.id,
                'embedding': embedding_obj.get_embedding(),
                'person_name': embedding_obj.person.name
            }
        except ObjectDoesNotExist:
            return None

    def create_embeddings_from_batch(self, face_crops):
        """
        Create embeddings for a batch of face crops using DeepFaceLite.
        Returns a list of embeddings (numpy arrays).
        """
        try:
            results = []
            for crop in face_crops:
                reps = self.DeepFace.represent(img_array=crop, enforce_detection=False)
                if reps and "embedding" in reps[0]:
                    results.append(np.array(reps[0]["embedding"], dtype=np.float32))
            return results
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return []

    def reload_cache(self):
        """Manually reload the embedding cache."""
        self._load_cache()
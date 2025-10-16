import json
import asyncio
import cv2
import numpy as np
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from services.model_loader import load_yolo_model
from .face_manager import FaceEmbeddingManager
import time
from concurrent.futures import ThreadPoolExecutor
from asgiref.sync import sync_to_async
import logging
logger = logging.getLogger(__name__)



class FaceDetectionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yolo_model = None
        self.face_manager = None
        self.detector_pool = ThreadPoolExecutor(max_workers=4)
        self.recognizer_pool = ThreadPoolExecutor(max_workers=2)
        self.is_processing = False
        self.frame_counter = 0
        self.stream = None  # Initialize stream attribute
        self.last_receive_time = None
        self.last_send_time = None
        self.last_frame_data = None
        self.user = None
        self.heartbeat_interval = 30  # seconds
        self.detection_queue = asyncio.Queue()
        self.recognition_task = None

    async def connect(self):
        # Extract the user from the scope
        self.user = self.scope.get("user")

        if not self.user or not self.user.is_authenticated:
            # Reject connection if user is not logged in
            await self.close()
            logger.warning("Unauthorized websocket connection attempt.")
            return

        # If authenticated, accept connection
        await self.accept()
        logger.info(f"WebSocket connection established by user: {self.user.username}")

        try:
            # Load models in executor (non-blocking)
            loop = asyncio.get_event_loop()
            self.yolo_model, self.face_manager = await asyncio.wait_for(
                loop.run_in_executor(self.detector_pool, self._load_models),
                timeout=30.0
            )

            logger.info("Models loaded successfully.")

            self.recognition_task = asyncio.create_task(self._recognition_worker())

            await self.send(text_data=json.dumps({
                'type': 'connection_established',
                'message': f'WebSocket connected successfully. Models loaded for {self.user.username}.'
            }))

        except asyncio.TimeoutError:
            logger.error("Model loading timed out")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Model loading timed out. Please try again.'
            }))
            await self.close()

        except Exception as e:
            logger.error(f"Error during connection: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to load models: {str(e)}'
            }))
            await self.close()

    def _load_models(self):
        """Load models synchronously (called in thread pool)"""
        return load_yolo_model(), FaceEmbeddingManager(self.user)
    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with code: {close_code}")

        # Set a flag to prevent new processing
        self.is_processing = True

        # Wait a bit for any ongoing processing to finish
        await asyncio.sleep(5)  # Increased wait time

        # Then shut down the thread pool gracefully
        try:
            if self.recognition_task:
                self.recognition_task.cancel()
            self.detector_pool.shutdown(wait=False, cancel_futures=True)
            self.recognizer_pool.shutdown(wait=False, cancel_futures=True)

            logger.info("Thread pool shut down gracefully")


        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")

        # Clean up any other resources
        if hasattr(self, 'stream') and self.stream:
            for track in self.stream.getTracks():
                track.stop()
        logger.info("WebSocket disconnected and resources cleaned up.")

    async def receive(self, text_data=None, bytes_data=None):
        self.last_receive_time = time.time()
        try:
            if bytes_data:
                # Process binary frame data
                await self.process_frame(bytes_data)
                logger.info("Processed binary frame data.")
            elif text_data is not None:
                # Process JSON messages
                try:
                    # Check if text_data is not None and not empty
                    if text_data and text_data.strip():
                        data = json.loads(text_data)
                        logger.info(f"Received text data: {data}")
                        if data.get('type') == 'heartbeat':
                            await self.send(text_data=json.dumps({
                                'type': 'heartbeat_response',
                                'timestamp': data.get('timestamp')
                            }))
                            logger.info("Heartbeat response sent.")
                    else:
                        logger.warning("Received empty text data")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {text_data}. Error: {e}")
                except Exception as e:
                    logger.error(f"Error processing text data: {e}")
            else:
                logger.warning("Received empty message")
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Processing error: {str(e)}'
            }))

    async def process_frame(self, frame_data):
        if self.is_processing:
            logger.warning("Frame skipped to avoid backlog.")
            return

        # Check if thread pool is still available
        if self.detector_pool._shutdown:
            logger.warning("Thread pool is shut down, cannot process frame")
            return

        self.is_processing = True
        self.last_frame_data = frame_data
        try:
            logger.info(f"Processing frame: {len(frame_data)} bytes")

            # Process the frame in thread pool (CPU-intensive tasks only)
            loop = asyncio.get_event_loop()
            processed_data = await loop.run_in_executor(
                self.detector_pool, self._process_detection_only, frame_data
            )

            if processed_data is None:
                return

            processed_frame, detections = processed_data

            # Send detection results immediately
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info("Frame encoded to base64.")

            processing_time = None
            if self.last_receive_time:
                processing_time = int((time.time() - self.last_receive_time) * 1000)

            # Send detection results immediately
            await self.send(text_data=json.dumps({
                'type': 'detection_results',
                'frame': frame_base64,
                'detections': detections or [],
                'processing_time': processing_time,
                'timestamp': time.time(),
                'frame_id': f"frame_{self.frame_counter}"
            }))


            logger.info(f"Sent detection results for frame {self.frame_counter} with {len(detections)} detections.")
            if detections:
                await self.detection_queue.put((processed_frame, detections))


        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            if not self.detector_pool._shutdown:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Frame processing error: {str(e)}'
                }))
        finally:
            self.is_processing = False
            logger.info("Ready for next frame.")

    def _process_detection_only(self, frame_data):
        """Process frame for detection only (no recognition)"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                logger.warning("Failed to decode image or empty frame")
                return None

            # Create a copy for processing
            processed_frame = frame.copy()

            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            detections = []

            for result in results:
                for box in result.boxes:
                    # Check if box has valid data
                    if box.xyxy is None or len(box.xyxy[0]) != 4:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0]) if box.conf is not None else 0

                    # Only process if confidence is above threshold and box is valid
                    if conf > 0.5 and x2 > x1 and y2 > y1:
                        # Ensure coordinates are within frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        # Check if crop area is valid
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Draw bounding box on processed frame

                        processed_frame = results[0].plot() if len(results) > 0 else frame.copy()

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'embedding_processed': False
                        })

            return processed_frame, detections

        except Exception as e:
            logger.error(f"Error in detection processing: {e}")
            return None

    async def _recognition_worker(self):
        """Background consumer for batched recognition"""
        while True:
            try:
                frame, dets = await self.detection_queue.get()
                try:
                    if not dets:
                        continue

                    # Crop all faces
                    face_crops = []
                    valid_dets = []
                    for det in dets:
                        x1, y1, x2, y2 = det["bbox"]
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop is not None and face_crop.size > 0:
                            face_crops.append(face_crop)
                            valid_dets.append(det)

                    if not face_crops:
                        continue

                    # Run embedding extraction in batch
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        self.recognizer_pool,
                        self.face_manager.create_embeddings_from_batch,  # new batch method
                        face_crops
                    )

                    if embeddings is None or len(embeddings) == 0:
                        continue

                    # Recognize each face
                    for emb, det in zip(embeddings, valid_dets):
                        result = await sync_to_async(self.face_manager.recognize_face)(emb)
                        if result:
                            formatted_result = {
                                "student_id": result.get("student_id", "Unknown"),
                                "person_name": result.get("person_name", "Unknown"),
                                "confidence": float(result.get("confidence", 0)),
                                "distance": float(result.get("distance", float("inf"))),
                                "bbox": det["bbox"],
                                "error": result.get("error", None),
                            }
                            await self.send(text_data=json.dumps({
                                "type": "recognition_results",
                                "recognition": formatted_result,
                                "timestamp": time.time(),
                            }))

                finally:
                    self.detection_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recognition worker error: {e}")



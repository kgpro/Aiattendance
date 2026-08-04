from django.shortcuts import render

from django.views.decorators.http import require_http_methods
from .face_manager import FaceEmbeddingManager
from django.contrib.auth.decorators import login_required

import logging

logger= logging.getLogger(__name__)  # Use module-level logger
# Initialize the face manager
face_manager = FaceEmbeddingManager()


@require_http_methods(["GET"])
def homepage(request):
    logger.info(f"Homepage accessed by user: {request.user.username}")
    return render(request, 'index.html')


@login_required
def detection(request):
    logger.info(f"Detection page accessed by user: {request.user.username}")
    return render(request, 'detection.html')


@login_required
def students(request):
    logger.info(f"Students page accessed by user: {request.user.username}")
    return render(request, 'students.html')














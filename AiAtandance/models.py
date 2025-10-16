from django.db import models
from django.contrib.auth.models import User
import pickle
import base64

class Person(models.Model):
    """Model to store student information"""
    user_name = models.ForeignKey(User, on_delete=models.CASCADE, related_name='students')
    student_id = models.CharField(max_length=15, unique=True, db_index=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.student_id} - {self.name}"

    class Meta:
        ordering = ['name']
        app_label = 'AiAtandance'


class FaceEmbedding(models.Model):
    """Model to store face embeddings for each student"""
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='embeddings')
    embedding_data = models.TextField()  # Base64 encoded pickle data
    image_path = models.CharField(max_length=255, blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def set_embedding(self, embedding_array):
        """Convert numpy array to base64 string for storage"""
        pickled_data = pickle.dumps(embedding_array)
        self.embedding_data = base64.b64encode(pickled_data).decode('utf-8')

    def get_embedding(self):
        """Convert base64 string back to numpy array"""
        pickled_data = base64.b64decode(self.embedding_data.encode('utf-8'))
        return pickle.loads(pickled_data)

    def __str__(self):
        return f"{self.person.student_id} - Embedding {self.id}"

    class Meta:
        ordering = ['-created_at']
        app_label = 'AiAtandance'


class AttendanceLog(models.Model):
    """Model to store attendance records"""
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='attendance_logs')
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField()
    distance = models.FloatField()
    embedding_used = models.ForeignKey(FaceEmbedding, on_delete=models.SET_NULL, null=True, blank=True)
    image_path = models.CharField(max_length=255, blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"{self.person.student_id} - {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']
        app_label = 'AiAtandance'


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
import json
import cv2
import numpy as np
from AiAtandance.face_manager import FaceEmbeddingManager

from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import datetime, timedelta
from AiAtandance.models import Person, AttendanceLog, FaceEmbedding


@login_required
@require_http_methods(["GET"])
def attendance_stats(request):
    today = timezone.now().date()

    user_name = request.user

    total_students = Person.objects.filter(is_active=True, user_name=user_name).count()

    today_attendance = AttendanceLog.objects.filter(timestamp__date=today,person__user_name=user_name).values('person').distinct().count()

    attendance_rate = (today_attendance / total_students * 100) if total_students > 0 else 0

    return JsonResponse({
        'total_students': total_students,
        'today_attendance': today_attendance,
        'attendance_rate': round(attendance_rate, 1)
    })


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class StudentListView(View):
    def get(self, request):
        # Get all active students
        students = Person.objects.filter(is_active=True, user_name=request.user )

        # Get today's date
        today = timezone.now().date()

        student_data = []
        for student in students:
            # Get images count
            images_count = FaceEmbedding.objects.filter(person=student, is_active=True).count()

            # Get last attendance date
            last_attendance = AttendanceLog.objects.filter(person=student).order_by('-timestamp').first()
            last_attendance_date = last_attendance.timestamp if last_attendance else None

            # Check if student has attendance for today
            today_attendance = AttendanceLog.objects.filter(
                person=student,
                timestamp__date=today
            ).first()

            status = "absent"
            if today_attendance:
                status = "present"

            student_data.append({
                'id': student.id,
                'name': student.name,
                'person_id': student.student_id,
                'department': student.department ,
                'email': student.email or 'N/A',
                'images_count': images_count,
                'last_attendance': last_attendance_date.isoformat() if last_attendance_date else None,
                'today_status': status
            })

        return JsonResponse({'students': student_data})


@method_decorator(login_required, name='dispatch')
class StudentDetailView(View):
    def get(self, request, _id):
        try:
            student = Person.objects.get(id=_id, is_active=True)

            # Get enrollment date (created date)
            enrollment_date = student.created_at.date() if student.created_at else 'Unknown'

            # Get images count
            images_count = FaceEmbedding.objects.filter(person=student, is_active=True).count()

            # Check today's attendance status
            today = timezone.now().date()
            today_attendance = AttendanceLog.objects.filter(
                person=student,
                timestamp__date=today
            ).first()

            today_status = "absent"
            if today_attendance:
                today_status = "present"

            # Get attendance data for the last 30 days for chart
            thirty_days_ago = timezone.now() - timedelta(days=30)
            attendance_data = []
            labels = []

            for i in range(30):
                current_date = (timezone.now() - timedelta(days=i)).date()
                labels.insert(0, current_date.strftime('%Y-%m-%d'))

                has_attendance = AttendanceLog.objects.filter(
                    person=student,
                    timestamp__date=current_date
                ).exists()

                attendance_data.insert(0, 1 if has_attendance else 0)

            student_details = {
                'id': student.id,
                'name': student.name,
                'person_id': student.student_id ,
                'department': student.department ,
                'email': student.email,
                'enrollment_date': enrollment_date,
                'images_count': images_count,
                'today_status': today_status,
                'attendance_data': {
                    'labels': labels,
                    'data': attendance_data
                }
            }

            return JsonResponse(student_details)

        except Person.DoesNotExist:
            return JsonResponse({'error': 'Student not found'}, status=404)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class EnrollStudentView(View):
    def post(self, request):
        try:
            # Parse form data
            name = request.POST.get('name')
            student_id = request.POST.get('person_id')
            department = request.POST.get('department')
            email = request.POST.get('email')



            # Check if student already exists
            if Person.objects.filter(student_id=student_id, is_active=True).exists():
                return JsonResponse({'error': 'Student with this student_id already exists'}, status=400)

            # Create new student
            student = Person.objects.create(
                user_name=request.user,
                name=name,
                student_id=student_id,
                department=department,
                email=email
            )
            print("Student created:", student)

            # Process uploaded images
            images = request.FILES.getlist('images')
            if len(images) > 3:
                return JsonResponse({'error': 'Maximum 3 images allowed'}, status=400)

            if len(images) == 0:
                return JsonResponse({'error': 'At least one image is required'}, status=400)

            # Process each image
            for image in images:
                # Read image file
                image_data = image.read()
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    print("Invalid image:", image.name)
                    continue

                face_manager = FaceEmbeddingManager() #user specific face manager

                # Create embedding
                embedding = face_manager.create_embedding_from_face_crop(img)

                if embedding is not None:
                    # Save embedding
                    face_embedding = FaceEmbedding(
                        person=student,
                        image_path=image.name
                    )
                    face_embedding.set_embedding(embedding)
                    face_embedding.save()

            return JsonResponse({
                'success': True,
                'message': 'Student enrolled successfully',
                'student_id': student.id
            })

        except Exception as e:
            return JsonResponse({'error': f'Error enrolling student: {str(e)}'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class UploadImagesView(View):
    def post(self, request):
        try:
            _id = request.POST.get('student_id')

            # Get student
            try:
                student = Person.objects.get(id=_id, is_active=True)
            except Person.DoesNotExist:
                return JsonResponse({'error': 'Student not found'}, status=404)

            # Process uploaded images
            images = request.FILES.getlist('images')
            if len(images) > 3:
                return JsonResponse({'error': 'Maximum 3 images allowed'}, status=400)

            if len(images) == 0:
                return JsonResponse({'error': 'No images provided'}, status=400)

            # Process each image
            for image in images:
                # Read image file
                image_data = image.read()
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                face_manager = FaceEmbeddingManager() #user specific face manager
                # Create embedding
                embedding = face_manager.create_embedding_from_face_crop(img)

                if embedding is not None:
                    # Save embedding
                    face_embedding = FaceEmbedding(
                        person=student,
                        image_path=image.name
                    )
                    face_embedding.set_embedding(embedding)
                    face_embedding.save()

            # Get updated images count
            images_count = FaceEmbedding.objects.filter(person=student, is_active=True).count()

            return JsonResponse({
                'success': True,
                'message': 'Images uploaded successfully',
                'images_count': images_count
            })

        except Exception as e:
            return JsonResponse({'error': f'Error uploading images: {str(e)}'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class StudentAttendanceView(View):
    def get(self, request, _id):
        try:
            student = Person.objects.get(id=_id, is_active=True)

            # Get attendance data for the last 30 days
            thirty_days_ago = timezone.now() - timedelta(days=30)
            attendance_data = []
            labels = []

            for i in range(30):
                current_date = (timezone.now() - timedelta(days=i)).date()
                labels.insert(0, current_date.strftime('%Y-%m-%d'))

                has_attendance = AttendanceLog.objects.filter(
                    person=student,
                    timestamp__date=current_date
                ).exists()

                attendance_data.insert(0, 1 if has_attendance else 0)

            return JsonResponse({
                'labels': labels,
                'data': attendance_data
            })

        except Person.DoesNotExist:
            return JsonResponse({'error': 'Student not found'}, status=404)

    def post(self, request, _id):
        try:
            data = json.loads(request.body)
            status = data.get('status')

            student = Person.objects.get(id=_id, is_active=True)


            today = timezone.now().date()
            existing_attendance = AttendanceLog.objects.filter(
                person=student,
                timestamp__date=today
            ).first()
            if existing_attendance and status == 'present':
                return JsonResponse({'error': 'Attendance already marked for today'}, status=400)

            # Create attendance log for present status
            if status == 'present':
                # Get the first embedding for this student to use as reference
                embedding_obj = FaceEmbedding.objects.filter(
                    person=student,
                    is_active=True
                ).first()

                attendance_log = AttendanceLog.objects.create(
                    person=student,
                    confidence=1.0,
                    distance=0.0,
                    embedding_used=embedding_obj,
                    metadata={
                        'marked_manually': True,
                        'marked_by': request.user.username
                    }
                )
                return JsonResponse({
                    'success': True,
                    'message': 'Attendance marked as present',
                    'attendance_id': attendance_log.id
                })
            else:

                AttendanceLog.objects.filter(person=student, timestamp__date=today).delete()
                return JsonResponse({
                    'success': True,
                    'message': 'Attendance marked as absent'
                })

        except Person.DoesNotExist:
            return JsonResponse({'error': 'Student not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'Error marking attendance: {str(e)}'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(login_required, name='dispatch')
class DeleteStudentView(View):
    def post(self, request, _id):
        try:
            student = Person.objects.get(id=_id, is_active=True)

            FaceEmbedding.objects.filter(person=student).delete()
            AttendanceLog.objects.filter(person=student).delete()
            student.delete()
            return JsonResponse({
                'success': True,
                'message': 'Student deleted successfully'
            })

        except Person.DoesNotExist:
            return JsonResponse({'error': 'Student not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'Error deleting student: {str(e)}'}, status=500)
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .face_manager import FaceEmbeddingManager
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import Person, FaceEmbedding, AttendanceLog
from datetime import datetime, timedelta
from django.db.models import Count, Q


# Initialize the face manager
face_manager = FaceEmbeddingManager()


@csrf_exempt
@require_http_methods(["GET"])
def homepage(request):
    return render(request, 'index.html')


@login_required
def detection(request):
    return render(request, 'detection.html')


@login_required
def students(request):
    return render(request, 'students.html')


@login_required
@require_http_methods(["GET"])
def get_statistics(request):
    user_name = request.user.username
    try:
        total_persons = Person.objects.filter(user_name=user_name, is_active=True).count()
        total_embeddings = FaceEmbedding.objects.filter(is_active=True, person__user_name=user_name).count()
        total_attendance = AttendanceLog.objects.filter(is_active=True, person__user_name=user_name).count()

        # Recent attendance (last 24 hours)
        recent_cutoff = timezone.now() - timedelta(hours=24)
        recent_attendance = AttendanceLog.objects.filter(
            timestamp__gte=recent_cutoff
        ).count()

        # Top persons by embeddings - FIXED: Using Q object correctly
        top_persons = Person.objects.filter(
            is_active=True,
            user_name=user_name
        ).annotate(
            embedding_count=Count('embeddings', filter=Q(person__user_name=user_name, embeddings__is_active=True))
        ).order_by('-embedding_count')[:5]

        stats = {

            'total_persons': total_persons,
            'total_embeddings': total_embeddings,
            'total_attendance_logs': total_attendance,
            'recent_attendance_24h': recent_attendance,
            'model_name': "facenet512",
            'distance_metric': "cosine",
            'threshold': 0.4,
            'top_persons': [(p.name, p.embedding_count) for p in top_persons]

        }
        return JsonResponse(stats)
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
        return {}




@login_required
@require_http_methods(["GET"])
def filtered_attendance(request):
    date_filter = request.GET.get('date', '')
    status_filter = request.GET.get('status', 'all')
    search_filter = request.GET.get('search', '')

    # Build query
    query = Q()

    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, '%Y-%m-%d').date()
            query &= Q(timestamp__date=filter_date)
        except ValueError:
            pass

    if search_filter:
        query &= Q(person__name__icontains=search_filter)

    attendance_logs = AttendanceLog.objects.filter(query).select_related('person')

    if status_filter == 'present':
        attendance_logs = attendance_logs  # Assuming all logs are 'present' for simplicity

    data = [{
        'student_id': log.person.employee_id or 'N/A',
        'name': log.person.name,
        'time': log.timestamp.strftime('%H:%M:%S'),
        'status': 'Present',
        'confidence': f"{log.confidence * 100:.1f}%"
    } for log in attendance_logs.order_by('-timestamp')[:50]]  # Limit to 50 most recent

    return JsonResponse({'attendance': data})


@login_required
@require_http_methods(["GET"])
def attendance_stats(request):
    today = timezone.now().date()

    total_students = Person.objects.filter(is_active=True).count()

    today_attendance = AttendanceLog.objects.filter(
        timestamp__date=today
    ).values('person').distinct().count()

    attendance_rate = (today_attendance / total_students * 100) if total_students > 0 else 0

    return JsonResponse({
        'total_students': total_students,
        'today_attendance': today_attendance,
        'attendance_rate': round(attendance_rate, 1)
    })














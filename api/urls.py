from django.urls import path
from . import views

app_name = "api"

urlpatterns = [

    path('attendance/stats/', views.attendance_stats, name='attendance_stats'),

    # Student list and details
    path("students/all", views.StudentListView.as_view(), name="student-list"),

    path("students/<int:_id>/", views.StudentDetailView.as_view(), name="student-detail"),

    # Enrollment & image uploads
    path("enroll/", views.EnrollStudentView.as_view(), name="enroll-student"),

    path("upload-images/", views.UploadImagesView.as_view(), name="upload-images"),

    # Attendance (per student)
    path("students/<int:_id>/attendance/", views.StudentAttendanceView.as_view(), name="mark-attendance"),

    # Delete student
    path("students/<int:_id>/delete/", views.DeleteStudentView.as_view(), name="delete-student"),

]

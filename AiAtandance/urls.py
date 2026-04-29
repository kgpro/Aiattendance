from django.contrib import admin
from django.urls import path, include
from AiAtandance import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('auth/', include('authentication.urls')),

    path('api/', include('api.urls')),

    path('',views.homepage, name='homepage'),

    path('detection/',views.detection ,name='detection'),


    path("students/",views.students, name="students"),

    # path(staudents/)

]

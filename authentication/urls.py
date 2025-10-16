from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'authentication'
urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='authentication/login.html'), name='login'),
    path('logout/', views.custom_logout, name='logout'),
]
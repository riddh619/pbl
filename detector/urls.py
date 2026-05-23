from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('predict/', views.predict, name='predict'),
    path('chat/', views.chat, name='chat'),
    path('about/', views.about, name='about'),
]

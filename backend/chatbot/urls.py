from django.urls import path

from . import views

app_name = 'chatbot'

urlpatterns = [
    path('queries/', views.QueryAPIView.as_view()),
]
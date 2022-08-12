from django.urls import path
from . import views

urlpatterns = [
    path("", views.Homepage.as_view(), name="homepage"),
    path("register/", views.Register.as_view(), name="register"),
    path("upload/<slug:patient_slug>/", views.upload, name="upload"),
    path("result/<slug:test_slug>/", views.result, name="result"),
    
]
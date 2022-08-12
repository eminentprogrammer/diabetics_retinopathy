from django.shortcuts import render, redirect
from django.views.generic import View
from .forms import RegisterForm
from .models import Test
from django.http import HttpResponse
from joblib import load
from django.conf import settings
import PIL

from .process import process_img



def model():
    pass 

class Homepage(View):
    def get(self, request):
        # <view logic>
        return render(request, "homepage.html", locals())

class Register(View):
    form = RegisterForm()
    def get(self, request):
        form = self.form
        return render(request, "diabetics/register.html", locals())

    def post(self, request):
        form = RegisterForm(request.POST)
        fullname = request.POST['fullname']
        email = request.POST['email']

        if form.is_valid():
            instance = form.save(commit=False)
            instance.save()
            patient_slug = instance.slug

        return redirect("upload", patient_slug)



def upload(request, patient_slug):
    if not Test.objects.filter(slug=patient_slug).exists():
        return HttpResponse("Patient does not exist !!!")

    if request.POST:
        patient_instance = Test.objects.get(slug=patient_slug)
        img = request.FILES['right_eye']
        patient_instance.uploaded_data = img
        patient_instance.save()
        response = process_img(patient_instance)

        return redirect("result", patient_instance.slug)
    return render(request, "diabetics/upload_singles.html", locals())


def result(request, test_slug=None):
    patient_instance = Test.objects.get(slug=test_slug)
    
    return render(request, "diabetics/results.html", locals())

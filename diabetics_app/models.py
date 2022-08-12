from django.db import models
import random
from uuid import uuid4
from django.urls import reverse

def TEST_IMAGE_LOCATION(instance, filename):
    path = f"diabetics_test/{instance.fullname.replace(' ','_')}/{filename}"
    return path


class Test(models.Model):
    fullname    = models.CharField(max_length=500)
    email       = models.EmailField()
    
    uploaded_data    = models.ImageField(upload_to=TEST_IMAGE_LOCATION, null=True, blank=True)
    generated_data   = models.ImageField(upload_to=TEST_IMAGE_LOCATION, null=True, blank=True)

    result      = models.CharField(max_length=500, blank=True, null=True)

    slug        = models.SlugField(null=True, blank=True)
    tested_on   = models.DateField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.slug is None:
            self.slug = uuid4()
        super().save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse("result", args=[(self.slug)])

    def __str__(self):
        return self.fullname
    
    def get_data_url(self):
        return self.uploaded_data

    class Meta:
        verbose_name_plural = "Patients' Record"
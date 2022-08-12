from django.contrib import admin
from .models import Test


class TestAdmin(admin.ModelAdmin):
    pass

admin.site.register(Test)
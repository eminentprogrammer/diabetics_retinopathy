from django import forms
from .models import Test

class RegisterForm(forms.ModelForm):
    fullname = forms.CharField(label="", widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Patient Fullname'}))
    email    = forms.EmailField(label="", widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Email address'}))

    class Meta:
        model = Test
        fields = ('fullname','email',)

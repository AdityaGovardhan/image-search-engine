from . import models
from django.forms import ModelForm


class Task1Form(ModelForm):
    class Meta:
        model = models.Task1Model


class Task2Form(ModelForm):
    class Meta:
        model = models.Task2Model


class Task3Form(ModelForm):
    class Meta:
        model = models.Task3Model


class Task4Form(ModelForm):
    class Meta:
        model = models.Task4Model


class Task5Form(ModelForm):
    class Meta:
        model = models.Task5Model


class Task6Form(ModelForm):
    class Meta:
        model = models.Task6Model

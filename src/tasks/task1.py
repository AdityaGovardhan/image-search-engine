import sys

sys.path.insert(0, 'src')

from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from backend.task1_classifier import Task1_Classifier
import pprint
from django.http import HttpResponse
import json


class Task1(CreateView):
    model = models.Task1Model
    fields = ('number_of_latent_semantics', 'dataset',)
    template_name = 'task1.html'

    def get_context_data(self, **kwargs):
        context = super(Task1, self).get_context_data(**kwargs)
        context.update({'task1_page': 'active'})
        return context


def execute_task1(request):
    k = int(request.POST.get("number_of_latent_semantics"))
    dataset = request.POST.get("dataset")

    labelled_folder_path = "/Labelled/" + dataset
    unlabelled_folder_path = "/Unlabelled/" + dataset
    task1_classifier_obj = Task1_Classifier()
    prediction = task1_classifier_obj.get_label_for_folder(labelled_folder_path, unlabelled_folder_path, k)

    return render(request, 'visualize_images.html', {'images': prediction, "from_task": "task1"})

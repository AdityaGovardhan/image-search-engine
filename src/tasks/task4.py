from django.shortcuts import render
from django.views.generic import CreateView
from django.http import HttpResponse
from src import models
import json
from classifiers.ppr_based_classifier import PPRClassifier

class Task4(CreateView):
    model = models.Task4Model
    fields = ('classifier',)
    template_name = 'task4.html'

    def get_context_data(self, **kwargs):
        context = super(Task4, self).get_context_data(**kwargs)
        context.update({'task4_page': 'active'})
        return context



def execute_task4(request):
    similar_objects = {}
    ppr_obj = PPRClassifier()
    images_with_labels = ppr_obj.get_predicted_labels("/Labelled/Set2", "/Unlabelled/Set2")

    return render(request, 'visualize_images.html', {'images': images_with_labels, "from_task": "task4"})
    # return HttpResponse('You received a response'+json.dumps(similar_objects), status=200)

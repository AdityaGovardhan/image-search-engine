from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from django.http import HttpResponse
import json



class Task3(CreateView):
    model = models.Task3Model
    fields = ('most_similar_images',)
    template_name = 'task3.html'

    def get_context_data(self, **kwargs):
        context = super(Task3, self).get_context_data(**kwargs)
        context.update({'task3_page': 'active'})
        return context


def execute_task3(request):
    print(request.__dict__)
    print(request.method)
    similar_objects = {}
    return HttpResponse('You received a response'+json.dumps(similar_objects), status=200)
    # pass

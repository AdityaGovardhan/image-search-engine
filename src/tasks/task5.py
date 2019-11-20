from django.shortcuts import render
from django.views.generic import CreateView
from src import models


class Task5(CreateView):
    model = models.Task5Model
    fields = ('number_of_layers', 'number_of_hashes_per_layer', 'set_of_vectors', 'query_image', 'most_similar_images')
    template_name = 'task5.html'

    def get_context_data(self, **kwargs):
        context = super(Task5, self).get_context_data(**kwargs)
        context.update({'task5_page': 'active'})
        return context



def execute_task5():
    pass

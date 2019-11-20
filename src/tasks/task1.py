from django.shortcuts import render
from django.views.generic import CreateView
from src import models


class Task1(CreateView):
    model = models.Task1Model
    fields = ('number_of_latent_semantics',)
    template_name = 'task1.html'

    def get_context_data(self, **kwargs):
        context = super(Task1, self).get_context_data(**kwargs)
        context.update({'task1_page': 'active'})
        return context



def execute_task1():
    pass

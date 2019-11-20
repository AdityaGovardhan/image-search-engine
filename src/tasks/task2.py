from django.shortcuts import render
from django.views.generic import CreateView
from src import models


class Task2(CreateView):
    model = models.Task2Model
    fields = ('number_of_clusters',)
    template_name = 'task2.html'

    def get_context_data(self, **kwargs):
        context = super(Task2, self).get_context_data(**kwargs)
        context.update({'task2_page': 'active'})
        return context



def execute_task2():
    pass

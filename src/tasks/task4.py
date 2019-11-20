from django.shortcuts import render
from django.views.generic import CreateView
from src import models


class Task4(CreateView):
    model = models.Task4Model
    fields = ('classifier',)
    template_name = 'task4.html'

    def get_context_data(self, **kwargs):
        context = super(Task4, self).get_context_data(**kwargs)
        context.update({'task4_page': 'active'})
        return context



def execute_task4():
    pass

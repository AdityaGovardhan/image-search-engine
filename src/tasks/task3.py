import sys
sys.path.insert(0, 'src')

from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from django.http import HttpResponse
import json


from backend.pageRank import PageRank
from django.shortcuts import render
from backend.utils import get_pickle_directory

class Task3(CreateView):
    model = models.Task3Model
    fields = ('most_similar_images',)
    template_name = 'task3.html'

    def get_context_data(self, **kwargs):
        context = super(Task3, self).get_context_data(**kwargs)
        context.update({'task3_page': 'active'})
        return context


def execute_task3(request):
    similar_objects = {}
    pg_obj = PageRank()
    #Query 1 use the images from Labelled/Set2
    # dominant_images = pg_obj.get_K_dominant_images(5, 10, ["Hand_0008333.jpg", "Hand_0006183.jpg", "Hand_0000074.jpg"])
    dominant_images = pg_obj.get_K_dominant_images(5, 20, ['Hand_0009002.jpg', 'Hand_0008128.jpg', 'Hand_0008662.jpg'], "/Labelled/Set2")
    # dominant_images = pg_obj.get_K_dominant_images(5, 10, ['Hand_0009446.jpg'], "/Labelled/Set2")
    # dominant_images = pg_obj.get_K_dominant_images(32, 20, ['Hand_0011362.jpg', 'Hand_0008662.jpg', 'Hand_0011505.jpg'],
    #                                                "/Labelled/Set2")
    # return HttpResponse('You received a response'+json.dumps(similar_objects), status=200)
    return render(request, 'visualize_images.html', {'images': dominant_images})
    # pass

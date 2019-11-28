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
    fields = ('k', 'K', 'folder_name', 'user_images',)
    template_name = 'task3.html'

    def get_context_data(self, **kwargs):
        context = super(Task3, self).get_context_data(**kwargs)
        context.update({'task3_page': 'active'})
        return context

def execute_task3(request):
    k = int(request.POST.get('k'))
    K = int(request.POST.get('K'))
    folder_name = request.POST.get('folder_name')
    user_images = request.POST.get('user_images').replace(" ", "").split(",")

    K = K + 3  # Just to give a slack for the input images

    if (folder_name[0] != '/'):
        folder_name = "/" + folder_name

    pg_obj = PageRank()
    #Query 1 use the images from Labelled/Set2
    # dominant_images = pg_obj.get_K_dominant_images(5, 10, ["Hand_0008333.jpg", "Hand_0006183.jpg", "Hand_0000074.jpg"])
    # dominant_images = pg_obj.get_K_dominant_images(5, 20, ['Hand_0009002.jpg', 'Hand_0008128.jpg', 'Hand_0008662.jpg'], "/Labelled/Set2")
    # dominant_images = pg_obj.get_K_dominant_images(5, 10, ['Hand_0008333.jpg'], "/Labelled/Set2")
    # dominant_images = pg_obj.get_K_dominant_images(32, 20, ['Hand_0011362.jpg', 'Hand_0008662.jpg', 'Hand_0011505.jpg'],
    #                                                "/Labelled/Set2")
    #Output of Sample Query 1 Task 3
    dominant_images = pg_obj.get_K_dominant_images(k, K, user_images,folder_name)
    # return HttpResponse('You received a response'+json.dumps(similar_objects), status=200)
    return render(request, 'visualize_images.html', {'images': dominant_images, "from_task": "task3"})

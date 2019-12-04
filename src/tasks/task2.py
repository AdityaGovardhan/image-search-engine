import sys

sys.path.insert(0, 'src')

from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from image_clustering import Image_Clustering


class Task2(CreateView):
    model = models.Task2Model
    fields = ('number_of_clusters','dataset',)
    template_name = 'task2.html'

    def get_context_data(self, **kwargs):
        context = super(Task2, self).get_context_data(**kwargs)
        context.update({'task2_page': 'active'})
        return context


def execute_task2(request):
    no_of_clusters = int(request.POST.get("number_of_clusters"))
    dataset = request.POST.get("dataset")

    labelled_folder_path = "/Labelled/" + dataset
    unlabelled_folder_path = "/Unlabelled/" + dataset

    clustering_obj = Image_Clustering()
    points_in_cluster, prediction = clustering_obj.cluster_images(no_of_clusters, labelled_folder_path, unlabelled_folder_path)

    return render(request, 'task2_visualize.html', {'cluster_images': points_in_cluster,'labelled_images': prediction, "from_task": "task2"})

from django.shortcuts import render
from django.views.generic import CreateView
from django.http import HttpResponse
from src import models
import json
from classifiers.ppr_based_classifier import PPRClassifier
from classifiers.classifier_caller import ClassifierCaller
import time


class Task4(CreateView):
    model = models.Task4Model
    fields = ('classifier', 'labelled_folder_name', 'unlabelled_folder_name',)
    template_name = 'task4.html'

    def get_context_data(self, **kwargs):
        context = super(Task4, self).get_context_data(**kwargs)
        context.update({'task4_page': 'active'})
        return context


def execute_task4(request):
    labelled_folder_path = request.POST.get("labelled_folder_name")
    unlabelled_folder_path = request.POST.get("unlabelled_folder_name")

    classifier = request.POST.get("classifier")

    images_with_labels = [()]
    ppr_obj = PPRClassifier()
    accuracy = 0
    if labelled_folder_path[0] != '/':
        labelled_folder_path = "/" + labelled_folder_path
    if unlabelled_folder_path[0] != '/':
        unlabelled_folder_path = "/" + unlabelled_folder_path

    # TODO ppr align with svm and dtl
    if classifier == "Personalized Page Rank":
        images_with_labels, accuracy = ppr_obj.get_predicted_labels(labelled_folder_path, unlabelled_folder_path)

        return render(request, 'visualize_images.html', {'images': images_with_labels,
                                                         "from_task": "task4", "accuracy": accuracy,
                                                         "classifier": classifier})

    else:
        classifier_caller = ClassifierCaller(classifier, labelled_folder_path[-4:].lower(),
                                             unlabelled_folder_path[-4:].lower())
        classifier_caller.call_classifier()
        time.sleep(4)
        result, images_with_labels = classifier_caller.get_result()
        # TODO
        return render(request, 'visualize_images.html',
                      {'images': images_with_labels, "from_task": "task4",
                       "accuracy": result['accuracy'], "classifier": classifier})

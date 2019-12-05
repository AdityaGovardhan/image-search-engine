from django.shortcuts import render
from django.views.generic import CreateView
from src import models
import json
from classifiers.ppr_based_classifier import PPRClassifier
from classifiers.classifier_caller import ClassifierCaller
import time


class Task4(CreateView):
    model = models.Task4Model
    fields = ('classifier', 'labelled_dataset', 'unlabelled_dataset', 'kernel', 'number_of_clusters')
    template_name = 'task4.html'

    def get_context_data(self, **kwargs):
        context = super(Task4, self).get_context_data(**kwargs)
        context.update({'task4_page': 'active'})
        return context


def execute_task4(request):
    labelled_dataset = request.POST.get("labelled_dataset")
    unlabelled_dataset = request.POST.get("unlabelled_dataset")
    number_of_clusters = request.POST.get("number_of_clusters")
    kernel = request.POST.get("kernel")
    classifier = request.POST.get("classifier")
    if classifier == "Personalized Page Rank":
        ppr_obj = PPRClassifier()
        labelled_folder_path = "/Labelled/" + labelled_dataset
        unlabelled_folder_path = "/Unlabelled/" + unlabelled_dataset
        images_with_labels, accuracy = ppr_obj.get_predicted_labels(labelled_folder_path, unlabelled_folder_path)

        return render(request, 'visualize_images.html', {'images': images_with_labels,
                                                         "from_task": "task4", "accuracy": accuracy,
                                                         "classifier": classifier})
    else:
        classifier_caller = ClassifierCaller(classifier, labelled_dataset, unlabelled_dataset, kernel, number_of_clusters)
        classifier_caller.call_classifier()
        # time.sleep(4)
        result, images_with_labels = classifier_caller.get_result()
        if result:
            return render(request, 'visualize_images.html',
                          {'images': images_with_labels, "from_task": "task4",
                           "accuracy": result['accuracy']*100, "classifier": classifier})
        else:
            return render(request, 'visualize_images.html',
                          {'images': images_with_labels, "from_task": "task4",
                           "accuracy": "Images not available in 11k Dataset", "classifier": classifier})

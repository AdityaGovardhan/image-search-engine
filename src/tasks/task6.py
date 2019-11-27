from django.shortcuts import render
from django.views.generic import CreateView
from relevance_feedback import RelevanceFeedback
from src import models
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json


class Task6(CreateView):
    model = models.Task6Model
    fields = ('relevance_feedback',)
    template_name = 'task6.html'

    def get_context_data(self, **kwargs):
        context = super(Task6, self).get_context_data(**kwargs)
        context.update({'task6_page': 'active'})
        return context


def execute_task6(request):
    rf = RelevanceFeedback()
    m = int(request.POST.get('most_similar_images'))
    q_name = query_image=request.POST.get('query_image')
    q = rf.database_connection.get_feature_data_for_image('histogram_of_gradients', q_name)
    obj_feature_matrix = rf.database_connection.get_object_feature_matrix_from_db('histogram_of_gradients')
    data_matrix = obj_feature_matrix['data_matrix']
    n_i = rf.calculate_n_i(D_matrix=data_matrix)
    initial_list_images = rf.calculate_initial_prob_similarity(D_matrix=data_matrix, images=obj_feature_matrix['images'], n_i=n_i)
    initial_list_images  = initial_list_images[:m]
    return render(request, 'visualize_images.html', {'images': initial_list_images, "from_task": "task6"})


@csrf_exempt
def process_feedback(request):
    m = 5
    rf = RelevanceFeedback()
    relevant = request.POST.get("relevant[]")
    irrelevant = request.POST.get("irrelevant[]")
    obj_feature_matrix = rf.database_connection.get_object_feature_matrix_from_db('histogram_of_gradients')
    data_matrix = obj_feature_matrix['data_matrix']
    n_i = rf.calculate_n_i(D_matrix=data_matrix)

    relevant = json.loads(relevant)

    new_rank_list = rf.calculate_feedback_prob_similarity(D_matrix=data_matrix, images=obj_feature_matrix['images'], relevant_items=relevant,
                                                            n_i=n_i)

    new_rank_list = new_rank_list[:m]
    # return HttpResponse('You received a response', status=200)
    return render(request, 'visualize_images.html', {'images': new_rank_list, "from_task": "task6"})
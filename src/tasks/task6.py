from django.shortcuts import render
from django.views.generic import CreateView
from relevance_feedback import RelevanceFeedback
from src import models
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from utils import read_from_pickle


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
    rel_feedback_type = request.POST.get('relevance_feedback')
    m = 5
    q_name = 'Hand_0000012.jpg'
    q = rf.database_connection.get_feature_data_for_image('histogram_of_gradients', q_name)
    obj_feature_matrix = rf.database_connection.get_object_feature_matrix_from_db('histogram_of_gradients')
    data_matrix = obj_feature_matrix['data_matrix']

    if rel_feedback_type == 'Probabilistic':
        n_i = rf.calculate_n_i(D_matrix=data_matrix)
        initial_list_images = rf.calculate_initial_prob_similarity(D_matrix=data_matrix,
                                                                   images=obj_feature_matrix['images'], n_i=n_i)
        initial_list_images = initial_list_images[:m]
        return render(request, 'visualize_images.html',
                      {'images': initial_list_images, "from_task": "task6", "rel_type": rel_feedback_type, "q": q_name,
                       "t": m})

    elif rel_feedback_type == 'Support Vector Machine':
        print('SVM')
        init_ranking, Vt = rf.get_init_ranking(obj_feature_matrix=obj_feature_matrix, q=q)
        return render(request, 'visualize_images.html',
                      {'images': init_ranking, "from_task": "task6", "rel_type": rel_feedback_type, "q": q_name,
                       "t": m})
        # new_rank_list=rf.get_SVM_based_feedback(init_rank_list=init_ranking,q=q,q_name=q_name,Vt=Vt)

    elif rel_feedback_type == 'Decision Tree Classifier':
        print('DTC')
        init_ranking, Vt = rf.get_init_ranking(obj_feature_matrix=obj_feature_matrix, q=q)
        return render(request, 'visualize_images.html',
                      {'images': init_ranking, "from_task": "task6", "rel_type": rel_feedback_type, "q": q_name,
                       "t": m})

    elif rel_feedback_type == 'Personalized Page Rank':
        print('PPR')
        init_ranking, Vt = rf.get_init_ranking(obj_feature_matrix=obj_feature_matrix, q=q)
        return render(request, 'visualize_images.html',
                      {'images': init_ranking, "from_task": "task6", "rel_type": rel_feedback_type, "q": q_name,
                       "t": m})


@csrf_exempt
def process_feedback(request):
    rf = RelevanceFeedback()
    relevant = request.POST.get("relevant[]")
    irrelevant = request.POST.get("irrelevant[]")
    rel_type = json.loads(request.POST.get("rel_type"))
    m = int(request.POST.get("t"))

    q_name=json.loads(request.POST.get("q"))
    # obj_feature_matrix = rf.database_connection.get_object_feature_matrix_from_db('histogram_of_gradients')
    obj_similar_thousand_names = read_from_pickle('test_dataset.pickle')
    obj_similar_thousand_names=[x[0] for x in obj_similar_thousand_names]
    obj_feature_matrix = rf.database_connection.HOG_descriptor_from_image_ids(image_ids = obj_similar_thousand_names)
    data_matrix = obj_feature_matrix['data_matrix']
    new_rank_list=[]
    relevant=json.loads(relevant)
    irrelevant=json.loads(irrelevant)
    q=rf.database_connection.get_feature_data_for_image('histogram_of_gradients',q_name)
    # Vt=rf.get_Vt(obj_feature_matrix=obj_feature_matrix)

    if rel_type == 'Probabilistic':
        n_i = rf.calculate_n_i(D_matrix=data_matrix)
        new_rank_list = rf.calculate_feedback_prob_similarity(D_matrix=data_matrix, images=obj_feature_matrix['images'],
                                                              relevant_items=relevant,
                                                              n_i=n_i)

        new_rank_list = new_rank_list[:m]

    elif rel_type == 'Support Vector Machine':
        new_rank_list=rf.get_SVM_based_feedback(q=q,rel_items=relevant,irl_items=irrelevant,obj_feature_matrix=obj_feature_matrix,m=m)
        # new_rank_list=rf.get_SVM_based_feedback(q=q,Vt=Vt,rel_items=relevant,irl_items=irrelevant,obj_feature_matrix=obj_feature_matrix,m=m)

    elif rel_type == 'Decision Tree Classifier':
        new_rank_list=rf.get_DTC_based_feedback(q=q,rel_items=relevant,irl_items=irrelevant,obj_feature_matrix=obj_feature_matrix,m=m)

    elif rel_type == 'Personalized Page Rank':
        new_rank_list=rf.get_PPR_based_feedback(q=q,rel_items=relevant,irl_items=irrelevant,obj_feature_matrix=obj_feature_matrix,m=m)
    else:
        new_rank_list.append(('Please select a relevance feedback type and start again from task 5', '0'))

    return render(request, 'visualize_images.html',
                  {'images': new_rank_list, "from_task": "task6", "rel_type": rel_type, "q": q_name, "t": m})

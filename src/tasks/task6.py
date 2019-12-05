from django.shortcuts import render
from django.views.generic import CreateView
from relevance_feedback import RelevanceFeedback
from src import models
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from utils import read_from_pickle, save_to_pickle
from database_connection import DatabaseConnection
from singular_value_decomposition import SingularValueDecomposition
import numpy as np



class Task6(CreateView):
    model = models.Task6Model
    fields = ('query_image', 'most_similar_images', 'relevance_feedback')
    template_name = 'task6.html'

    def get_context_data(self, **kwargs):
        context = super(Task6, self).get_context_data(**kwargs)
        context.update({'task6_page': 'active'})
        return context


def execute_task6(request):
    query_image = request.POST.get('query_image')
    most_similar_images = int(request.POST.get('most_similar_images'))
    relevance_feedback = request.POST.get('relevance_feedback')
    lsh = read_from_pickle('lsh_model')
    db_connection = DatabaseConnection()
    image_vector = db_connection.get_feature_data_for_image('histogram_of_gradients', query_image)
    image_vector = np.asarray(image_vector.flatten())

    if read_from_pickle('all_img_features_LSH.pickle') != None:
        all_image_hog_features = read_from_pickle('all_img_features_LSH.pickle')
    else:
        all_image_hog_features = db_connection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
        save_to_pickle(all_image_hog_features,'all_img_features_LSH.pickle')
    #SVD on hog features
    if(read_from_pickle('svd_hog_lsh.pickle')!=None):
        transformed_data = read_from_pickle('svd_hog_lsh.pickle')
        transformed_data = transformed_data['data_matrix']
    else:
        svd = SingularValueDecomposition()
        transformed_data = svd.get_transformed_data(all_image_hog_features['data_matrix'],400)
        save_to_pickle({"data_matrix":transformed_data,"images":all_image_hog_features['images']},'svd_hog_lsh.pickle')

    index_of_query_image = (all_image_hog_features['images']).index(query_image)
    image_vector = transformed_data[index_of_query_image]
    image_vector = np.asarray(image_vector.flatten())

    new_obj={}
    new_obj['data_matrix'] = transformed_data
    new_obj['images'] = all_image_hog_features['images']
    (sorted_k_values, result_stats) = lsh.find_ksimilar_images(k=most_similar_images, image_vector=image_vector,
                                                               all_image_hog_features=new_obj)

    # Now getting a bigger test dataset for relevance feedback
    if relevance_feedback == "Probabilistic":
        (test_dataset, result_stats) = lsh.find_ksimilar_images(k=10 + most_similar_images, image_vector=image_vector,
                                                            all_image_hog_features=new_obj)
    else:
        (test_dataset, result_stats) = lsh.find_ksimilar_images(k=200 + most_similar_images, image_vector=image_vector,
                                                                all_image_hog_features=new_obj)

    save_to_pickle(test_dataset, 'test_dataset.pickle')
    print(sorted_k_values[:most_similar_images])
    return render(request, 'visualize_images.html',
                  {'images': sorted_k_values[:most_similar_images], "from_task": "task5",
                   'rel_type': relevance_feedback,
                   "q": query_image, "t": most_similar_images,
                   "num_total": result_stats['total'], "num_unique": result_stats['unique']})


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

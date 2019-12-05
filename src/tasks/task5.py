from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from locality_sensitive_hashing import LSH
from database_connection import DatabaseConnection
import numpy as np
from utils import save_to_pickle,read_from_pickle
from singular_value_decomposition import SingularValueDecomposition


class Task5(CreateView):
    model = models.Task5Model
    fields = ('number_of_layers', 'number_of_hashes_per_layer',)
    template_name = 'task5.html'

    def get_context_data(self, **kwargs):
        context = super(Task5, self).get_context_data(**kwargs)
        context.update({'task5_page': 'active'})
        return context


def execute_task5(request):
    l = int(request.POST.get('number_of_layers'))
    k = int(request.POST.get('number_of_hashes_per_layer'))
    lsh = LSH(k=k, l=l)
    dbconnection = DatabaseConnection()

    if read_from_pickle('all_img_features_LSH.pickle') != None:
        all_image_hog_features = read_from_pickle('all_img_features_LSH.pickle')
    else:
        all_image_hog_features = dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
        save_to_pickle(all_image_hog_features,'all_img_features_LSH.pickle')
    #SVD on hog features
    if(read_from_pickle('svd_hog_lsh.pickle')!=None):
        transformed_data = read_from_pickle('svd_hog_lsh.pickle')
        transformed_data = transformed_data['data_matrix']
    else:
        svd = SingularValueDecomposition()
        transformed_data = svd.get_transformed_data(all_image_hog_features['data_matrix'],400)
        save_to_pickle({"data_matrix":transformed_data,"images":all_image_hog_features['images']},'svd_hog_lsh.pickle')

    # index_of_query_image = (all_image_hog_features['images']).index(query_image)
    # image_vector = transformed_data[index_of_query_image]
    bit_map = lsh.generate_representation_for_all_layers(transformed_data,all_image_hog_features['images'])

    save_to_pickle(lsh, 'lsh_model')
    return render(request, 'task5a_output.html')

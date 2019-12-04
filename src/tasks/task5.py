from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from locality_sensitive_hashing import LSH
from database_connection import DatabaseConnection
from utils import save_to_pickle, read_from_pickle


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
        save_to_pickle(all_image_hog_features, 'all_img_features_LSH.pickle')

    bit_map = lsh.generate_representation_for_all_layers(all_image_hog_features['data_matrix'],
                                                         all_image_hog_features['images'])

    save_to_pickle(lsh, 'lsh_model')
    return render(request, 'task5a_output.html')

from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from locality_sensitive_hashing import LSH
from database_connection import DatabaseConnection
import numpy as np
from utils import save_to_pickle,read_from_pickle


class Task5(CreateView):
    model = models.Task5Model
    fields = ('number_of_layers', 'number_of_hashes_per_layer', 'set_of_vectors', 'query_image', 'most_similar_images',
              'relevance_feedback')
    template_name = 'task5.html'

    def get_context_data(self, **kwargs):
        context = super(Task5, self).get_context_data(**kwargs)
        context.update({'task5_page': 'active'})
        return context


def execute_task5(request):
    l = int(request.POST.get('number_of_layers'))
    k = int(request.POST.get('number_of_hashes_per_layer'))
    query_image = request.POST.get('query_image')
    t = int(request.POST.get('most_similar_images'))
    rel_type = request.POST.get('relevance_feedback')
    lsh = LSH(k=k, l=l)
    dbconnection = DatabaseConnection()

    if(read_from_pickle('all_img_features_LSH.pickle')!=None):
        all_image_hog_features = read_from_pickle('all_img_features_LSH.pickle')
    else:
        all_image_hog_features = dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
        save_to_pickle(all_image_hog_features,'all_img_features_LSH.pickle')
    
    bit_map = lsh.generate_representation_for_all_layers(all_image_hog_features['data_matrix'],all_image_hog_features['images'])
    image_vector = dbconnection.get_feature_data_for_image('histogram_of_gradients',query_image)
    image_vector = np.asarray(image_vector.flatten())
    
    (sorted_k_values,result_stats) = lsh.find_ksimilar_images(k=t,image_vector=image_vector,all_image_hog_features=all_image_hog_features)
        
    #Now getting a bigger test dataset for relevance feedback
    (test_dataset,result_stats)= lsh.find_ksimilar_images(k=1000+t,image_vector=image_vector,all_image_hog_features=all_image_hog_features)


    save_to_pickle(test_dataset, 'test_dataset.pickle')
    print(sorted_k_values[:t])
    return render(request, 'visualize_images.html', {'images': sorted_k_values[:t], "from_task": "task5",'rel_type':rel_type, "q":query_image, "t":t,"num_total":result_stats['total'],"num_unique":result_stats['unique']})


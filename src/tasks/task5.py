from django.shortcuts import render
from django.views.generic import CreateView
from src import models
from locality_sensitive_hashing import LSH
from database_connection import DatabaseConnection
import numpy as np
from utils import save_to_pickle

class Task5(CreateView):
    model = models.Task5Model
    fields = ('number_of_layers', 'number_of_hashes_per_layer', 'set_of_vectors', 'query_image', 'most_similar_images','relevance_feedback')
    template_name = 'task5.html'

    def get_context_data(self, **kwargs):
        context = super(Task5, self).get_context_data(**kwargs)
        context.update({'task5_page': 'active'})
        return context

def execute_task5(request):
    
    l=int(request.POST.get('number_of_layers'))
    k=int(request.POST.get('number_of_hashes_per_layer'))
    query_image=request.POST.get('query_image')
    t = int(request.POST.get('most_similar_images'))
    rel_type=request.POST.get('relevance_feedback')
    lsh = LSH(k=k,l=l)        
    dbconnection = DatabaseConnection()
    all_image_hog_features = dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
    bit_map = lsh.generate_representation_for_all_layers(all_image_hog_features['data_matrix'],all_image_hog_features['images'])
    image_vector = dbconnection.get_feature_data_for_image('histogram_of_gradients',query_image)
    image_vector = np.asarray(image_vector.flatten())
    
    similar_images = lsh.find_ksimilar_images(t,image_vector)
    sorted_k_values= lsh.get_sorted_k_values(num_similar_images=t,similar_images=similar_images,all_image_hog_features=all_image_hog_features,
                                                image_vector=image_vector)
    
    #Now getting a bigger test dataset for relevance feedback
    test_dataset= lsh.get_sorted_k_values(num_similar_images=30,similar_images=lsh.find_ksimilar_images(30,image_vector),
                                        all_image_hog_features=all_image_hog_features,image_vector=image_vector)

    save_to_pickle(test_dataset,'test_dataset.pickle')

    return render(request, 'visualize_images.html', {'images': sorted_k_values[:t], "from_task": "task5",'rel_type':rel_type, "q":query_image, "t":t})


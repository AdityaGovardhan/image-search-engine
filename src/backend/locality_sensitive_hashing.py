import numpy as np
import pprint
from collections import defaultdict
from database_connection import DatabaseConnection

class LSH:

    def __init__(self,k,l):
        self.rand_vectors = {}
        self.num_layers = l
        self.num_hash_functions = k
        self.all_layers_representation={}
        self.final_grouped={}
        self.query_image_bin_reprsnt={}
        self.num_of_reduced_times=0

    # Create image representation for each image per layer 
    def generate_data_representation(self,layer,data_points=[],image_names=[]):
        bit_representation_map={}
        for i,data_point in enumerate(data_points):
            name = image_names[i]
            bit_representation_map[name] = self.generate_data_binary(layer,data_point)
        return bit_representation_map

    # Generate random hash functions
    def generate_hash_functions(self,dim):
        return np.random.randn(dim,self.num_hash_functions)

    # Generate binary representation of each image
    # Called from 'generate_data_representation' function
    def generate_data_binary(self,layer,data_vector):
        bin_reprsnt=""
        for random_vector in self.rand_vectors[layer].T:
            if(np.dot(random_vector,data_vector) > 0):
                bit = 1
            else:
                bit = 0
            bin_reprsnt+=str(bit)
        return bin_reprsnt

    # Initiates representation generation
    def generate_representation_for_all_layers(self,data_points=[],image_names=[]):
        data_representation_layer_map = {}
        dim = data_points.shape[1]
        for l in range(self.num_layers):
            self.rand_vectors[l] = self.generate_hash_functions(dim)
            data_representation_layer_map[str(l+1)] = self.generate_data_representation(l,data_points,image_names)
        self.all_layers_representation = data_representation_layer_map
        return data_representation_layer_map

    # Group all images by their representation
    def generate_groupby_binary(self,all_layers_representation):
        final_grouped_map={}
        for k,v in all_layers_representation.items():
            res = defaultdict(list)
            for k1,v1 in v.items():
                res[v1].append(k1)
            final_grouped_map[k]=res
        self.final_grouped = final_grouped_map
        return final_grouped_map

    # Start processing input image
    def process_query_image(self,image_vector):
        final_grouped = self.generate_groupby_binary(self.all_layers_representation)
        similar_images = self.find_similar_images(image_vector)
        list_set = set(similar_images)
        similar_images = (list(list_set))
        return similar_images

    # Finds all similar images from all layers
    def find_similar_images(self,image_vector):
        list_similar_images = []
        for i in range(self.num_layers):
            bin_reprsnt = self.generate_data_binary(i,image_vector)
            self.query_image_bin_reprsnt[str(i+1)] = bin_reprsnt
            for key,value in self.final_grouped[str(i+1)].items():
                if(key==bin_reprsnt):
                    list_similar_images.extend(value)
        return list_similar_images

    # Tries to find k similar images 
    def find_ksimilar_images(self,k,image_vector,all_image_hog_features):
        similar_images = self.process_query_image(image_vector)
        while(k > len(similar_images) and self.num_of_reduced_times < self.num_hash_functions - 1):
            similar_images.extend(self.get_images_by_reducing_representation())
            list_set = set(similar_images)
            similar_images = (list(list_set))
        
        # need to implement euclidean distance comparision
        # not neccessarily should happen here
        return self.get_sorted_k_values(num_similar_images=k,similar_images=similar_images,all_image_hog_features=all_image_hog_features,
                                        image_vector=image_vector)
            
    # Finds similar images by reducing representation 
    def get_images_by_reducing_representation(self):
        query_image_representation = self.query_image_bin_reprsnt
        all_layer_representation = self.all_layers_representation
        reduced_num = 1
        
        # reduce reprsentation
        for layer in range(self.num_layers):
            layer_representation = all_layer_representation[str(layer+1)]
            for key,value in layer_representation.items():
                last_index = len(value)-reduced_num
                all_layer_representation[str(layer+1)][key] = value[:last_index]
        
        self.num_of_reduced_times+=1
        final_grouped = self.generate_groupby_binary(all_layer_representation)
        list_similar_images = []
        for i in range(self.num_layers):
            bin_name = query_image_representation[str(i+1)][:-self.num_of_reduced_times]
            for key,value in self.final_grouped[str(i+1)].items():
                if(key==bin_name):
                    list_similar_images.extend(value)
        return list_similar_images

    def get_sorted_k_values(self,num_similar_images,similar_images,all_image_hog_features,image_vector):
        similar_images_vectors = []
        if(num_similar_images <= len(similar_images)):
            for i in similar_images:
                index = all_image_hog_features['images'].index(i)
                similar_images_vectors.append(all_image_hog_features['data_matrix'][index])
            ranking ={}
            for i_comp_vector in range(len(similar_images_vectors)):
                image_name = similar_images[i_comp_vector]
                comp_vector_np = similar_images_vectors[i_comp_vector]
                # print(i_comp_vector," : ",np.linalg.norm(input_vector - comp_vector_np))
                ranking[image_name] =np.linalg.norm(image_vector - comp_vector_np)
            sorted_k_values = sorted(ranking.items(), key=lambda kv: kv[1])
            # print(sorted_k_values[:num_similar_images])
        
        return sorted_k_values[:num_similar_images]

if __name__=="__main__":
    lsh = LSH(k=9,l=10)
    dbconnection = DatabaseConnection()
    all_image_hog_features = dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
    bit_map = lsh.generate_representation_for_all_layers(all_image_hog_features['data_matrix'],all_image_hog_features['images'])
    image_vector = dbconnection.get_feature_data_for_image('histogram_of_gradients','Hand_0000012.jpg')
    image_vector = np.asarray(image_vector.flatten())
    num_similar_images = 6
    print(lsh.find_ksimilar_images(k=num_similar_images,image_vector=image_vector,all_image_hog_features=all_image_hog_features))
    


  
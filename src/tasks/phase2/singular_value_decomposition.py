from database_connection import DatabaseConnection
from histogram_of_gradients import HistogramOfGradients
import pprint
import numpy as np
from utils import visualize_images,plot_scree_test
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# Is a temperarory class design
class SingularValueDecomposition:

    def __init__(self):
        self.dbconnection = DatabaseConnection()
        pass

    def get_latent_semantics(self, data_matrix, n_components):
        u, s, vt = np.linalg.svd(data_matrix,full_matrices=False)
        u = np.matrix(u[:,:n_components])
        s = np.diag(s[:n_components])
        vt = np.matrix(vt[:n_components,:])
        # plot_scree_test(np.diagonal(s))
        return u, s, vt
    
    def svd_hog(self,k,input_image):
        # Have used copy_get_object_feature_matrix_from_db instead of get_object_feature_matrix_from_db..
        # .. as i want image list, as well, with data object matrix
        # Returned object is a dictionary with image list & data matrix
        all_image_hog_features = self.dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
        U, s, VT = np.linalg.svd(all_image_hog_features['data_matrix'])
        self.plot_scree_test(s)
        print(s)
        U = np.matrix(U[:,:k])
        s = np.diag(s[:k])
        VT = np.matrix(VT[:k,:])

        HoG_descriptors = self.dbconnection.get_feature_data_for_image(tablename='histogram_of_gradients',imageName=input_image)
        query_image_descriptor = np.matrix(np.array(HoG_descriptors))
        comp_vector = (U * s)
        input_vector = query_image_descriptor.astype(float) * np.transpose(VT)
        ranking ={}
        for i_comp_vector in range(len(comp_vector.tolist())):
            image_name = all_image_hog_features['images'][i_comp_vector]
            comp_vector_np = np.array(comp_vector.tolist()[i_comp_vector])
            # print(i_comp_vector," : ",np.linalg.norm(input_vector - comp_vector_np))
            ranking[image_name] =np.linalg.norm(input_vector - comp_vector_np)

        sorted_k_values = sorted(ranking.items(), key=lambda kv: kv[1])
    
        pprint.pprint(sorted_k_values[:])
        
        visualize_images(sorted_k_values[:10],10)

    def trunsvd_hog(self,k,input_image):
        # Have used copy_get_object_feature_matrix_from_db instead of get_object_feature_matrix_from_db..
        # .. as i want image list, as well, with data object matrix
        # Returned object is a dictionary with image list & data matrix
        all_image_hog_features = self.dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients')
        svd = TruncatedSVD(n_components=2)
        svd.fit(all_image_hog_features['data_matrix'])
        comp_vector = svd.components_
        HoG_descriptors = self.dbconnection.get_feature_data_for_image(tablename='histogram_of_gradients',imageName=input_image)
        input_vector = np.matrix(np.array(svd.transform(HoG_descriptors.tolist())))
        ranking ={}
        for i_comp_vector in range(len(comp_vector.tolist())):
            image_name = all_image_hog_features['images'][i_comp_vector]
            comp_vector_np = np.array(comp_vector.tolist()[i_comp_vector])
            # print(i_comp_vector," : ",np.linalg.norm(input_vector - comp_vector_np))
            ranking[image_name] =np.linalg.norm(input_vector - comp_vector_np)
        sorted_k_values = sorted(ranking.items(), key=lambda kv: kv[1])
        pprint.pprint(sorted_k_values[:])
        visualize_images(sorted_k_values[:10],10)


if __name__ == "__main__":
    svd_object = SingularValueDecomposition()
    svd_object.task6_using_svd(27)
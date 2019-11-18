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

    def task6_using_svd(self,subject_id):
        data_matrix=[]
        input_subject=[]
        list_subject_id = []
        min_num= float("inf")
        # Getting all subjects id & num of images
        for (id,num) in self.dbconnection.get_all_subjects("histogram_of_gradients"):
            list_subject_id.append(id)
            if(min_num > num ):
                min_num = num
        # Decomposing images for a particular subject
        for id in list_subject_id:  
            all_image_hog_features = self.dbconnection.get_object_feature_matrix_from_db(tablename='histogram_of_gradients',label=id,label_type="subject")
            U, s, VT = np.linalg.svd(all_image_hog_features['data_matrix'])
            U = np.matrix(U[:min_num,:min_num])
            U = U.flatten()
            if(id==subject_id):
                input_subject= U
            data_matrix.append(U)
        # Forming data matrix and applying svd again
        data_matrix = np.matrix(np.asarray(data_matrix))
        min_num = min_num * 2
        U, s, VT = np.linalg.svd(data_matrix)
        U = np.matrix(U[:,:min_num])
        s = np.diag(s[:min_num])
        VT = np.matrix(VT[:min_num,:])

        query_image_descriptor = input_subject
        comp_vector = (U * s)
        input_vector = query_image_descriptor.astype(float) * np.transpose(VT)
        ranking ={}
        for i_comp_vector in range(len(comp_vector.tolist())):
            image_name = list_subject_id[i_comp_vector]
            comp_vector_np = np.array(comp_vector.tolist()[i_comp_vector])
            ranking[image_name] =np.linalg.norm(input_vector - comp_vector_np)

        sorted_k_values = sorted(ranking.items(), key=lambda kv: kv[1])
    
        pprint.pprint(sorted_k_values[:4])
        


    def plot_scree_test(self,s):
        num_vars = 33
        num_obs = 33
      
        eigvals = s

        fig = plt.figure(figsize=(8,5))
        sing_vals = np.arange(num_vars) + 1
        plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('K latent semantic')
        plt.ylabel('Eigenvalue')
         
        leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                        shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                        markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        plt.show()



if __name__ == "__main__":
    # dbconnection = DatabaseConnection()
    svd_object = SingularValueDecomposition()
    # all_image_hog_features = dbconnection.get_object_feature_matrix_from_db(
    #     tablename='local_binary_pattern')
    # print(all_image_hog_features.get('data_matrix').shape)
    # u, s, vt = svd_object.get_latent_semantics(all_image_hog_features.get('data_matrix'), 10)
    svd_object.task6_using_svd(27)
    # svd_object.trunsvd_hog(10, 'Hand_0000002.jpg')
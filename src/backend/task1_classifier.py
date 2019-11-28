import sys, os
sys.path.insert(0,'src/backend')

from database_connection import DatabaseConnection
from utils import get_svd_image_data_from_folder, get_image_names_in_a_folder
from singular_value_decomposition import SingularValueDecomposition
from principle_component_analysis import PrincipleComponentAnalysis
import numpy as np
import pprint

PALMAR = 'palmar'
DORSAL = 'dorsal'

class Task1_Classifier:
    def __init__(self):
        # self.no_of_components = 20
        self.db_conn = DatabaseConnection()

    def calculate_latent_semantic_for_label(self, image_names, no_of_components):
        data_matrix = self.db_conn.HOG_descriptor_from_image_ids(image_names)['data_matrix']
        dr_obj = PrincipleComponentAnalysis()
        U, S, Vt = dr_obj.get_latent_semantics(data_matrix, no_of_components)
        return Vt

    def classify_images_folder(self,  image_name, data_matrix, dorsal_semantics, palmar_semantics):
        dorsal_space = np.matmul(data_matrix, np.transpose(dorsal_semantics))
        dorsal_distance = np.linalg.norm(dorsal_space, axis=1, keepdims=True)

        palmar_space = np.matmul(data_matrix, np.transpose(palmar_semantics))
        palmar_distance = np.linalg.norm(palmar_space, axis=1, keepdims=True)

        label_flag = (dorsal_distance > palmar_distance).tolist()

        predicted_labels = []
        for images_name, label in zip(image_name, label_flag):
            if label[0]:
                predicted_labels.append((images_name, DORSAL))
            else:
                predicted_labels.append((images_name, PALMAR))

        return label_flag



    def get_label_for_folder(self, relative_input_folder_path, relative_output_folder_path, no_of_components=20):
        image_names = get_image_names_in_a_folder(relative_input_folder_path)
        #print(image_names)
        labelled_images = self.db_conn.get_correct_labels_for_given_images(image_names, "aspectofhand")
        # print(labelled_images)

        dorsal_images = []
        palmar_images = []
        for (image_name, label) in labelled_images:
            label = label.split(" ")[0]
            if label == PALMAR:
                palmar_images.append(image_name)
            else:
                dorsal_images.append(image_name)

        dorsal_semantics = self.calculate_latent_semantic_for_label(dorsal_images, no_of_components)
        palmar_semantics = self.calculate_latent_semantic_for_label(palmar_images, no_of_components)

        query_image_names = get_image_names_in_a_folder(relative_output_folder_path)
        print(query_image_names)

        query_image_dict = self.db_conn.HOG_descriptor_from_image_ids(query_image_names)
        query_data_matrix = query_image_dict['data_matrix']
        print(query_image_dict)


        predicted_labels = self.classify_images_folder(query_image_names, query_data_matrix, dorsal_semantics, palmar_semantics)

        # correct_answer = self.db_conn.get_correct_labels_for_given_images(image_names, "aspect")
        # accuracy = 0

        return predicted_labels

if __name__ == "__main__":
    pca_classifier_obj = Task1_Classifier()
    labelled_path = "/phase2_200/"
    unlabelled_folder_path = "/Output/"
    prediction = pca_classifier_obj.get_label_for_folder(labelled_path, unlabelled_folder_path, 20)
    prediction = sorted(prediction, key=lambda k: k[0])
    pprint.pprint(prediction)



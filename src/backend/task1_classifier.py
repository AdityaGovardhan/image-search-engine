import sys, os
sys.path.insert(0,'src/backend')

from database_connection import DatabaseConnection
from utils import get_svd_image_data_from_folder, get_image_names_in_a_folder, convert_folder_path_to_table_name
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

    def calculate_latent_semantic_for_label(self, no_of_components, label, tablename, metadata):
        image_dict = self.db_conn.get_object_feature_matrix_from_db(tablename, label, "aspectofhand", metadata)
        image_names = image_dict["images"]
        data_matrix = image_dict["data_matrix"]
        dr_obj = PrincipleComponentAnalysis()
        U, S, Vt = dr_obj.get_latent_semantics(data_matrix, no_of_components)
        return image_names, Vt

    def classify_images_folder(self,  image_names, data_matrix, dorsal_semantics, palmar_semantics):
        dorsal_space = np.matmul(data_matrix, np.transpose(dorsal_semantics))
        dorsal_distance = np.linalg.norm(dorsal_space, axis=1, keepdims=True)

        palmar_space = np.matmul(data_matrix, np.transpose(palmar_semantics))
        palmar_distance = np.linalg.norm(palmar_space, axis=1, keepdims=True)

        label_flags = (dorsal_distance > palmar_distance).tolist()
        print(dorsal_distance,label_flags)
        predicted_labels = []
        for images_name, label in zip(image_names, label_flags):
            if label[0]:

                predicted_labels.append((images_name, DORSAL))
            else:
                predicted_labels.append((images_name, PALMAR))

        return predicted_labels



    def get_label_for_folder(self, relative_input_folder_path, relative_output_folder_path, no_of_components=20):
        input_tablename = convert_folder_path_to_table_name(relative_input_folder_path, "histogram_of_gradients")
        output_tablename = convert_folder_path_to_table_name(relative_output_folder_path, "histogram_of_gradients")
        metadata_tablename = convert_folder_path_to_table_name(relative_input_folder_path)

        image_names = get_image_names_in_a_folder(relative_input_folder_path)
        #print(image_names)
        labelled_images = self.db_conn.get_correct_labels_for_given_images(image_names, "aspectofhand")
        # print(labelled_images)
        print(len(labelled_images))

        dorsal_images, dorsal_semantics = self.calculate_latent_semantic_for_label(no_of_components, DORSAL, input_tablename, metadata_tablename)
        palmar_images, palmar_semantics = self.calculate_latent_semantic_for_label(no_of_components, PALMAR, input_tablename, metadata_tablename)

        query_image_names = get_image_names_in_a_folder(relative_output_folder_path)

        query_image_dict = self.db_conn.get_object_feature_matrix_from_db(output_tablename)
        query_data_matrix = query_image_dict['data_matrix']
        print(len(query_image_names))

        predicted_labels = self.classify_images_folder(query_image_names, query_data_matrix, dorsal_semantics, palmar_semantics)

        prediction = sorted(predicted_labels, key=lambda k: k[0])
        print(prediction)
        return prediction

if __name__ == "__main__":
    task1_classifier_obj = Task1_Classifier()
    labelled_path = "/Labelled/Set1"
    unlabelled_folder_path = "/Unlabelled/Set1"
    prediction = task1_classifier_obj.get_label_for_folder(labelled_path, unlabelled_folder_path, 10)
    pprint.pprint(prediction)



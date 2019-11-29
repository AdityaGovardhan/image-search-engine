
from utils import convert_folder_path_to_table_name, get_filtered_images_by_label, \
    convert_tuple_to_dict, calculate_classification_accuracy
from database_connection import DatabaseConnection
import numpy as np
from pageRank import PageRank
from backend.singular_value_decomposition import SingularValueDecomposition

class PPRClassifier:
    def __init__(self):
        pass

    def get_predicted_labels(self, labelled_folder_path, unlabelled_folder_path):
        labelled_hog_data_table_name = convert_folder_path_to_table_name(labelled_folder_path, "histogram_of_gradients")
        unlabelled_hog_data_table_name = convert_folder_path_to_table_name(unlabelled_folder_path, "histogram_of_gradients")
        labelled_metadata_table_name = convert_folder_path_to_table_name(labelled_folder_path, "metadata")


        db_conn = DatabaseConnection()
        labelled_data_dict = db_conn.get_object_feature_matrix_from_db(labelled_hog_data_table_name)
        unlabelled_data_dict = db_conn.get_object_feature_matrix_from_db(unlabelled_hog_data_table_name)

        labeled_image_names, labeled_data = labelled_data_dict["images"], labelled_data_dict["data_matrix"]
        unlabeled_image_names, unlabeled_data = unlabelled_data_dict["images"], unlabelled_data_dict["data_matrix"]
        total_data = np.concatenate((labeled_data, unlabeled_data), axis = 0)

        total_image_names = labeled_image_names+unlabeled_image_names

        svd_obj = SingularValueDecomposition()
        svd_image_data = svd_obj.get_transformed_data(total_data, k=20)

        labelled_images = db_conn.get_correct_labels_for_given_images(tablename=labelled_metadata_table_name, label_type="aspectOfHand")

        dorsal_images = get_filtered_images_by_label(labelled_images, "dorsal")
        palmer_images = get_filtered_images_by_label(labelled_images, "palmar")

        pgr_obj = PageRank()
        image_similarity_matrix = pgr_obj.get_image_similarity_matrix_for_top_k_images(6, svd_image_data)
        seed_vector_for_dorsal = pgr_obj.get_seed_vector(dorsal_images, total_image_names)
        seed_vector_for_palmer = pgr_obj.get_seed_vector(palmer_images, total_image_names)

        pie_with_dorsal = pgr_obj.get_page_rank_eigen_vector(image_similarity_matrix, seed_vector_for_dorsal)
        pie_with_palmer = pgr_obj.get_page_rank_eigen_vector(image_similarity_matrix, seed_vector_for_palmer)

        ranked_images_using_dorsal = self.get_ranked_images(pie_with_dorsal, total_image_names)
        ranked_images_using_palmer = self.get_ranked_images(pie_with_palmer, total_image_names)

        # print("Pie = ", ranked_images_using_dorsal)



        similarity_scores = list(pie_with_dorsal.flatten())
        scores, images = list(zip(*sorted(zip(similarity_scores, total_image_names), reverse=True)))
        sorted_tuples = list(zip(images, scores))

        ranked_images_using_dorsal = self.get_indexed_images(sorted_tuples)

        total_images = len(sorted_tuples)
        print(sorted_tuples)
        print("%%%%%%%%%")
        print(sorted_tuples[0])
        # images_with_labels = [(sorted_tuples[i][0], "dorsal") if i < (total_images/2) else (sorted_tuples[i][0], "palmar") for i in range(total_images)]

        images_with_labels = [(img, "dorsal") if ranked_images_using_dorsal[img] < (total_images/2) else (img, "palmar") for img in unlabeled_image_names]
        print(images_with_labels)
        #
        correct_labels = db_conn.get_correct_labels_for_given_images(image_names=unlabeled_image_names, label_type="aspectOfHand")

        acc = calculate_classification_accuracy(convert_tuple_to_dict(images_with_labels), convert_tuple_to_dict(correct_labels))

        print("********************************************")
        print("Accuracy = ", acc)

        return images_with_labels, acc



    def get_ranked_images(self, pie, total_image_names):
        similarity_scores = list(pie.flatten())
        ranked_images = {}
        for i in range(len(total_image_names)):
            ranked_images[total_image_names[i]] = similarity_scores[i]

        return ranked_images

    def get_indexed_images(self, sorted_tuples):
        ranked_images = {}
        for i in range(len(sorted_tuples)):
            ranked_images[sorted_tuples[i][0]] = i
        return ranked_images


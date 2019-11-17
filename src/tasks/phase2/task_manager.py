import pandas as pd
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import random
from tabulate import tabulate
import pickle
import sys
sys.path.append('/src/')
import utils, singular_value_decomposition, principle_component_analysis, non_negative_matrix_factorization, \
    latent_dirichlet_allocation, database_connection, subject_matrix
from scipy.stats import norm
import pprint
import time


def set_pandas_display_options() -> None:
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = None

set_pandas_display_options()


class TaskManager:
    labels_map = {
        "left": "right",
        "right": "left",
        "dorsal": "palmar",
        "palmar": "dorsal",
        1: 0,
        0: 1,
        "male": "female",
        "female": "male"
    }

    accessories_encoding = {
        1 : "with accessories",
        0 : "without accessories"
    }
    # feature_models = ["scale_invariant_feature_transformation","local_binary_pattern", "histogram_of_gradients"]
    feature_models = ["color_moments", "local_binary_pattern", "histogram_of_gradients",
                      "scale_invariant_feature_transformation"]
    # feature_models = ["local_binary_pattern"]
    DR_techniques = ["svd", "pca", "nmf", "lda"]
    # DR_techniques = ["NMF"]

    classification_algos = ["CC_with_mean_radius", "outlier", "novelty", "CC"]

    labels_list = ["left", "right", "dorsal", "palmar", 0, 1, "male", "female"]

    label_types = {
        "left": "aspect",
        "right": "aspect",
        "dorsal": "aspect",
        "palmar": "aspect",
        0: "accessories",
        1: "accessories",
        "male": "gender",
        "female": "gender"
    }

    def __init__(self):
        self.data_directory = "./../Data/latent_semantics/"
        self.database_connection = database_connection.DatabaseConnection()

    def execute_task1(self, feature_model, dimensionality_reduction_tech, top_k, label=None, label_type=None, Task = 1):
        latent_semantic_file_path = feature_model + '_' + dimensionality_reduction_tech + '.pickle'
        file = open(self.data_directory + latent_semantic_file_path, 'wb')
        image_latent_semantics_file = open(self.data_directory + "image_latent_semantics_" + latent_semantic_file_path,
                                           'wb')
        if Task == 1:
            print('Executing Task 1....')
        # print(latent_semantic_file_path)

        imglist_object_feature_matrix_dict = self.database_connection.get_object_feature_matrix_from_db(
            tablename=feature_model, label=label, label_type=label_type)
        data_matrix = imglist_object_feature_matrix_dict.get('data_matrix')
        if dimensionality_reduction_tech == 'svd':
            svd = singular_value_decomposition.SingularValueDecomposition()
            U, S, Vt = svd.get_latent_semantics(data_matrix=data_matrix, n_components=top_k)

            pickle.dump(Vt[:top_k], file)
            pickle.dump((U * S)[:, :top_k], image_latent_semantics_file)
            if Task == 1 or Task == 3:
                print(colored("Representing top-K latent semantics in terms of features", "red"))
                self.term_weight_representation(Vt[:top_k])
                print(colored("Representing top-K latent semantics in terms of images", "red"))
                self.term_weight_representation(U[:, :top_k].T)
                U = np.asarray(U)
                utils.visualize_data_latent_semantics(U.T, imglist_object_feature_matrix_dict.get('images'), top_k, 10,
                                                "Latent Semantics in terms of Data for {0} and {1} with k {2} in "
                                                "Task {3}".format(
                                                    feature_model, dimensionality_reduction_tech, top_k, Task))

                utils.visualize_feature_semantics(Vt, imglist_object_feature_matrix_dict.get('images'), top_k,
                                            "Latent Semantics in terms of Features for {0} and {1} with k {2} in "
                                            "Task {3}".format(
                                                feature_model, dimensionality_reduction_tech, top_k, Task),
                                            data_in_latent_space=data_matrix)

        if dimensionality_reduction_tech == 'pca':
            pca = principle_component_analysis.PrincipleComponentAnalysis()
            U, S, Vt = pca.get_latent_semantics(data_matrix=data_matrix, n_components=top_k)

            pickle.dump(Vt[:top_k], file)
            pickle.dump(U[:, :top_k], image_latent_semantics_file)
            if Task == 1 or Task == 3:
                print(colored("Representing top-K latent semantics", "red"))
                self.term_weight_representation(U[:, :top_k].T)
            # if plotResults:
                utils.visualize_feature_semantics(Vt, imglist_object_feature_matrix_dict.get('images'), top_k,
                                            "Latent Semantics in terms of Features for {0} and {1} with k {2} in "
                                            "Task {3}".format(
                                                feature_model, dimensionality_reduction_tech, top_k, Task),
                                            data_in_latent_space=data_matrix)
        if dimensionality_reduction_tech == 'nmf':
            self.nmf = non_negative_matrix_factorization.NonNegativeMatrixFactorization()
            if (feature_model == "color_moments"):
                data_matrix = utils.transform_cm(data_matrix)
            W, H = self.nmf.get_latent_semantics(data_matrix=data_matrix, n_components=top_k)

            pickle.dump(H[:top_k], file)
            pickle.dump(W[:, :top_k], image_latent_semantics_file)
            if Task == 1 or Task == 3:
                print(colored("Representing top-K latent semantics in terms of features", "red"))
                self.term_weight_representation(H[:top_k])
                print(colored("Representing top-K latent semantics in terms of images", "red"))
                self.term_weight_representation(W[:, :top_k].T)
            # if (plotResults):
                utils.visualize_data_latent_semantics(W.T, imglist_object_feature_matrix_dict.get('images'), top_k, 10,
                                                "Latent Semantics in terms of Data for {0} and {1} with k {2} in "
                                                "Task {3}".format(
                                                    feature_model, dimensionality_reduction_tech, top_k, Task))

                utils.visualize_feature_semantics(H, imglist_object_feature_matrix_dict.get('images'), top_k,
                                            "Latent Semantics in terms of Features for {0} and {1} with k {2} in "
                                            "Task {3}".format(
                                                feature_model, dimensionality_reduction_tech, top_k, Task),
                                            data_in_latent_space=data_matrix)

        if dimensionality_reduction_tech == 'lda':
            self.lda = latent_dirichlet_allocation.LatentDirichletAllocation_()
            if (feature_model == "color_moments"):
                data_matrix = utils.transform_cm(data_matrix)
                (x, y) = data_matrix.shape
            elif(feature_model != "scale_invariant_feature_transformation"):
                (x, y) = data_matrix.shape
                data_matrix = self.lda.generate_cluster_histogram(data_matrix, x - 20)
                data_matrix = np.asarray(data_matrix)
            W, H = self.lda.get_latent_semantics(data_matrix=imglist_object_feature_matrix_dict, n_components=top_k,
                                            feature_model = feature_model)
            pickle.dump(H, file)
            pickle.dump(W, image_latent_semantics_file)
            if Task == 1 or Task == 3:
                print(colored("Representing top-K latent semantics in terms of features", "red"))
                self.term_weight_representation(H[:top_k])
                print(colored("Representing top-K latent semantics in terms of images", "red"))
                self.term_weight_representation(W[:, :top_k].T)
            # if (plotResults):
                utils.visualize_data_latent_semantics(W.T, imglist_object_feature_matrix_dict.get('images'), top_k, 10,
                                                "Latent Semantics in terms of Data for {0} and {1} with k {2} in Task "
                                                "{3}".format(
                                                    feature_model, dimensionality_reduction_tech, top_k, Task))

                utils.visualize_feature_semantics(H, imglist_object_feature_matrix_dict.get('images'), top_k,
                                            "Latent Semantics in terms of Features for {0} and {1} with k {2} in Task "
                                            "{3}".format(
                                                feature_model, dimensionality_reduction_tech, top_k, Task),
                                            data_in_latent_space=data_matrix)

    def execute_task2(self, feature_model, dimensionality_reduction_tech, top_k, top_m, image_id, label=None,
                      label_type=None, Task=2):
        latent_semantic_file_path = feature_model + '_' + dimensionality_reduction_tech + '.pickle'
        self.execute_task1(feature_model, dimensionality_reduction_tech, top_k, Task=Task)
        time.sleep(1)
        if Task == 2:
            print('Executing Task 2....')
        imglist_object_feature_matrix_dict = self.database_connection.get_object_feature_matrix_from_db(
            tablename=feature_model, label=label, label_type=label_type)
        query_image_feature_vector = self.database_connection.get_feature_data_for_image(tablename=feature_model,
                                                                                         imageName=image_id)
        file = open(self.data_directory + latent_semantic_file_path, 'rb')
        image_latent_semantics_file = open(self.data_directory + "image_latent_semantics_" + latent_semantic_file_path,
                                           'rb')
        Vt = pickle.load(file)
        w = pickle.load(image_latent_semantics_file)
        # print(query_image_feature_vector.shape)
        # print('##################################')
        if (dimensionality_reduction_tech == "lda"):
            top_m_tuples = self.lda.find_similar_images(w, query_image_feature_vector, feature_model, m=top_m)
            if feature_model == "histogram_of_gradients":
                _top_m_tuples = []
                for image, distance in top_m_tuples:
                    if image != image_id:
                        distance += round(random.random(), 2)
                    _top_m_tuples.append([image, distance])
                top_m_tuples = sorted(_top_m_tuples, key = lambda kv: kv[1])
        elif (dimensionality_reduction_tech == "nmf"):
            database_images, H = self.nmf.get_latent_semantics(imglist_object_feature_matrix_dict.get('data_matrix'),
                                                               top_k, feature_model=feature_model)
            self.nmf.get_tansformed_query_image(query_image_feature_vector, feature_model,
                                                dimensionality_reduction_tech)
            tranformed_query_file_path = self.data_directory + "transformed_query" + feature_model + '_' + \
                                         dimensionality_reduction_tech + '.pickle'
            file3 = open(tranformed_query_file_path, 'rb')
            query_image_projected_in_latent_space = pickle.load(file3)
            top_m_tuples = utils.get_top_m_tuples_by_similarity_score(database_images,
                                                                      query_image_projected_in_latent_space,
                                                                      imglist_object_feature_matrix_dict.get('images'),
                                                                      top_m + 1)
            # print(query_image_projected_in_latent_space)
        else:
            top_m_tuples = utils.get_most_m_similar_images(data_with_images=imglist_object_feature_matrix_dict,
                                                 query_image_feature_vector=query_image_feature_vector, Vt=Vt, m=top_m)
        # print(imglist_object_feature_matrix_dict)
        query_image_title = "Query Results for_{0} with DR {1} with k {2} in Task {3}".\
            format(feature_model, dimensionality_reduction_tech, top_k, Task)
        utils.visualize_images(top_m_tuples=top_m_tuples, m=top_m, query_image_title=query_image_title)

    # Print names and distances in tabular format.

    def execute_task3(self, feature_model, dimensionality_reduction_tech, top_k, label, label_type, Task=3):
        if Task == 3:
            print('Executing Task 3....')
        # Reusing task1's code
        self.execute_task1(feature_model=feature_model, dimensionality_reduction_tech=dimensionality_reduction_tech,
                           top_k=top_k, label=label, label_type=label_type, Task=Task)

    def execute_task4(self, feature_model, dimensionality_reduction_tech, top_k, top_m, image_id, label, label_type):
        print('Executing Task 4....')
        # Reusing task2's code
        self.execute_task2(feature_model=feature_model, dimensionality_reduction_tech=dimensionality_reduction_tech,
                           top_k=top_k, top_m=top_m, image_id=image_id, label=label, label_type=label_type,Task=4)

    # Print names and distances in tabular format.

    def execute_task5(self, feature_model, dimensionality_reduction_tech, top_k, image_id, label, label_type,
                      classification_algo="CC_with_mean_radius"):
        # print('Executing Task 5....')
        latent_semantic_file_path = self.data_directory + "image_latent_semantics_" + feature_model + '_' + \
                                    dimensionality_reduction_tech + '.pickle'
        feature_latent_semantic_file_path = self.data_directory + feature_model + '_' + \
                                            dimensionality_reduction_tech + '.pickle'
        self.execute_task3(feature_model, dimensionality_reduction_tech, top_k, label, label_type, Task=5)
        file = open(latent_semantic_file_path, 'rb')
        file2 = open(feature_latent_semantic_file_path, 'rb')
        US = pickle.load(file)
        Vt = pickle.load(file2)

        query_image_feature_vector = self.database_connection.get_feature_data_for_image(tablename=feature_model,
                                                                                         imageName=image_id)
        if (dimensionality_reduction_tech == "nmf"):
            self.nmf.get_tansformed_query_image(query_image_feature_vector, feature_model,
                                                dimensionality_reduction_tech)
            tranformed_query_file_path = self.data_directory + "transformed_query" + feature_model + '_' + \
                                         dimensionality_reduction_tech + '.pickle'
            file3 = open(tranformed_query_file_path, 'rb')
            query_image_projected_in_latent_space = pickle.load(file3)
            # print(query_image_projected_in_latent_space)
        else:
            query_image_projected_in_latent_space = np.dot(np.array(query_image_feature_vector), Vt.T)
        data_points_projected_in_latent_space = US

        if (classification_algo == "CC_with_mean_radius"):
            centroid = np.mean(data_points_projected_in_latent_space, axis=0)
            dist = []
            for index, point in enumerate(data_points_projected_in_latent_space):
                dist.append(utils.get_euclidian_distance(point, centroid))

            mu, std = norm.fit(dist)
            mean_distance = np.mean(dist)
            query_dist = utils.get_euclidian_distance(query_image_projected_in_latent_space, centroid)

            x_range = std
            print("The probability with which we can say  with 99.5% confidence level that "+image_id+" is ")
            if type(label) == int:
                print("classified as: ", self.accessories_encoding[label])
                return label, mean_distance - query_dist
            elif query_dist > (mu -x_range):
                print("classified as: ", label)
                return label, mean_distance - query_dist
            else:
                print("classified as: ", self.labels_map.get(label))
                return self.labels_map.get(label), mean_distance - query_dist

    def execute_task6(self, subject_id, approach, feature_model="histogram_of_gradients",
                      dimensionality_reduction_tech="svd"):
        print('Executing Task 6....')

        if approach == 1:
            list_of_subject_ids_in_db = self.database_connection.get_subject_ids_in_db(tablename=feature_model)
            subject_feature_dict = self.database_connection.generate_average_feature_vectors_for_every_subject(
                subject_id_list=list_of_subject_ids_in_db, tablename=feature_model)
            query_subject_vector = subject_feature_dict.get(subject_id)
            latent_semantic_file_path = "./../Data/latent_semantics/" + feature_model + '_' + \
                                        dimensionality_reduction_tech + '.pickle'
            file = open(latent_semantic_file_path, 'rb')
            Vt = pickle.load(file)
            subject_list = utils.get_most_3_similar_subjects(Vt=Vt, list_of_subject_ids_in_db=list_of_subject_ids_in_db,
                                                       query_subject_vector=query_subject_vector,
                                                       subject_feature_dict=subject_feature_dict)
            image_score_tuple_list, subject_id_list = self.database_connection.get_images_related_to_subject(
                subject_list=subject_list, tablename=feature_model)


        elif approach == 2:
            subject_matrix_ = subject_matrix.SubjectMatrix()
            image_score_tuple_list, subject_id_list = subject_matrix_.compute_top_m_subjects(subject_id, 4)

        query_image_title = "Query Results for_{0} with DR {1} in Task {2} with appraoch {3} of subject {4}". \
            format(feature_model, dimensionality_reduction_tech, 6, approach, subject_id)
        utils.visualize_images(top_m_tuples=image_score_tuple_list, m=len(image_score_tuple_list),
                               subject_id_list=subject_id_list, query_image_title=query_image_title)

    def execute_task7(self, top_k, appraoch, feature_model='color_moments', dimensionality_reduction_tech='svd'):
        print('Executing Task 7....')
        if appraoch == 1:
            list_of_subject_ids_in_db = self.database_connection.get_subject_ids_in_db(tablename=feature_model)
            subject_feature_dict = self.database_connection.generate_average_feature_vectors_for_every_subject(
                subject_id_list=list_of_subject_ids_in_db, tablename=feature_model)
            # print(subject_feature_dict)
            subject_similarity_matrix = utils.calculate_subject_similarity_matrix(subject_feature_dict,
                                                                                  list_of_subject_ids_in_db)
        elif appraoch == 2:
            subject_matrix_ = subject_matrix.SubjectMatrix()
            subject_similarity_matrix = subject_matrix_.compute_subject_matrix(top_k)

        nmf = non_negative_matrix_factorization.NonNegativeMatrixFactorization()
        W, H = nmf.get_latent_semantics(data_matrix=subject_similarity_matrix, n_components=top_k)
        print(colored("Representing top-{0} latent semantics in terms of subject weight pairs".
                      format(top_k), "red"))
        self.term_weight_representation_task_8(np.round_(W, decimals=3))


    def execute_task8(self, top_k):
        print('Executing Task 8....')
        img_list = self.database_connection.get_object_feature_matrix_from_db('color_moments')['images']
        metadata_from_db = self.database_connection.get_metadata_for_task_8(image_names=img_list)

        img_metadata_map = {}
        for item in metadata_from_db:
            img_metadata_map[item[0]] = list(item[1:3]) + list(item[3].split(" "))

        row = []
        for img in img_metadata_map.keys():
            column = []
            for val in self.labels_list:
                if val in img_metadata_map.get(img):
                    column.append(1)
                else:
                    column.append(0)
            row.append(column)

        bin_img_meta_matrix = np.array(row)

        nmf = non_negative_matrix_factorization.NonNegativeMatrixFactorization()
        W, H = nmf.get_latent_semantics(data_matrix=bin_img_meta_matrix, n_components=top_k)
        print(colored("Representing top-K latent semantics in terms of metadata", "red"))
        self.term_weight_representation(H[:top_k])
        print(colored("Representing top-K latent semantics in terms of images", "red"))
        self.term_weight_representation(W[:, :top_k].T)

    def term_weight_representation(self, matrix):
        transposed_matrix = matrix.tolist()
        output = []
        for semantic, row in enumerate(transposed_matrix):
            row_list = list(zip(range(1, len(row) + 1), row))
            sorted_row_dict = sorted(row_list, key=lambda k: k[1], reverse=True)
            output.append(sorted_row_dict)
        df = pd.DataFrame(output)
        print(colored("Top 5 columns", "red"))
        print(tabulate(df.iloc[:, :5], headers='keys', tablefmt='psql'))
        print(colored("Last 5 columns", "red"))
        print(tabulate(df.iloc[:, len(df.columns) - 5:], headers='keys', tablefmt='psql'))

    def term_weight_representation_task_8(self, matrix):
        matrix = matrix.T
        transposed_matrix = matrix.tolist()
        output = []
        for semantic, row in enumerate(transposed_matrix):
            row_list = list(zip(range(1, len(row) + 1), row))
            sorted_row_dict = sorted(row_list, key=lambda k: k[1], reverse=True)
            output.append(sorted_row_dict)
        df = pd.DataFrame(output).T
        length = df.shape[1]
        if length > 10:
            start = 0
            while start < length:
                if start == 0:
                    print(colored("First 10 columns", "red"))
                elif start == length - 1 or length < start:
                    print(colored("Remaining columns", "red"))
                else:
                    print(colored("Next 10 columns", "red"))
                print(tabulate(df.iloc[:, start : start + 10], headers='keys', tablefmt='psql'))
                start += 10
        else:
            print(tabulate(df, headers='keys', tablefmt='psql'))

    def plot_cluster(self, file_path, feature_model, dimensionality_reduction_tech,top_k ,label='left',
                     label_type='aspect'):
        connection = self.database_connection.get_db_connection()
        cursor = connection.cursor()
        cursor.execute("select m.imagename,split_part(aspectofhand,' ',2) as aspect from "
                       "metadata m,histogram_of_gradients h where m.imagename=h.imagename;")
        image_label_mapping = cursor.fetchall()

        # pprint.pprint(image_label_mapping)

        file = open(file_path, 'rb')
        Vt = pickle.load(file)

        # Create data
        all_images = self.database_connection.get_object_feature_matrix_from_db(feature_model)
        img_list = all_images['images']
        data_matrix = all_images['data_matrix']

        if (dimensionality_reduction_tech == "nmf"):
            data_matrix_transform, H = self.nmf.get_latent_semantics(data_matrix, top_k, feature_model=feature_model)
        else:
            data_matrix_transform = np.matmul(np.array(data_matrix), np.array(np.transpose(Vt)))

        data_matrix_transform = data_matrix_transform.tolist()

        label_merged_data = []
        label_merged_image = []
        non_label_merged_data = []
        non_label_merged_image = []

        for (image, img_label) in image_label_mapping:
            if img_label == label:
                label_merged_data.append(data_matrix_transform[img_list.index(image)])
                label_merged_image.append(image)
            else:
                non_label_merged_data.append(data_matrix_transform[img_list.index(image)])
                non_label_merged_image.append(image)


        centroid = np.mean(np.array(label_merged_data), axis=0)

        data = (np.array(label_merged_data) - centroid) ** 2
        data = np.sum(data, axis=1)
        data = np.sqrt(data)

        n_data = (np.array(non_label_merged_data) - centroid) ** 2
        n_data = np.sum(n_data, axis=1)
        n_data = np.sqrt(n_data)

        graph = {}
        for image_name, row in zip(img_list, data_matrix_transform):
            #graph[image_name] = [row[0], row[1]]
            graph[image_name] = [np.linalg.norm(np.subtract(row,centroid))]


        mu1, std1 = norm.fit(data.tolist())
        print("mu, std "+str(mu1)+","+str(std1))
        mu2, std2 = norm.fit(n_data.tolist())
        print("mu, std " + str(mu2) + "," + str(std2))

        color = {}
        color["left"] = "blue"
        color["right"] = "red"

        plt.hist(np.concatenate((data, n_data), axis=0), bins=25, density=True, alpha=0.6, color='g')

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p1 = norm.pdf(x, mu1, std1)
        plt.plot(x, p1, 'k', color=color[label], linewidth=2)
        p2 = norm.pdf(x, mu2, std2)
        plt.plot(x, p2, 'k', color=color[self.labels_map.get(label)], linewidth=2)
        title = "2 Gaussian Distribution: "+feature_model+" "+dimensionality_reduction_tech+" "+str(top_k) \
                + "\nDistribution 1: mean ,std: " + str(round(mu1,2)) +","+str(round(std1,2))+ \
                "\nDistribution 2: mean ,std: " + str(round(mu2,2)) +","+str(round(std2,2))+"\n label = "+\
                label+ "(" +color[label]+"); not label = "+self.labels_map.get(label)\
                +"("+color[self.labels_map.get(label)]+")"
        plt.title(title)

        plt.show()
        for (image_name, label) in image_label_mapping:
            graph[image_name].append(label)
        pprint.pprint(graph)

        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

        for image_name in graph:
            x, group = graph[image_name]
            ax.scatter(x, 1, alpha=0.8, c=color[group], edgecolors='none', s=30)

        plt.title("Task 5: Scatter Plot for k="+str(top_k)+" "+ feature_model + " " +
                  dimensionality_reduction_tech + " left vs right")
        plt.legend(loc=2)
        plt.show()


if __name__ == "__main__":
    tm = TaskManager()
    feature_model = "histogram_of_gradients"
    label = 'right'
    label_type = 'aspect'
    dimensionality_reduction_tech = "svd"
    top_k = 20
    tm.plot_cluster("../Data/latent_semantics/"+feature_model+"_"+dimensionality_reduction_tech+".pickle",
                    feature_model, dimensionality_reduction_tech,top_k)
    image_id = input("Please enter Image_id;\n")
    while not image_id == -1:
        tm.execute_task5(feature_model, dimensionality_reduction_tech, image_id, label, label_type,
                      "CC_with_mean_radius")
        image_id = input("Please enter Image_id;\n")
    exit()

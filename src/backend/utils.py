import sys

import glob
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import matplotlib

from database_connection import DatabaseConnection
import os
from pathlib import Path
import pickle
from singular_value_decomposition import SingularValueDecomposition


ALPHA = 0.85
PICKLE_FILE_NAME = "page_rank_interim.pickle"

def get_pickle_directory():
    data_dir = get_data_directory()
    path = str(Path(data_dir + '/pickle'))
    if(not os.path.exists(path)):
        os.mkdir(path)
    return path

def get_data_directory():
    path = str(Path(os.getcwd() + '/src/Data'))
    return path

def get_image_directory(content_type='database_images'):
    data_dir = get_data_directory()
    if content_type == 'database_images':
        return str(Path(data_dir + '/images'))
    elif content_type == 'classification_images':
        return str(Path(data_dir + '/phase3_sample_data'))

def get_dot_distance(vector1, vector2):
    return np.dot(vector1, vector2)

def get_cosine_similarity(vector1, vector2):
    return spatial.distance.cosine(vector1, vector2)

def get_euclidian_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def plot_scree_test(eigen_values):
    num_vars = len(eigen_values)

    fig = plt.figure(figsize=(8, 5))
    sing_vals = np.arange(num_vars) + 1
    plt.plot(sing_vals, eigen_values, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('K latent semantic')
    plt.ylabel('Eigenvalue')

    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    plt.show()

def convert_folder_path_to_table_name(folder_name, pre_string = "metadata"):
    """
     :param folder_name: e.g. /Labelled/Set2
    :param pre_string: pass the string to prepend before the folder name
    :return:
    """

    folder_name = folder_name.replace(" ", "")
    folder_name = folder_name.replace("/", "_")
    folder_name = folder_name.lower()

    if(folder_name[0] == '_'):
        table_name = pre_string + folder_name
    else:
        table_name = pre_string + "_" + folder_name
    return table_name

def get_most_m_similar_images(data_with_images, query_image_feature_vector, Vt, m):
    """
    Author: Vibhu Varshney
    This funcion computes the similarity score between the query image vector and the images in the database
    :param data_with_images_: This is a dict/map with image name list and the data matrix
    :param query_image_feature_vector : Query Image feature vector after applying the feature extraction model
    :param Vt: This is the latent-vector by original feature matrix generated from either model
    :param m: Number of similar images to be returned
    :return: dictionary of m most similar images as keys and their scores as value
    """
    db_data_matrix = data_with_images.get('data_matrix')
    imageNames = data_with_images.get('images')
    database_images_latent_vectors = np.dot(db_data_matrix, np.transpose(Vt))
    query_image_latent_vector = np.dot(np.array(query_image_feature_vector),Vt.T)
    return get_top_m_tuples_by_similarity_score(database_images_latent_vectors,
                                query_image_latent_vector, imageNames, m+1) #+1 because the db contains the query image also

def get_top_m_tuples_by_similarity_score(database_images_latent_vectors, query_image_latent_vector, imageNames, m, distance_measure = "Euclidean"):
    similar_images = get_similarity_score(database_images_latent_vectors, query_image_latent_vector, imageNames, distance_measure)
    if(distance_measure == "cosine"):
        similar_images = sorted(similar_images.items(), key=lambda k: k[1], reverse=True)
    else:
        similar_images = sorted(similar_images.items(), key=lambda k: k[1])
    top_m_tuples = similar_images[:m]
    return top_m_tuples

def get_similarity_score(database_images_latent_vectors, query_image_latent_vector, imageNames, distance_measure = "Euclidean"):
    """
    Author: Vibhu Varshney

    :param database_images_latent_vectors:
    :param query_image_latent_vector:
    :param imageNames:
    :return:
    """
    similar_images = {}
    for i in range(len(database_images_latent_vectors)):
        imageName = imageNames[i]
        db_latent_vector = database_images_latent_vectors[i]
        if(distance_measure == "Euclidean"):
            distance = get_euclidian_distance(query_image_latent_vector, db_latent_vector)
        elif(distance_measure == "dot"):
            distance = get_dot_distance(query_image_latent_vector, db_latent_vector)
        elif (distance_measure == "cosine"):
            distance = get_cosine_similarity(query_image_latent_vector, db_latent_vector)

        similar_images[imageName] = distance
    return similar_images

def save_to_pickle(object_to_save, file_name):
    pickle_directory = get_pickle_directory()
    with open(os.path.join(pickle_directory,file_name), 'wb') as f:
        pickle.dump(object_to_save, f)
    f.close()

def read_from_pickle(file_name):
    pickle_directory = get_pickle_directory()
    with open(os.path.join(pickle_directory,file_name), 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def get_image_names_in_a_folder(relative_folder_path):
    """
    Author: Vibhu Varshney
    :param relative_folder_path: here give the path with a '/' ahead e.g. '/Labelled/Set 2'
    :return:
    list of image names
    """

    data_dir = get_image_directory("classification_images")
    path = str(Path(data_dir + relative_folder_path)) + '/*.jpg'
    files = glob.glob(path)
    image_names = [os.path.basename(x) for x in files]
    return image_names

def get_svd_image_data_from_folder(relative_folder_path, k=10):
    """
    :param relative_folder_path: here give the path with a '/' ahead e.g. '/Labelled/Set2'
    :return:
    data_matrix after applying SVD on it and also the image names present inside the relative_folder_path
    """
    image_names = get_image_names_in_a_folder(relative_folder_path)
    db_conn = DatabaseConnection()
    data_image_dict = db_conn.HOG_descriptor_from_image_ids(image_names)
    data_matrix = data_image_dict['data_matrix']
    svd_obj = SingularValueDecomposition()
    svd_image_data = svd_obj.get_transformed_data(data_matrix, k)
    return svd_image_data, data_image_dict['images']

def get_filtered_images_by_label(labelled_images, filter_by):
    return [x[0] for x in labelled_images if filter_by in x[1]]

def convert_tuple_to_dict(tuple):
    dict = {}
    for each in tuple:
        dict[each[0]] = each[1]

    return dict

def calculate_classification_accuracy(pred_labels, correct_labels):
    cnt = 0
    keys = pred_labels.keys()

    for key in keys:
        if(pred_labels[key] in correct_labels[key]):
            cnt += 1
    print(cnt)
    return (cnt/len(pred_labels))*100

def get_train_and_test_dataframes_from_db(train_table, train_table_metadata, test_table, num_dims=None):

    label_map = {"dorsal": 0, "palmar": 1}

    # retrieve data
    db = DatabaseConnection()
    train_dataset = db.get_object_feature_matrix_from_db(train_table)
    test_dataset = db.get_object_feature_matrix_from_db(test_table)

    # get out data matrix
    train_data = train_dataset['data_matrix']
    train_images = train_dataset['images']
    test_data = test_dataset['data_matrix']
    test_images = test_dataset['images']

    # svd transform
    if num_dims == None:
        tf_train_data = train_data
        tf_test_data = test_data
    else:
        svd = SingularValueDecomposition(num_dims)
        tf_train_data = svd.fit_transform(train_data)
        tf_test_data = svd.transform(test_data)

    # convert list of tuples to dict
    train_labels_map = dict(db.get_correct_labels_for_given_images(train_images, 'aspectOfHand', train_table_metadata))
    exp_test_labels_map = dict(db.get_correct_labels_for_given_images(test_images, 'aspectOfHand'))

    # dataframe setup starts here

    # train_df
    train_col_names = ['imagename', 'hog_svd_descriptor', 'label']
    train_df = pd.DataFrame(columns=train_col_names)

    for i, image in enumerate(train_images):
        temp = train_labels_map[image]
        label = temp.split(' ')[0]

        train_df.loc[len(train_df)] = [image, tf_train_data[i], label_map[label]]

    #test_df
    test_col_names = ['imagename', 'hog_svd_descriptor', 'expected_label', 'predicted_label']
    test_df = pd.DataFrame(columns=test_col_names)

    for i, image in enumerate(test_images):
        temp = exp_test_labels_map[image]
        label = temp.split(' ')[0]

        test_df.loc[len(test_df)] = [image, tf_test_data[i], label_map[label], 'null']

    return train_df, test_df

def get_result_metrics(classifier_name, y_expected, y_predicted):

    y_expected = np.array(y_expected, dtype=int)
    y_predicted = np.array(y_predicted, dtype=int)

    # Predicting the Test set results
    print("Results for classifier {0}".format(classifier_name))
    accuracy = accuracy_score(y_expected, y_predicted)
    print("Accuracy score is: {}".format(accuracy))
    precision = precision_score(y_expected, y_predicted)
    print("Precision score is: {}".format(precision))
    recall = recall_score(y_expected, y_predicted)
    print("Recall score is: {}".format(recall))
    f1 = f1_score(y_expected, y_predicted)
    print("F1 score is: {}".format(f1))
    print("------Confusion Matirx------")
    print(confusion_matrix(y_expected, y_predicted))

    result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return result
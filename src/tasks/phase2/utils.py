import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib
import cv2,  os
from math import sqrt
from database_connection import DatabaseConnection
import psycopg2
import itertools
import collections
import pprint
import os
from pathlib import Path

def get_image_directory():
    return str(Path(os.getcwd()).parent) + '\Data\images'

def get_euclidian_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

# TODO: to be removed entirely? BOW?
def sift_euclidean_comparison(t_db, t_image, image_name_list):

    t = 0
    file_dict = dict()
    while t < t_db.shape[0]:
        
        q_image = t_image
        db_image = t_db[t: t + 32]
        file_name = image_name_list[t // 32]

        min_dists = []
        for query_img_feat in q_image:
            min_dist = np.inf
            for db_img_feat in db_image:
                dist = np.linalg.norm(query_img_feat - db_img_feat)
                if dist < min_dist:
                    min_dist = dist
            min_dists.append(min_dist)

        sum_of_distances = sum(min_dists)

        file_dict[file_name] = sum_of_distances

        t += 32

    for w in sorted(file_dict, key=file_dict.get):
        print(w, "=", file_dict[w])

def read_from_database(model,label=None):
    database_connection = DatabaseConnection()
   
    if label==None:
        img_data_matrix_dict=database_connection.get_object_feature_matrix_from_db(tablename=model)
        return img_data_matrix_dict

def get_dot_distance(vector1, vector2):
    return np.dot(vector1, vector2)


def get_cosine_similarity(vector1, vector2):
    return spatial.distance.cosine(vector1, vector2)

def plot_the_image_on_canvas(row, col, title, imageID, index, figNum = 100, showIndex = False):
    plt.figure(figNum)
    ax = plt.subplot(row, col, index)
    img = cv2.imread(imageID)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    # plt.axis('off')
    if(showIndex):
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(index)
    else:
        plt.axis('off')
        # plt.ylabel(index)
    plt.title(title)



def plot_the_result(top_m_tuples, folder_path, m, notScores = False, subject_id_list=[], query_image_title = "Query Results"):
    """
    Author: Vibhu Varshney
    This function plots the images in a single frame
    :param top_m_tuples: The tuples (image_name, similarity_score/distance) includes the input image as well
    :param folder_path: The path where all images are stored
    :param m: Note here m is the value of top m images but the top_m_tuples contains input image tuple as well so I have
     done m+1 inside the code, you need not to pass the m+1 value
    :return: None. Just plot the images passed
    """
    output_image_path = str(Path(os.getcwd()).parent) + "/Data/Output/"
    index = 1
    m_sqrt = sqrt(m + 1)
    row = round(m_sqrt)
    col = round(m_sqrt)
    if (round(m_sqrt) < m_sqrt):
        col = row + 1
    # plot_the_image_on_canvas(row, col, "Input Image\n" + image_name, image_id, 1)
    j=0
        
    for key, v in top_m_tuples:
        
        append=''
        if len(subject_id_list)==0:
            append=''
        else:
            append=f"Subject ID={subject_id_list[j]}"
        if notScores:
            plot_the_image_on_canvas(row, col, key + "\n" +append+"\n"+ v, folder_path + "/" + key, index)
        else:
            if(index == 1):
                plot_the_image_on_canvas(row, col, "Input Image\n"+append, folder_path + "/" + key, 1)
            else:
                plot_the_image_on_canvas(row, col, key + "\n"+append+"\n" + str(np.around(v, 5)), folder_path + "/" + key, index)
        index += 1
        j+=1
    plt.suptitle(query_image_title)
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.tight_layout()
    # mng.window.state('zoomed')

    plt.savefig("{0}{1}.png".format(output_image_path, "_".join(query_image_title.split())))
    plt.show()

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
    return get_top_m_tuples_by_similarity_score(database_images_latent_vectors, query_image_latent_vector, imageNames, m+1) #+1 because the db contains the query image also

def get_most_3_similar_subjects(Vt,list_of_subject_ids_in_db,query_subject_vector,subject_feature_dict):
    
    data_matrix=[]
    for i in subject_feature_dict.values():
        data_matrix.append(i.reshape((i.shape[1])))
    data_matrix=np.array(data_matrix)        
    subject_images_latent_vectors = np.dot(data_matrix, np.transpose(Vt))
    query_subject_latent_vector = np.dot(query_subject_vector,Vt.T)

    return get_top_m_tuples_by_similarity_score(m=3+1,database_images_latent_vectors=subject_images_latent_vectors,query_image_latent_vector=query_subject_latent_vector,
                                                    imageNames=list_of_subject_ids_in_db)


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


def visualize_images(top_m_tuples, m, subject_id_list=[], query_image_title="Query Results"):
    path = get_image_directory()
    plot_the_result(top_m_tuples, path, m,subject_id_list=subject_id_list, query_image_title=query_image_title)


def calculate_subject_similarity_matrix(subject_feature_dict, list_of_subject_ids_in_db):
    subject_similarity_matrix=[]
    for subject_1 in list_of_subject_ids_in_db:
        subject_sim_row = []
        for subject_2 in list_of_subject_ids_in_db:
            sim_score = get_euclidian_distance(subject_feature_dict.get(subject_1),subject_feature_dict.get(subject_2))
            subject_sim_row.append(sim_score)
        subject_similarity_matrix.append(subject_sim_row)    

    return np.array(subject_similarity_matrix)


def visualize_data_latent_semantics(latent_semantics, image_list, k, top_x=10, title = "Results", data_in_latent_space = None):
    """
    For Extra Credit
    Author: Vibhu Varshney

    :param latent_semantics: This is the latent semantic matric of shape(k, n) where k rows are the latent sematics and
    n columns can be data objects or features based on what you are visualizing
    :param image_list:
    :param k: Keep it below 25 for better visualization
    :param top_x: Number of top 'x' data objects or features that you want to display for each latent semantic. Keep it
    below 10 for better visualization
    :param k:

    :return:
    """
    output_image_path = str(Path(os.getcwd()).parent) + "/Data/Output/"
    if(k>15):
        k=15
        latent_semantics = latent_semantics[:k,:]
    row = top_x
    col = k
    sorted_tuples_images = []

    feature_image_map = {}
    # print(latent_semantics.shape)
    for each_semantic in latent_semantics:
        # print(each_semantic)
        latent_semantic = dict(enumerate(each_semantic))
        # pprint.pprint(latent_semantic)
        sorted_list = sorted(latent_semantic.items(), key=lambda kv: kv[1], reverse=True)
        size_of_list = len(sorted_list)
        sorted_list = sorted_list[:(top_x//2)] + sorted_list[(size_of_list-(top_x//2)):]
        # pprint.pprint(sorted_list)
        tuples_per_latent = []
        # print(len(sorted_list))
        # pprint.pprint(sorted_list)
        latent_feature_vectors = {}
        image_feature_map = collections.defaultdict(list)
        for tuple in sorted_list:
            tuples_per_latent.append((image_list[tuple[0]],tuple[1]))
        sorted_tuples_images.append(tuples_per_latent)

    sorted_tuples_images = list(map(list, zip(*sorted_tuples_images)))
    # print("sorteddddddddddd")
    # pprint.pprint(sorted_tuples_images)
    index = 1
    folder_path = get_image_directory()
    flattened_sorted_tuples_images = list(itertools.chain(*sorted_tuples_images))
    # print("flatttenedddd")
    # pprint.pprint(flattened_sorted_tuples_images)

    for key, v in flattened_sorted_tuples_images:
        plot_the_image_on_canvas(row, col, str(np.around(v, 4)), folder_path + "/" + key, index)
        index += 1
    plt.suptitle(title+"\n"+"X-axis: Top 15 latent semantics, Y-axis: 5 highest and 5 lowest weight data objects")
    # plt.legend([])

    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    plt.savefig("{0}{1}.png".format(output_image_path, "_".join(title.split())), dpi=500)
    plt.show()
    

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
    #leg.draggable()
    plt.show()

def visualize_feature_semantics(latent_semantics, image_list, k, title = "Results", data_in_latent_space = None):
    """
       For Extra Credit
       Author: Vibhu Varshney

       :param latent_semantics: This is the latent semantic matric of shape(k, n) where k rows are the latent sematics and
       n columns can be data objects or features based on what you are visualizing
       :param image_list:
       :param k: Keep it below 25 for better visualization
       :param top_x: Number of top 'x' data objects or features that you want to display for each latent semantic. Keep it
       below 10 for better visualization
       :param k:

       :return:
       """
    output_image_path = str(Path(os.getcwd()).parent) + "/Data/Output/"

    m_sqrt = sqrt(k)
    row = round(m_sqrt)
    col = round(m_sqrt)
    if (round(m_sqrt) < m_sqrt):
        col = row + 1
    sorted_tuples_images = []

    for each_semantic in latent_semantics:
        closest_image = get_top_m_tuples_by_similarity_score(data_in_latent_space,
                                                             each_semantic, image_list,
                                                             1, "cosine")
        # pprint.pprint(closest_image)
        sorted_tuples_images.append(closest_image[0])
    index = 1
    folder_path = get_image_directory()

    for key, v in sorted_tuples_images:
        plot_the_image_on_canvas(row, col, str(np.around(v, 4)), folder_path + "/" + key, index, figNum=200, showIndex=True)
        index += 1
    plt.suptitle(title)

    # mng = plt.get_current_fig_manager()

    # mng.window.state('zoomed')
    plt.savefig("{0}{1}.png".format(output_image_path, "_".join(title.split())))
    plt.show()


def transform_cm(original_cm):
    min_val = np.amin(original_cm)
    print("orig ",original_cm.shape)
    tranformed_cm = original_cm + abs(min_val)
    print("trans ", tranformed_cm.shape)
    # pprint.pprint(tranformed_cm)
    # pprint.pprint(np.amin(tranformed_cm))
    return tranformed_cm


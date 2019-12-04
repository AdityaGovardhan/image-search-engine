import cv2, os
import numpy as np
from scipy.stats import moment
from itertools import repeat
import pickle
import math
import os, sys, inspect
# sys.path.insert(0, '../backend/')
import singular_value_decomposition
from database_connection import DatabaseConnection
from utils import get_most_m_similar_images, get_image_names_from_tuples
from classifiers import support_vector_machine, decision_tree_learning
from locality_sensitive_hashing import LSH
from utils import read_from_pickle
from singular_value_decomposition import SingularValueDecomposition
from backend.pageRank import PageRank


class RelevanceFeedback:
    def __init__(self):
        self.database_connection = DatabaseConnection()
        self.conn = self.database_connection.get_db_connection()
        print('Initiating RelevanceFeedback....')

    def compute_new_query_vector(self, q_old, relevant_items, irrel_items, alpha=0.5, beta=0.45, gamma=0.05):
        print('Computing new query vector.....')

        avg_rel_vec = np.zeros(q_old.shape)
        avg_irl_vec = np.zeros(q_old.shape)

        # Aggregating relevant items
        for item in relevant_items:
            vector = self.database_connection.get_feature_data_for_image('histogram_of_gradients', item)
            avg_rel_vec = avg_rel_vec + vector

        # Aggregating irrelevant items
        for item in irrel_items:
            vector = self.database_connection.get_feature_data_for_image('histogram_of_gradients', item)
            avg_irl_vec = avg_irl_vec + vector

        if len(relevant_items) != 0:
            avg_rel_vec = avg_rel_vec / len(relevant_items)

        if len(irrel_items) != 0:
            avg_irl_vec = avg_irl_vec / len(irrel_items)

        q_new = alpha * q_old + beta * avg_rel_vec - gamma * avg_irl_vec
        return q_new

    def get_user_feedback(self, init_rank_list, q_name, caller='misc'):
        print('Taking user feedback now...')
        rel_items = []
        irl_items = []

        if caller == 'prb':
            for item in init_rank_list[0]:
                if item[0] == q_name:
                    continue
                else:
                    print(f'Is image {item[0]} relevant ? (y/n)')
                    if input() is 'y':
                        rel_items.append(item[0])
                    else:
                        irl_items.append(item[0])
        else:
            for item in init_rank_list:
                if item[0] == q_name:
                    continue
                else:
                    print(f'Is image {item[0]} relevant ? (y/n)')
                    if input() is 'y':
                        rel_items.append(item[0])
                    else:
                        irl_items.append(item[0])

        return rel_items, irl_items

    def get_SVM_based_feedback(self, q, rel_items, irl_items, obj_feature_matrix, m):
        q_new = self.compute_new_query_vector(q_old=q, relevant_items=rel_items, irrel_items=irl_items)
        X_train, Y_train = self.create_X_Y_as_np_matrix(rel_items=rel_items, irl_items=irl_items)

        # Training SVM classifier
        svm = support_vector_machine.SupportVectorMachine()
        svm.fit(X=X_train, y=Y_train)

        # Now getting more test data from LSH indexes
        test_dataset = read_from_pickle('test_dataset.pickle')
        X_test, imageNames = self.create_X_test_as_np_matrix(test_dataset=test_dataset)
        Y_pred = svm.predict(u=X_test)
        relevant_pred_img_names = [imageNames[i] for i in range(0, len(Y_pred)) if Y_pred[i] == 1]
        new_obj_feature_matrix = self.database_connection.HOG_descriptor_from_image_ids(
            image_ids=relevant_pred_img_names)

        new_rank_list = get_most_m_similar_images(data_with_images=new_obj_feature_matrix,
                                                  query_image_feature_vector=q_new, m=m)
        return new_rank_list

    def get_DTC_based_feedback(self, q, rel_items, irl_items, obj_feature_matrix, m):
        q_new=self.compute_new_query_vector(q_old=q,relevant_items=rel_items,irrel_items=irl_items)
        X_train,Y_train=self.create_X_Y_as_np_matrix(rel_items=rel_items,irl_items=irl_items)

        #Training SVM classifier
        dtl = decision_tree_learning.DecisionTreeLearning()
        dtl.fit(X=X_train,y=Y_train)

        # Now getting more test data from LSH indexes
        test_dataset=read_from_pickle('test_dataset.pickle')
        X_test,imageNames=self.create_X_test_as_np_matrix(test_dataset=test_dataset)
        Y_pred = dtl.predict(u=X_test)
        relevant_pred_img_names=[imageNames[i] for i in range(0,len(Y_pred)) if Y_pred[i]==1]
        new_obj_feature_matrix= self.database_connection.HOG_descriptor_from_image_ids(image_ids=relevant_pred_img_names)

        new_rank_list=get_most_m_similar_images(data_with_images=new_obj_feature_matrix,query_image_feature_vector=q_new,m=m)
        return new_rank_list

    def get_PPR_based_feedback(self, q, rel_items, irl_items, obj_feature_matrix, m):
        q_new = self.compute_new_query_vector(q_old=q, relevant_items=rel_items, irrel_items=irl_items)
        topology_images = read_from_pickle('test_dataset.pickle')
        image_names = get_image_names_from_tuples(topology_images)
        db_conn = DatabaseConnection()
        data_image_dict = db_conn.HOG_descriptor_from_image_ids(image_names)
        data_matrix = data_image_dict['data_matrix']
        image_names = data_image_dict['images']
        svd_obj = SingularValueDecomposition()
        svd_image_data = svd_obj.get_transformed_data(data_matrix, 8)  # change this for 11K images

        pg_obj = PageRank()
        image_similarity_matrix = pg_obj.get_image_similarity_matrix_for_top_k_images(6, svd_image_data)
        seed_vector = pg_obj.get_seed_vector(rel_items, image_names)
        pie = pg_obj.get_page_rank_eigen_vector(image_similarity_matrix, seed_vector)
        new_rank_list = pg_obj.get_top_K_images_based_on_scores(pie, image_names, m)

        return new_rank_list

    def get_init_ranking(self, obj_feature_matrix,
                         q):  # For SVM, DTC, PPR.... check calculate_init_prob_similarity for Probab based
        svd = singular_value_decomposition.SingularValueDecomposition()
        data_matrix = obj_feature_matrix['data_matrix']
        U, S, Vt = svd.get_latent_semantics(data_matrix=data_matrix, n_components=25)
        init_rank_list = get_most_m_similar_images(data_with_images=obj_feature_matrix, query_image_feature_vector=q,
                                                   Vt=Vt, m=5)
        return init_rank_list, Vt

    # rel_items,irl_items=rf.get_user_feedback(init_rank_list=init_rank_list,q_name=q_name)
    # q_new=rf.compute_new_query_vector(q_old=q,relevant_items=rel_items,irrel_items=irl_items)
    # new_rank_list=get_most_m_similar_images(data_with_images=obj_feature_matrix,query_image_feature_vector=q_new,Vt=Vt,m=5)

    def get_Vt(self, obj_feature_matrix):  # For SVM, DTC, PPR.... check calculate_init_prob_similarity for Probab based
        svd = singular_value_decomposition.SingularValueDecomposition()
        data_matrix = obj_feature_matrix['data_matrix']
        U, S, Vt = svd.get_latent_semantics(data_matrix=data_matrix, n_components=25)
        return Vt

    def get_probabilistic_relevance_feedback(self, D_matrix, images, q_name, m):
        n_i = self.calculate_n_i(D_matrix=D_matrix)
        init_scores = self.calculate_initial_prob_similarity(D_matrix=D_matrix, images=images, n_i=n_i)
        rel_items, irl_items = self.get_user_feedback(init_rank_list=[init_scores[:m]], q_name=q_name, caller='prb')
        new_rank_list = self.calculate_feedback_prob_similarity(D_matrix=D_matrix, images=images,
                                                                relevant_items=rel_items, n_i=n_i)
        return new_rank_list[:m]

    def calculate_feedback_prob_similarity(self, D_matrix, images, relevant_items, n_i):
        N = D_matrix.shape[0]
        R = len(relevant_items)
        n_i = n_i[0]
        r_i = self.calculate_r_i(D_matrix=D_matrix, images=images, relevant_items=relevant_items)
        r_i = r_i[0]

        feedback_scores = {}
        j = 0
        for d in D_matrix:
            sim_score = 0
            for i in range(0, len(n_i)):
                numerator = (r_i[i] + 0.5) / (R + 1 - r_i[i])
                denominator = (n_i[i] - r_i[i] + 0.5) / (N - R + 1 - n_i[i] + r_i[i])
                sim_score = sim_score + d[i] * math.log2(numerator / denominator)

            feedback_scores[images[j]] = sim_score
            j += 1

        feedback_scores = sorted(feedback_scores.items(), key=lambda k: k[1], reverse=True)
        return feedback_scores

    def calculate_initial_prob_similarity(self, D_matrix, images, n_i):
        N = D_matrix.shape[0]
        n_i = n_i[0]

        init_scores = {}

        j = 0
        for d in D_matrix:
            sim_score = 0
            for i in range(0, len(n_i)):
                sim_score = sim_score + d[i] * math.log2((N - n_i[i] + 0.5) / (n_i[i] + 0.5))

            init_scores[images[j]] = sim_score
            j += 1

        init_scores = sorted(init_scores.items(), key=lambda k: k[1], reverse=True)

        return init_scores

    def calculate_r_i(self, D_matrix, images, relevant_items):
        r_i = np.zeros((1, D_matrix.shape[1]))
        i = 0
        for row in D_matrix:
            temp = [1 if row[x] > 0 and images[i] in relevant_items else 0 for x in range(0, len(row))]
            r_i = r_i + np.array(temp).T
            i += 1

        return r_i

    def calculate_n_i(self, D_matrix):

        n_i = np.zeros((1, D_matrix.shape[1]))
        for row in D_matrix:
            temp = [1 if row[x] > 0 else 0 for x in range(0, len(row))]
            n_i = n_i + np.array(temp).T

        return n_i

    def create_X_Y_as_np_matrix(self, rel_items, irl_items):
        X = []
        Y = []

        # Adding relevant items in X and Y
        for item in rel_items:
            fv = self.database_connection.get_feature_data_for_image('histogram_of_gradients', item)
            X.append(fv.reshape(fv.shape[1]))
            Y.append(1)

        # Adding irrelevant items in X and Y
        for item in irl_items:
            fv = self.database_connection.get_feature_data_for_image('histogram_of_gradients', item)
            X.append(fv.reshape(fv.shape[1]))
            Y.append(-1)

        return np.array(X), np.array(Y)

    def create_X_test_as_np_matrix(self, test_dataset):
        X = []
        imageNames = []
        # Adding relevant items in X and Y
        for item in test_dataset:
            fv = self.database_connection.get_feature_data_for_image('histogram_of_gradients', item[0])
            X.append(fv.reshape(fv.shape[1]))
            imageNames.append(item[0])

        return np.array(X), imageNames

if __name__ == '__main__':
    rf = RelevanceFeedback()
    # q_name='Hand_0000012.jpg'
    # q=rf.database_connection.get_feature_data_for_image('histogram_of_gradients',q_name)
    # obj_feature_matrix=rf.database_connection.get_object_feature_matrix_from_db('histogram_of_gradients')
    # init_ranking,Vt=rf.get_init_ranking(obj_feature_matrix=obj_feature_matrix,q=q)
    # new_rank_list=rf.get_SVM_based_feedback(init_rank_list=init_ranking,q=q,q_name=q_name,Vt=Vt)
    svm = support_vector_machine.SupportVectorMachine()
    svm.plot()


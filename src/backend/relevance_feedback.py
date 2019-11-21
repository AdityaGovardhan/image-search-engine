import cv2, os
import numpy as np
from scipy.stats import moment
from itertools import repeat
import pickle

import os,sys,inspect
# sys.path.insert(0, '../backend/')
import singular_value_decomposition
from database_connection import DatabaseConnection
from utils import get_most_m_similar_images

class RelevanceFeedback:

	def __init__(self):
		self.database_connection = DatabaseConnection()
		self.conn = self.database_connection.get_db_connection()
		print('Initiating RelevanceFeedback....')

	def compute_new_query_vector(self,q_old,relevant_items,irrel_items,alpha=0.5,beta=0.45,gamma=0.05):
		print('Computing new query vector.....')
		
		avg_rel_vec=np.zeros(q_old.shape)
		avg_irl_vec=np.zeros(q_old.shape)

		#Aggregating relevant items
		for item in relevant_items:
			vector=self.database_connection.get_feature_data_for_image('histogram_of_gradients',item)
			avg_rel_vec=avg_rel_vec+vector

		#Aggregating irrelevant items
		for item in irrel_items:
			vector=self.database_connection.get_feature_data_for_image('histogram_of_gradients',item)
			avg_irl_vec=avg_irl_vec+vector

		avg_rel_vec=avg_rel_vec/len(relevant_items)
		avg_irl_vec=avg_irl_vec/len(irrel_items)		

		q_new= alpha*q_old + beta*avg_rel_vec - gamma*avg_irl_vec
		return q_new

	def get_user_feedback(self,init_rank_list,q_name):
		print('Taking user feedback now...')
		rel_items=[]
		irl_items=[]
		for item in init_rank_list:

			if item[0] == q_name:
				continue
			else:	
				print(f'Is image {item[0]} relevant ? (y/n)')
				if input() is 'y':
					rel_items.append(item[0])
				else:
					irl_items.append(item[0])	 

		return rel_items,irl_items

	def get_SVM_based_feedback(self):
		pass

	def get_Dec_Tree_based_feedback(self):
		pass

	def get_PPR_based_feedback(self):
		pass		

if __name__ == '__main__':		
	rf=RelevanceFeedback()
	q_name='Hand_0000012.jpg'
	q=rf.database_connection.get_feature_data_for_image('histogram_of_gradients',q_name)
	obj_feature_matrix=rf.database_connection.get_object_feature_matrix_from_db('histogram_of_gradients')
	data_matrix=obj_feature_matrix['data_matrix']	
	svd=singular_value_decomposition.SingularValueDecomposition()
	U,S,Vt=svd.get_latent_semantics(data_matrix=data_matrix,n_components=25)
	init_rank_list=get_most_m_similar_images(data_with_images=obj_feature_matrix,query_image_feature_vector=q,Vt=Vt,m=5)
	rel_items,irl_items=rf.get_user_feedback(init_rank_list=init_rank_list,q_name=q_name)
	q_new=rf.compute_new_query_vector(q_old=q,relevant_items=rel_items,irrel_items=irl_items)
	new_rank_list=get_most_m_similar_images(data_with_images=obj_feature_matrix,query_image_feature_vector=q_new,Vt=Vt,m=5)
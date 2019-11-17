import numpy as np
from singular_value_decomposition import SingularValueDecomposition
from database_connection import DatabaseConnection
from utils import get_most_m_similar_images,visualize_images
import pprint
import sys


class SubjectMatrix:
    def __init__(self):
        self.database_connection = DatabaseConnection()

    def get_subject_image_mapping(self):
        connection = self.database_connection.get_db_connection()
        cursor = connection.cursor()
        cursor.execute("select id,m.imagename, aspectofhand from metadata m,histogram_of_gradients h where m.imagename=h.imagename order by id;")
        list_of_mapping = cursor.fetchall()
        #pprint.pprint(list_of_mapping)
        mapping = {}
        for image in list_of_mapping:
            (sub_id,imagename,ascept) = image

            # [ascept1,ascept2] = ascept.split(' ')
            if sub_id not in mapping:
                mapping[sub_id] = {}
            # if accessories not in mapping[sub_id]:
            #     mapping[sub_id][accessories] = [imagename]
            # else:
            #     mapping[sub_id][accessories].append(imagename)
            # if gender not in mapping[sub_id]:
            #     mapping[sub_id][gender] = [imagename]
            # else:
            #     mapping[sub_id][gender].append(imagename)
            # if ascept1 not in mapping[sub_id]:
            #     mapping[sub_id][ascept1] = [imagename]
            # else:
            #     mapping[sub_id][ascept1].append(imagename)
            if ascept not in mapping[sub_id]:
                mapping[sub_id][ascept] = [imagename]
            else:
                mapping[sub_id][ascept].append(imagename)

        # pprint.pprint(mapping)
        return mapping,list_of_mapping

    def get_hog_svd_result(self,feature_model= 'histogram_of_gradients', dimensionality_reduction_tech="svd",top_k=10, label=None,label_type=None):
        latent_semantic_file_path = "./../Data/latent_semantics/" + feature_model + '_' + dimensionality_reduction_tech + '.pickle'

        imglist_object_feature_matrix_dict = self.database_connection.get_object_feature_matrix_from_db(
            tablename="histogram_of_gradients", label=label, label_type=label_type)
        #print(imglist_object_feature_matrix_dict)
        if dimensionality_reduction_tech == 'svd':
            svd = SingularValueDecomposition()
            U, S, Vt = svd.get_latent_semantics(data_matrix=imglist_object_feature_matrix_dict.get('data_matrix'),
                                                n_components=top_k)
            #print(Vt[:top_k].shape)
        return imglist_object_feature_matrix_dict,Vt

    def wieght_average(self, list_of_distance):
        ratio_list  = [1, 0.9, 0.8, 0.7]

        list_of_distance.sort()
        rank = 1
        wt_avg = list_of_distance[0]

        base = float(list_of_distance[0])

        max_limit = min(len(ratio_list), len(list_of_distance))
        while rank < max_limit and (list_of_distance[rank]/base) <= 2 - ratio_list[rank]:
            wt_avg += (ratio_list[rank] * list_of_distance[rank])
            #print(list_of_distance[rank]/base,":",wt_avg)
            rank += 1

        return (wt_avg/float(rank))

    def compute_top_m_subjects(self, subject_id, top_m):
        mapping,list_of_mapping = self.get_subject_image_mapping()
        (imglist_object_feature_matrix_dict, Vt) = self.get_hog_svd_result()
        # pprint.pprint(mapping)
        list_of_images = imglist_object_feature_matrix_dict['images']
        transfrom_datamtrix = np.array(imglist_object_feature_matrix_dict['data_matrix'])*np.transpose(Vt)
        transfrom_datamtrix = transfrom_datamtrix.tolist()
        min_distance={}

        for subject in mapping:
            all_category = {}
            for category in mapping[subject]:
                if category in mapping[subject_id]:
                    all_category[category]= []
                    for image_cmp in mapping[subject][category]:
                        for image_base in mapping[subject_id][category]:
                            image1 = np.array(transfrom_datamtrix[list_of_images.index(image_base)])
                            image2 = np.array(transfrom_datamtrix[list_of_images.index(image_cmp)])
                            distance = np.linalg.norm(image1-image2)
                            #if distance < all_category[category]:
                            all_category[category].append(distance)
                    all_category[category] = self.wieght_average(all_category[category])
            if len(all_category)>0:
                min_distance[subject] = sum(all_category.values())/len(all_category)

        rank_subject = sorted(min_distance.items(), key=lambda k: k[1])
        #pprint.pprint(rank_subject)

        top_m_tuples = []
        subject_mat_list = []
        for (sub_base, score) in rank_subject[0:top_m]:
            for (sub_new,image,_) in list_of_mapping:
                if sub_base == sub_new:
                    top_m_tuples.append((image,score))
                    subject_mat_list.append(sub_base)
        #pprint.pprint(top_m_tuples)
        # visualize_images(top_m_tuples,len(top_m_tuples),subject_mat_list)
        return top_m_tuples, subject_mat_list

    def compute_subject_matrix(self, top_k):
        mapping,list_of_mapping = self.get_subject_image_mapping()
        (imglist_object_feature_matrix_dict, Vt) = self.get_hog_svd_result()
        # pprint.pprint(mapping)
        list_of_images = imglist_object_feature_matrix_dict['images']
        transfrom_datamtrix = np.array(imglist_object_feature_matrix_dict['data_matrix'])*np.transpose(Vt)
        transfrom_datamtrix = transfrom_datamtrix.tolist()
        subject_mat = []
        row_no = 0
        for subject1 in mapping:
            # subject_mat.append([])
            temp_row = []
            for subject2 in mapping:
                all_category = {}
                for category in mapping[subject2]:
                    if category in mapping[subject1]:
                        all_category[category]= []
                        for image_cmp in mapping[subject2][category]:
                            for image_base in mapping[subject1][category]:
                                image1 = np.array(transfrom_datamtrix[list_of_images.index(image_base)])
                                image2 = np.array(transfrom_datamtrix[list_of_images.index(image_cmp)])
                                distance = np.linalg.norm(image1-image2)
                                #if distance < all_category[category]:
                                all_category[category].append(distance)
                        all_category[category] = self.wieght_average(all_category[category])
                if len(all_category) > 0:
                    # subject_mat[row_no].append(sum(all_category.values())/len(all_category))
                    temp_row.append(sum(all_category.values())/len(all_category))
                else:
                    temp_row.append(1000000.0)
            subject_mat.append(temp_row)

        return np.array(subject_mat)



"""
create table temp as select id,m.imagename from metadata m,histogram_of_gradients h where m.imagename=h.imagename;
\copy temp  to '/tmp/temp.csv' with CSV DELIMITER ',';
"""

if __name__=="__main__":
    sm = SubjectMatrix()
    top_m = 3
    top_k = 10

    sub_id = int(input("Please enter the subject id (-1 to exit):\n"))
    while not sub_id ==-1:
        a, b = sm.compute_top_m_subjects(sub_id,top_m)
        sub_id = int(input("Please enter the subject id (-1 to exit):\n"))
    # print(sm.wieghted_average([100,129,105,115]))
    sm.compute_subject_matrix(top_k)



import psycopg2
import numpy as np
import pickle
'''
    This class has database connection class.
    This class have get_db_connection method which will return database cursor.

'''


class DatabaseConnection:

    label_type_map = {
        "aspect":"aspectofhand",
        "gender":"gender",
        "accessories":"accessories"
    }

    def get_db_connection(self):
        psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
        connection = psycopg2.connect(host='localhost', port='5432', user='postgres',
                                password='postgres', dbname='Hands_db')
        return connection

    def create_feature_model_table(self, tablename):
        connection = self.get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""DROP Table IF EXISTS {};""".format(tablename))
        # Create metadata table
        cursor.execute("""CREATE TABLE IF NOT EXISTS {0}(
                                imageName TEXT NOT NULL,
                                features BYTEA NOT NULL,
                                PRIMARY KEY (imageName)
                                );
                                """.format(tablename))
        connection.commit()

    def insert_feature_data(self, tablename, feature_vectors):
        connection = self.get_db_connection()
        cursor = connection.cursor()
        cursor.executemany("""INSERT INTO {0}(imageName,features) VALUES (%(imageName)s, %(features)s)""".
                           format(tablename), feature_vectors)
        connection.commit()

    def get_feature_data_for_image(self, tablename, imageName):
        """
        Returns the feature vector for an image for a model
        Output Shape: [feature vector]
        """
        connection = self.get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT features from {0} where imagename = '{1}'".format(tablename, imageName))
        result_row = cursor.fetchone()
        return np.array(pickle.loads(result_row[0]))

    def get_subject_ids_in_db(self, tablename):
        """
        Returns the feature vector for an image for a model
        Output Shape: [feature vector]
        """
        connection = self.get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT id from {0} natural join metadata group by id ORDER BY id;".format(tablename))
        result_row = cursor.fetchall()
        subject_id_list=[i[0] for i in result_row]        
        return subject_id_list

    def generate_average_feature_vectors_for_every_subject(self,subject_id_list, tablename):
        """
        Returns average feature vectors for all subjects in DB
        & the average feature vector for the search subject id
        """
        subject_feature_dict={}
        connection = self.get_db_connection()
        cursor = connection.cursor()
        search_feature=np.array(0)

        for subject_id in subject_id_list:
            # print(subject_id)
            cursor.execute("SELECT features from {0} natural join metadata where id={1};".format(tablename, subject_id))
            result_row=cursor.fetchall()
            aggregated_feature_vectors=np.array([pickle.loads(i[0]) for i in result_row])
            average_feature_vectors=np.average(aggregated_feature_vectors,axis=0)           
            
            subject_feature_dict[subject_id]=average_feature_vectors

        return subject_feature_dict

    def get_images_related_to_subject(self,subject_list,tablename):
        """
        """
        image_score_tuple_list=[]
        connection = self.get_db_connection()
        cursor = connection.cursor()
        subject_id_list = []
        for subject_id in subject_list:
            cursor.execute("SELECT imagename from {0} natural join metadata where id={1};".format(tablename,
                                                                                                  subject_id[0]))
            result_column=cursor.fetchall()
            image_score_tuple_list+=[(i[0],subject_id[1]) for i in result_column]
            subject_id_list+=[subject_id[0] for i in result_column]

        return image_score_tuple_list,subject_id_list

    ##### Temporarily for task 6 #########
    def get_feture_all_subjects(self,tablename):
        connection = self.get_db_connection()
        cursor = connection.cursor()
        query ="SELECT meta.id,feat_table.features FROM "+tablename+" as feat_table join metadata as meta on " \
                                                                    "feat_table.imagename = meta.imagename"
        cursor.execute(query)
        db_output = cursor.fetchall()
        obj_feature_matrix = {}
        for image_row in db_output:
            if(not image_row[0] in  obj_feature_matrix):
                obj_feature_matrix[image_row[0]] = []
            obj_feature_matrix[image_row[0]].append(pickle.loads(image_row[1]))
        return obj_feature_matrix
    

    def get_all_subjects(self,tablename):
        connection = self.get_db_connection()
        cursor = connection.cursor()
        query ="SELECT meta.id,count(feat_table.imagename) FROM "+tablename+" as feat_table join metadata as meta on " \
                                                                            "feat_table.imagename = meta.imagename " \
                                                                            "group by meta.id"
        cursor.execute(query)
        db_output = cursor.fetchall()
        return db_output

    #### end ################################    

    def get_object_feature_matrix_from_db(self, tablename, label=None, label_type=None):
        """
        Returns the object feature(Data) matrix for dimensionality reduction for a model
        Output Shape: [# of images in db, feature length]
        """
        connection = self.get_db_connection()
        cursor = connection.cursor()
        if not label_type:
            query = "SELECT * from {0}".format(tablename)
        elif label_type == "aspect":
            query = "SELECT * FROM {0} WHERE imagename IN  (SELECT imagename FROM metadata WHERE aspectofhand " \
                    "LIKE '%{1}%')".format(tablename, label)
        elif label_type == "gender":
            query = "SELECT * FROM {0} WHERE imagename IN  (SELECT imagename FROM metadata WHERE gender = '{1}')".\
                format(tablename, label)
        elif label_type == "accessories":
            query = "SELECT * FROM {0} WHERE imagename IN  (SELECT imagename FROM metadata WHERE accessories = '{1}')".\
                format(tablename, label)
        elif label_type == "subject":
            query = "SELECT * FROM {0} WHERE imagename IN  (SELECT imagename FROM metadata WHERE id = '{1}')".\
                format(tablename, label)
        cursor.execute(query)
        db_output = cursor.fetchall()
        obj_feature_matrix = []
        images = []
        for image_row in db_output:
            images.append(image_row[0])
            obj_feature_matrix.extend(pickle.loads(image_row[1]))
        return {"images": images, "data_matrix": np.array(obj_feature_matrix)}

    def get_correct_labels_for_given_images(self, image_names = None, label_type=None):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        tablename = 'metadata'
        if not image_names:
            query = "SELECT imagename, {1} FROM {0}".format(tablename, self.label_type_map.get(label_type))
        else:
            query = "SELECT imagename, {1} FROM {0} WHERE imagename IN {2}".format(tablename, self.label_type_map.
                                                                                   get(label_type), tuple(image_names))
        cursor.execute(query)
        result = cursor.fetchall()
        # print("res = ",result)
        return result

    def get_metadata_for_task_8(self,image_names=None):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        tablename='metadata'
        query = "SELECT imagename,gender,accessories,aspectofhand FROM {0} WHERE imagename IN {1}"\
            .format(tablename, tuple(image_names))
        cursor.execute(query)
        result = cursor.fetchall()
        return result


if __name__ == "__main__":
    database_connection = DatabaseConnection()
    conn = database_connection.get_db_connection()
    o_f_mat = database_connection.get_object_feature_matrix_from_db('color_moments')['data_matrix'].shape
    print(database_connection.get_feature_data_for_image('color_moments', 'Hand_0008110.jpg'))
    print(o_f_mat)

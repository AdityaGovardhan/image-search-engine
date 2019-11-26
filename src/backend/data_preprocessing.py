from database_connection import DatabaseConnection
import os
from histogram_of_gradients import HistogramOfGradients
from multiprocessing import Process
from utils import get_image_directory
import os
from pathlib import Path


class DataPreProcessor:
    def __init__(self):
        '''
        1) Read each image from data dir and put metadata in database table.
        2) Pass that image to each feature model to get a vector of dimension (1 * m)
        3) Pass this vector to all the dimension reduction technique to get latent semantics.
        4) Put this latent semantic of each image from each model and technique with its metadata inside database
        '''

        self.database_connection = DatabaseConnection()

        self.process_metadata()
        self.process_classification_metadata()

        self.DATABASE_IMAGES_PATH = get_image_directory('database_images')
        self.CLASSIFICATION_IMAGES_PATH = get_image_directory('classification_images')

        feature_models = []

        feature_models.append("histogram_of_gradients")

        processes = []
        for i, feature in enumerate(feature_models):
            processes.append(Process(target=self.perform_feature_model(feature)))
            processes[i].start()

        for i in range(len(processes)):
            processes[i].join()

    # This function will read all the metadata of input images and put those metadata details in database.
    def process_metadata(self):
        csv_file_path = os.getcwd()[:-7] + 'Data/HandInfo.csv'
        connection = self.database_connection.get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""DROP Table IF EXISTS metadata;""")
        # Create metadata table
        cursor.execute("""CREATE TABLE IF NOT EXISTS metadata( 
                        id INT NOT NULL, 
                        age INT NOT NULL, 
                        gender TEXT NOT NULL, 
                        skinColor TEXT NOT NULL, accessories INT NOT NULL,
                        nailPolish INT NOT NULL,
                        aspectOfHand TEXT NOT NULL,
                        imageName TEXT NOT NULL,
                        irregularities INT NOT NULL,
                        PRIMARY KEY (imageName)
                        );
                        """)
        # file opened to avoid permission error in linux
        with open(csv_file_path, 'r') as f:
            next(f)
            cursor.copy_from(f, 'metadata', sep=',', null='')
        # cursor.execute("""copy metadata from '{}' csv header;""".format(csv_file_path))
        connection.commit()

    def process_classification_metadata(self):

        metadata_files = ['labelled_set1.csv', 'labelled_set2.csv', 'unlabelled_set1.csv', 'unlabelled_set2.csv']

        data_folder = os.getcwd()[:-7] + 'Data/phase3_sample_data/'

        connection = self.database_connection.get_db_connection()

        cursor = connection.cursor()

        for metadata_file in metadata_files:
            table_name = metadata_file.split('.')[0]
            metadata_file_path = data_folder + metadata_file
            
            cursor.execute("DROP Table IF EXISTS " + table_name + ";")

            cursor.execute("""CREATE TABLE IF NOT EXISTS """ + table_name + """(
                            some_number INT,
                            id INT NOT NULL,
                            age INT NOT NULL,
                            gender TEXT NOT NULL,
                            skinColor TEXT NOT NULL,
                            accessories INT NOT NULL,
                            nailPolish INT NOT NULL,
                            aspectOfHand TEXT,
                            imageName TEXT NOT NULL,
                            irregularities INT NOT NULL,
                            PRIMARY KEY (imageName)
                            );
                            """)
            # file opened to avoid permission error in linux
            with open(metadata_file_path, 'r') as f:
                next(f)
                cursor.copy_from(f, table_name, sep=',', null='')
            # cursor.execute("""copy """ + table_name + """ from '{}' csv header;""".format(metadata_file_path))
            connection.commit()

    def perform_feature_model(self, feature):
        histogram_of_gradients = HistogramOfGradients()
        feature_vectors = histogram_of_gradients.get_image_vectors(self.DATABASE_IMAGES_PATH)

        self.database_connection.create_feature_model_table(feature)
        self.database_connection.insert_feature_data(feature, feature_vectors)

    def perform_feature_model_for_classification(self, feature):
        # writing this
        pass


if __name__ == "__main__":
    print('Preprocessing....')
    data_preprocessor = DataPreProcessor()
    print('Preprocessed!')

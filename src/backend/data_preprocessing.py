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

        self.INPUT_DATA_PATH = str(Path(str(Path(os.getcwd()).parent) + "/Data/images/"))
        self.database_connection = DatabaseConnection()
        self.process_metadata()
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
        csv_file_path = os.getcwd()[:-7] + '/Data/HandInfo.csv'
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
        cursor.execute("""copy metadata from '{}' csv header;""".format(csv_file_path))
        connection.commit()

    def perform_feature_model(self, feature):
        histogram_of_gradients = HistogramOfGradients(self.INPUT_DATA_PATH)
        feature_vectors = histogram_of_gradients.get_image_vectors()

        self.database_connection.create_feature_model_table(feature)
        self.database_connection.insert_feature_data(feature, feature_vectors)


if __name__ == "__main__":
    print('Preprocessing....')
    data_preprocessor = DataPreProcessor()

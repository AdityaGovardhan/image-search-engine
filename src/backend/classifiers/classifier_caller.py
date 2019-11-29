from backend.classifiers import support_vector_machine, decision_tree_learning

# import sys
import numpy as np
import pandas as pd
# sys.path.append('..')

import backend.utils as utils

class ClassifierCaller:
    def __init__(self, classifier_name, training_dataset, testing_dataset):
        self.classifier_name = classifier_name
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.test_df = None
        self.result = None

    def call_classifier(self):
        if(self.classifier_name  == "Personalized Page Rank"):
            # Talk to Vibhu
            # images_with_labels, accuracy = ppr_obj.get_predicted_labels(labelled_folder_path, unlabelled_folder_path)
            return
        else:
            if (self.classifier_name  == "Support Vector Machine"):
                classifier = support_vector_machine.SupportVectorMachine()

            elif (self.classifier_name  == "Decision Tree Classifier"):
                classifier = decision_tree_learning.DecisionTreeLearning()
        
        k = 15

        train_table = 'histogram_of_gradients_labelled_' + self.training_dataset.lower()
        train_table_metadata = 'metadata_labelled_' + self.training_dataset.lower()
        test_table = 'histogram_of_gradients_unlabelled_' + self.testing_dataset.lower()

        train_df, self.test_df = utils.get_train_and_test_dataframes_from_db(train_table, train_table_metadata, test_table, num_dims=k)
        X_train, y_train = np.vstack(train_df['hog_svd_descriptor'].values), train_df['label'].to_numpy(dtype=int)

        classifier.fit(X_train, y_train)
        self.test_df['predicted_label'] = classifier.predict(np.vstack(self.test_df['hog_svd_descriptor'].values))


    def get_result(self):
        # Talk to Vibhu about returning (image_name, label(type = dorsal/palmar? 0/1?)) 
        self.result = utils.get_result_metrics(self.classifier_name, self.test_df['expected_label'], self.test_df['predicted_label'])

        return self.result
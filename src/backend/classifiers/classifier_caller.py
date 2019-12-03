from backend.classifiers import support_vector_machine, decision_tree_learning
from sklearn.svm import SVC
import numpy as np
import pandas as pd
# sys.path.append('..')
import backend.utils as utils


class ClassifierCaller:
    def __init__(self, classifier_name, training_dataset, testing_dataset, kernel, dimensionality_algo="svd"):
        self.classifier_name = classifier_name
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.test_df = None
        self.result = None
        self.algo = dimensionality_algo
        self.kernel = kernel

    def call_classifier(self):
        if self.classifier_name == "Support Vector Machine":
            if not self.kernel:
                self.kernel = 'linear'
            classifier = support_vector_machine.SupportVectorMachine(kernel=self.kernel)
            self.algo = "sift"

        elif self.classifier_name == "Decision Tree Classifier":
            classifier = decision_tree_learning.DecisionTreeLearning()
            self.algo = "sift"

        if self.algo == "sift":
            train_table = 'sift_labelled_' + self.training_dataset.lower()
            train_table_metadata = 'metadata_labelled_' + self.training_dataset.lower()
            test_table = 'sift_unlabelled_' + self.testing_dataset.lower()

            train_df, self.test_df = utils.get_train_and_test_dataframes_from_db(train_table, train_table_metadata,
                                                                                 test_table, algo=self.algo)

        elif self.algo == "pca":
            k = 10

            train_table = 'local_binary_pattern_labelled_' + self.training_dataset.lower()
            train_table_metadata = 'metadata_labelled_' + self.training_dataset.lower()
            test_table = 'local_binary_pattern_unlabelled_' + self.testing_dataset.lower()
            train_df, self.test_df = utils.get_train_and_test_dataframes_from_db(train_table, train_table_metadata,
                                                                                 test_table, num_dims=k, algo="pca")

        elif self.algo == "svd":
            k = 15
            train_table = 'histogram_of_gradients_labelled_' + self.training_dataset.lower()
            train_table_metadata = 'metadata_labelled_' + self.training_dataset.lower()
            test_table = 'histogram_of_gradients_unlabelled_' + self.testing_dataset.lower()

            train_df, self.test_df = utils.get_train_and_test_dataframes_from_db(train_table, train_table_metadata,
                                                                                 test_table, num_dims=k, algo="svd")
        X_train, y_train = np.vstack(train_df['hog_svd_descriptor'].values), train_df['label'].to_numpy(
            dtype=int)

        classifier.fit(X_train, y_train)
        self.test_df['predicted_label'] = classifier.predict(
            np.vstack(self.test_df['hog_svd_descriptor'].values))

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        x = clf.predict(np.vstack(self.test_df['hog_svd_descriptor'].values))
        result = utils.get_result_metrics("self.classifier_name", self.test_df['expected_label'],
                                          x)
        print("results---------------------------------------&&&&&&&&&&&&&")
        print(result)

    def get_result(self):
        self.result = utils.get_result_metrics(self.classifier_name, self.test_df['expected_label'],
                                               self.test_df['predicted_label'])
        print(self.result)
        images_with_labels = []
        for index, row in self.test_df.iterrows():
            if row['predicted_label'] == -1:
                images_with_labels.append((row['imagename'], 'dorsal'))
            else:
                images_with_labels.append((row['imagename'], 'palmar'))
        return self.result, images_with_labels

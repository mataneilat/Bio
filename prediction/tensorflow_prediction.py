
"""
    This module contains the predictor which uses tensor flow for hinge prediction
"""
import tensorflow as tf
from tensorflow import keras

from nma.gnm_analysis import *
from prediction.prediction_commons import *
from utils import *
import time
from benchmark import Benchmark


class TensorFlowPredictor(HingePredictor):
    """
    This predictor tries to employ machine learning techniques on the local inter-residue correlations in order to
    predict whether a given residues is a hinge or not
    """
    def __init__(self, local_sensitivity):
        """
        Constructs the predictor with the local sensitivity factor.
        :param      local_sensitivity: An integer representing the number of preceding and succeeding residues we wish to
                    consider
        """
        self.local_sensitivity = local_sensitivity
        self.model = None

    def _prepare_train_data(self, morphs_repository, train_morphs_ids, contact_map_repository):
        """
        Prepares the data used for model training.

        :param  morphs_repository:  The repository containing the morphs, i.e. the pdb files specifying the protein
                                    structures
        :param  train_morphs_ids:   The ids of the morphs that should be used from training.
        :param  contact_map_repository: The repository containing the contact maps corresponding to the morphs.
                                        In case this parameter is None, contact maps are not used as a part of training.
        :return: A tuple containing the train data and the train labels
        """
        train_data = []
        train_labels = []

        local_sensitivity = self.local_sensitivity

        def add_training_data_for_morph(morph, file_path, ubi, header):
            nonlocal train_data, train_labels

            k_inv = None
            if contact_map_repository is None:
                k_inv = calc_gnm_k_inv(ubi, header, None)
            else:
                # We should use the contact map
                contact_map = contact_map_repository.get_contact_map_rr(morph.morph_id, len(ubi))
                if contact_map is None:
                    # Contact map is not found, continue
                    return
                k_inv = calc_gnm_k_inv(ubi, header, contact_map=contact_map)

            (m,n) = k_inv.shape

            annotated_hinges = morph.get_hinges()

            morph_train_data = []
            morph_train_labels = []

            for i in range(local_sensitivity, m - local_sensitivity):
                sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, local_sensitivity)
                sub_matrix -= np.mean(sub_matrix)
                morph_train_data.append(sub_matrix)
                morph_train_labels.append(1 if i in annotated_hinges else 0)

            train_data += morph_train_data
            train_labels += morph_train_labels

        morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in train_morphs_ids,
                                                              add_training_data_for_morph)

        return train_data, train_labels

    def train_model(self, morphs_repository, train_morphs_ids, contact_map_repository):
        """
        Creates and trains the model for prediction.
        This method must be called before using the predict_hinges method.

        :param  morphs_repository:  The repository containing the morphs, i.e. the pdb files specifying the protein
                                    structures
        :param  train_morphs_ids:   The ids of the morphs that should be used from training.
        :param  contact_map_repository: The repository containing the contact maps corresponding to the morphs.
                                        In case this parameter is None, contact maps are not used as a part of training.
        """
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=tf.nn.relu,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4, activation=tf.nn.relu,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4)),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

        train_data, train_labels = self._prepare_train_data(morphs_repository, train_morphs_ids, contact_map_repository)

        model.fit(np.array(train_data), np.array(train_labels), epochs=5)

        self.model = model

    def _predict_confidence_levels(self, k_inv):
        """
        Predict confidence level for each resudie being a hinge using the trained model
        :param k_inv:   The inverted matrix produced by the GNM analysis which is interpreted as a correlation matrix.
        :return:    The predicted confidence levels.
        """
        if self.model is None:
            raise RuntimeError("The model was not trained")

        m = k_inv.shape[0]
        predictions = [0] * m
        for i in range(self.local_sensitivity, m - self.local_sensitivity):
            sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, self.local_sensitivity)
            sub_matrix -= np.mean(sub_matrix)
            sub_matrix_wrapped = np.array([sub_matrix])
            result = self.model.predict(sub_matrix_wrapped)
            predictions[i] = result[0][0]
        return predictions


    def predict_hinges(self, k_inv):
        """
        Predicts the hinge residues.
        :param  k_inv: The inverted matrix produced by the GNM analysis which is interpreted as a correlation matrix.
        :return:    The predicted hinge residues
        """
        before_tf = time.time()
        predicted_confidence_levels = self._predict_confidence_levels(k_inv)
        after_tf = time.time()
        Benchmark().update(k_inv.shape[0], 'Tensor Flow', after_tf - before_tf)
        return predict_hinges(predicted_confidence_levels, self.local_sensitivity, 90, 95, 0)


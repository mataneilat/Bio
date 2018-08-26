
from prody import *
import numpy as np
from bio.cast import *
from bio.hinge_atlas_parser import *
import os
from pathlib import Path
import operator
import matplotlib.pyplot as plt
from bio.utils import *
import tensorflow as tf
from tensorflow import keras

atlas_morphs = parse_hinge_atlas_text('./hingeatlas.txt')
atlas_directory = '/Users/mataneilat/Downloads/hinge_atlas_nonredundant'

sensitivity = 7

def get_hinges_default(ubi, header, cutoff=10, modeIndex=0):
    gnm = GNM('Ubiquitin')
    gnm.buildKirchhoff(ubi, cutoff=cutoff)
    gnm.calcModes()
    return gnm.getHinges(modeIndex)

class GammaRaptor(Gamma):

    def __init__(self, raptor_matrix):
        self.raptor_matrix = raptor_matrix

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        raptor_score = self.raptor_matrix[i, j]
        alpha = 0.8
        return alpha * raptor_score + (1 - alpha) * (1 - dist2 / 100.0)


class GammaDistance(Gamma):

    def __init__(self):
        pass

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        s = 1 - dist2 / 64.0
        return s * s


def calc_data_for_sub_matrix(sub_matrix):
    (n,m) = sub_matrix.shape

    pivot = int((m - 1) / 2)
    left_up = sub_matrix[0:pivot,0:pivot].copy()
    right_bottom = sub_matrix[pivot+1:m,pivot+1:m].copy()
    cross = sub_matrix[0:pivot,pivot+1:n].copy()

    return left_up.flatten().tolist() + right_bottom.flatten().tolist() + cross.flatten().tolist()

def calc_gnm_k_inv(ubi, header, raptor_matrix):
    gnm = GNM('Ubiquitin')
    if raptor_matrix is None:
        gnm.buildKirchhoff(ubi, cutoff=8, gamma=GammaDistance())
    else:
        gnm.buildKirchhoff(ubi, cutoff=8, gamma=GammaRaptor(raptor_matrix))

    gnm.calcModes()

    if gnm._array is None:
        raise ValueError('Modes are not calculated.')
    # obtain the eigenvectors
    V = gnm._array
    eigvals = gnm._eigvals
    (m, n) = V.shape

    k_inv = np.zeros((m,m))
    for i in range(2):
        eigenvalue = eigvals[i]
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)
    return k_inv


def prepare_train_data(training_morphs_list):

    train_data = []
    train_labels = []

    def add_training_data_for_morph(morph, ubi, header):
        nonlocal train_data, train_labels
        if morph.morph_id not in training_morphs_list:
            return

        k_inv = calc_gnm_k_inv(ubi, header, None)
        (m,n) = k_inv.shape

        annotated_hinges = morph.get_hinges()

        morph_train_data = []
        morph_train_labels = []

        for i in range(sensitivity, m-sensitivity):
            morph_train_data.append(get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity))
            morph_train_labels.append(1 if i in annotated_hinges else 0)

        train_data += morph_train_data
        train_labels += morph_train_labels

    perform_for_all_morphs(add_training_data_for_morph)

    return train_data, train_labels


def train_model(training_morphs_list):

    dim = sensitivity * 2 + 1
    model = keras.Sequential([
       # keras.layers.BatchNormalization(),
       keras.layers.Conv2D(data_format="channels_last", input_shape=(dim,dim,1),
                           filters=8, kernel_size=[2, 2], activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    train_data, train_labels = prepare_train_data(training_morphs_list)

    # Reshape data for model
    train_data = np.array(train_data)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)

    model.fit(train_data, np.array(train_labels), epochs=5)

    return model


def get_predictions(model, ubi, header):
    k_inv = calc_gnm_k_inv(ubi, header, None)
    (m,n) = k_inv.shape
    predictions = [0] * m
    for i in range(sensitivity, m-sensitivity):
       # residue_data = calc_data_for_sub_matrix(get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity))
        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity)
        sub_matrix_wrapped = np.array([sub_matrix])
        result = model.predict(sub_matrix.reshape(sub_matrix_wrapped.shape[0], sub_matrix_wrapped.shape[1],
                                                  sub_matrix_wrapped.shape[2], 1))
        predictions[i] = result[0][0]
    return predictions


def plot_predictions(model, ubi, header):
    predictions = get_predictions(model, ubi, header)
    plt.plot(predictions)
    plt.ylabel('some numbers')
    plt.show()


def perform_on_morph(morph_id, func, **func_kwargs):
    morph = atlas_morphs.get(morph_id)
    if morph is None:
        return
    ff0_path = '%s/%s/ff0.pdb' % (atlas_directory, morph_id)
    if Path(ff0_path).is_file():
        with open(ff0_path, 'r') as pdb_file:
            ubi, header = parsePDB(ff0_path, subset='calpha', header=True)
            func(morph, ubi, header, **func_kwargs)


def perform_for_all_morphs(func, **kwargs):
    directory = os.fsencode(atlas_directory)

    for file in os.listdir(directory):
        morph_filename = os.fsdecode(file)
        perform_on_morph(morph_filename, func, **kwargs)


def print_morph_hinges_results(the_model, morph_id, raptor_file=None):

    def plot_results(morph, ubi, header, model):
        print("Morph: ", morph.morph_id)
        print("Annotated Hinges: ", morph.get_hinges())
        print("Default Hinges: ", get_hinges_default(ubi, header))
        print("LEARNED HINGES: ", plot_predictions(model, ubi, header))

    perform_on_morph(morph_id, plot_results, model=the_model)


def print_all_morphs_results(model):

    directory = os.fsencode(atlas_directory)

    for file in os.listdir(directory):
        morph_filename = os.fsdecode(file)
        print_morph_hinges_results(model, morph_filename)


def collect_prediction_scores(the_model, morphs_test_list):
    results = {}

    def collect_predictions(morph, ubi, header, model):
        nonlocal results
        if morph.morph_id not in morphs_test_list:
            return

        predictions = get_predictions(model, ubi, header)
        results[morph.morph_id] = predictions

    perform_for_all_morphs(collect_predictions, model=the_model)
    return results



def predict_hinges(model, morphs_to_prediction_scores):
    morph_to_hinges = {}
    for morph, prediction_scores in morphs_to_prediction_scores.items():
        m = len(prediction_scores)
        hinges = []

        total_max = max(prediction_scores)
        for i in range(sensitivity, m-sensitivity):
            prediction_subset = prediction_scores[max(0, i - sensitivity):min(m, i + sensitivity)]
            subset_max = max(prediction_subset)
            subset_mean = np.mean(np.array(prediction_subset))
            alpha = 0.9
            if prediction_scores[i] < alpha * subset_max + (1-alpha) * subset_mean:
                continue
            if prediction_scores[i] < 0.7 * total_max:
                continue
            hinges.append(i)
        morph_to_hinges[morph] = hinges
    return morph_to_hinges

def score_prediction(predicted_hinges, annotated_hinges, total_residue_count):
    prediction_range = 2
    predicted_annotated_list = [False] * len(predicted_hinges)
    annotated_predicted = 0
    for annotated_hinge in annotated_hinges:
        was_predicted = False
        for i, predicted_hinge in enumerate(predicted_hinges):
            if annotated_hinge in range(predicted_hinge - prediction_range, predicted_hinge + prediction_range + 1):
                was_predicted = True
                predicted_annotated_list[i] = True
        if was_predicted:
            annotated_predicted += 1
    predicated_annotated = sum([1 if x else 0 for x in predicted_annotated_list])
    annotated_predicted_percentage = annotated_predicted / len(annotated_hinges)
    predicated_annotated_percentage = predicated_annotated / len(predicted_hinges)

    punish = float(len(predicted_hinges)) / total_residue_count

    return 0.8 * annotated_predicted_percentage + 0.2 * predicated_annotated_percentage - punish


def main():
    morphs_ids = list(atlas_morphs.keys())

    train_morph_ids = morphs_ids[:150]
    test_morph_ids = morphs_ids[150:]
    model = train_model(train_morph_ids)
    prediction_scores = collect_prediction_scores(model, test_morph_ids)

    morph_to_hinges = predict_hinges(model, prediction_scores)

    total_ml_score = 0
    total_default_score = 0

    def print_prediction_results(morph, ubi, header, hinges_dict):
        nonlocal total_ml_score, total_default_score
        if morph.morph_id not in hinges_dict:
            return

        total_residue_count = len(morph.residues)

        ml_hinges = hinges_dict[morph.morph_id]
        ml_score = score_prediction(ml_hinges, morph.get_hinges(), total_residue_count)
        total_ml_score += ml_score

        default_hinges = get_hinges_default(ubi, header)
        default_score = score_prediction(default_hinges, morph.get_hinges(), total_residue_count)
        total_default_score += default_score

        print("Annotated Hinges: ", morph.get_hinges())
        print("DEFAULT HINGES: ", default_hinges, default_score)
        print("ML HINGES:", ml_hinges, ml_score)


    perform_for_all_morphs(print_prediction_results, hinges_dict=morph_to_hinges)
    print("ML SCORE IS: ", total_ml_score)
    print("DEFAULT SCORE IS: ", total_default_score)


if __name__ == '__main__':
    main()

#    print_morph_hinges_results(model, atlas_morphs, atlas_directory, '06487-15304')
#    print_all_morphs_results(model)

#     predictions = collect_predictions_all_morphs(model)
#
#     subset_of_predictions_for_training = {k:predictions[k] for k in list(atlas_morphs.keys())[20:] if predictions.get(k) is not None}
#     subset_of_predictions_for_plotting = {k:predictions[k] for k in list(atlas_morphs.keys())[:20] if predictions.get(k) is not None}
#
#     model2 = train_model_2(subset_of_predictions_for_training)
#
#     get_model2_predictions(model2, subset_of_predictions_for_plotting)









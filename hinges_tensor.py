
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
        s = 1 - dist2 / 100.0
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
        gnm.buildKirchhoff(ubi, cutoff=10, gamma=GammaDistance())
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
    for i in range(1):
        eigenvalue = eigvals[i]
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)
    return k_inv


def prepare_train_data_for_morph(atlas_morphs, atlas_directory, morph_id):

    train_data = []
    train_labels = []

    morph = atlas_morphs.get(morph_id)
    if morph is None:
        return None, None

    annotated_hinges = morph.get_hinges()

    ff0_path = '%s/%s/ff0.pdb' % (atlas_directory, morph_id)
    if Path(ff0_path).is_file():
        with open(ff0_path, 'r') as pdb_file:

            ubi, header = parsePDB(ff0_path, subset='calpha', header=True)
            k_inv = calc_gnm_k_inv(ubi, header, None)

           # k_inv = normalize_matrix(k_inv)

            (m,n) = k_inv.shape
            for i in range(sensitivity, m-sensitivity):
               # residue_data = calc_data_for_sub_matrix(get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity))
                train_data.append(get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity))
                train_labels.append(1 if i in annotated_hinges else 0)

    return train_data, train_labels



def prepare_train_data():

    directory = os.fsencode(atlas_directory)

    train_data = []
    train_labels = []

    for file in os.listdir(directory):
        morph_filename = os.fsdecode(file)
        morph_train_data, morph_train_labels = prepare_train_data_for_morph(atlas_morphs, atlas_directory, morph_filename)
        if morph_train_data is not None and morph_train_labels is not None:
            train_data += morph_train_data
            train_labels += morph_train_labels

    train_data = np.array(train_data)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
    return train_data, train_labels

def train_model():

    dim = sensitivity * 2 + 1
    model = keras.Sequential([
       # keras.layers.BatchNormalization(),
        keras.layers.Conv2D(data_format="channels_last", input_shape=(dim,dim,1),
                            filters=1, kernel_size=[3, 3], activation=tf.nn.relu),
      #  keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])

    train_data, train_labels = prepare_train_data()

    model.fit(train_data, np.array(train_labels), epochs=5)

    return model


def prepare_model_2_data_for_residue(prediction_mean, prediction, i, is_annotated_hinge):
  #  diff = prediction[i] - prediction_mean
    data = prediction[i-10:i+10]

    label = 0
    if is_annotated_hinge:
        label = 1
    return np.array(data), label


def prepare_data_for_model_2(model1_predictions):

    train_data = []
    train_labels = []
    for morph, prediction in model1_predictions.items():

        annotated_hinges = atlas_morphs[morph].get_hinges()

        m = len(prediction)

        prediction_mean = np.mean(prediction)

        for i in range(sensitivity + 5, m-sensitivity - 5):
            is_annotated = i in annotated_hinges
            residue_data, residue_label = prepare_model_2_data_for_residue(prediction_mean, prediction, i, is_annotated)
            train_data.append(residue_data)
            train_labels.append(residue_label)

    return np.array(train_data), train_labels


def train_model_2(model1_predictions):

    train_data, train_labels = prepare_data_for_model_2(model1_predictions)

    model = keras.Sequential([
       # keras.layers.BatchNormalization(),

        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])

    model.fit(train_data, np.array(train_labels), epochs=5)

    return model



def get_model2_predictions(model, model1_predictions):

    for morph, prediction in model1_predictions.items():

        annotated_hinges = atlas_morphs[morph].get_hinges()
        print("ANNOTATED HINGES:", annotated_hinges)

        m = len(prediction)

        prediction_mean = np.mean(prediction)

        predictions = [0] * m

        for i in range(sensitivity + 5, m-sensitivity - 5):
            is_annotated = False
            if i in annotated_hinges:
                is_annotated = True
            residue_data, residue_label = prepare_model_2_data_for_residue(prediction_mean, prediction, i, is_annotated)
            result = model.predict(np.array([residue_data]))
            predictions[i] = result

        # PLOT
        plt.plot(predictions)
        plt.ylabel('some numbers')
        plt.show()



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

    def plot_results(morph, ubi, header, model=model):
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


def collect_predictions_all_morphs(the_model):
    results = {}

    def collect_predictions(morph, ubi, header, model):
        nonlocal results
        predictions = get_predictions(model, ubi, header)
        results[morph.morph_id] = predictions

    perform_for_all_morphs(collect_predictions, model=the_model)
    return results



def predict_hinges(model):
    predictions = collect_predictions_all_morphs(model)
    morph_to_hinges = {}
    for morph, prediction in predictions.items():
        m = len(prediction)
        hinges = []

        total_max = max(prediction)
        for i in range(sensitivity, m-sensitivity):
            prediction_subset = prediction[max(0, i - sensitivity):min(m, i + sensitivity)]
            subset_max = max(prediction_subset)
            subset_mean = np.mean(np.array(prediction_subset))
            alpha = 0.9
            if prediction[i] < alpha * subset_max + (1-alpha) * subset_mean:
                continue
            if prediction[i] < 0.7 * total_max:
                continue
            hinges.append(i)
        morph_to_hinges[morph] = hinges
    return morph_to_hinges


if __name__ == '__main__':
    model = train_model()
    morph_to_hinges = predict_hinges(model)

    def print_prediction_results(morph, ubi, header, hinges_dict):
        print("PREDICTED:", hinges_dict[morph.morph_id])
        print("Annotated Hinges: ", morph.get_hinges())

    perform_for_all_morphs(print_prediction_results, hinges_dict=morph_to_hinges)

#    print_morph_hinges_results(model, atlas_morphs, atlas_directory, '06487-15304')
    print_all_morphs_results(model)

#     predictions = collect_predictions_all_morphs(model)
#
#     subset_of_predictions_for_training = {k:predictions[k] for k in list(atlas_morphs.keys())[20:] if predictions.get(k) is not None}
#     subset_of_predictions_for_plotting = {k:predictions[k] for k in list(atlas_morphs.keys())[:20] if predictions.get(k) is not None}
#
#     model2 = train_model_2(subset_of_predictions_for_training)
#
#     get_model2_predictions(model2, subset_of_predictions_for_plotting)









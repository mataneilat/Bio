
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


def prepare_train_data_for_morph(atlas_morphs, atlas_directory, morph_id):

    train_data = []
    train_labels = []
    filtered_morphs = list(filter(lambda atlas_morph: atlas_morph.morph_id == morph_id, atlas_morphs))
    if len(filtered_morphs) == 0:
        print('Could not find morph %s in atlas annotations' % morph_id)
        return [], []
    if len(filtered_morphs) > 1:
        print('That is weird!')
        return
    morph = filtered_morphs[0]
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

    i = 0
    for file in os.listdir(directory):
        i += 1
        if i < 10:
            continue
        if i > 200:
            break
        morph_filename = os.fsdecode(file)
        morph_train_data, morph_train_labels = prepare_train_data_for_morph(atlas_morphs, atlas_directory, morph_filename)
        train_data += morph_train_data
        train_labels += morph_train_labels

    train_data = np.array(train_data)
    print(train_data.shape)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
    print(train_data.shape)
    return train_data, train_labels

def train_model():

        # keras.layers.Dense(8, activation=tf.nn.relu),
     #
       # keras.layers.Reshape([-1, 15 ,15 ,1]),
    model = keras.Sequential([
       # keras.layers.BatchNormalization(),
        keras.layers.Conv2D(data_format="channels_last", input_shape=(15,15,1),
                            filters=32, kernel_size=[5, 5], activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    train_data, train_labels = prepare_train_data()

    print("TRAIN:", train_data.shape)
    model.fit(train_data, np.array(train_labels), epochs=5)

    return model


def get_learned_hinges(model, ubi, header):
    k_inv = calc_gnm_k_inv(ubi, header, None)

    (m,n) = k_inv.shape

    results = [0] * m
    for i in range(sensitivity, m-sensitivity):
       # residue_data = calc_data_for_sub_matrix(get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity))
        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity)
        sub_matrix_wrapped = np.array([sub_matrix])
        result = model.predict(sub_matrix.reshape(sub_matrix_wrapped.shape[0], sub_matrix_wrapped.shape[1],
                                                  sub_matrix_wrapped.shape[2], 1))
        results[i] = result
        # print(i, result)
    plt.plot(results)
    plt.ylabel('some numbers')
    plt.show()


def print_morph_hinges_results(model, atlas_morphs, atlas_directory, morph_id, raptor_file=None):
    filtered_morphs = list(filter(lambda atlas_morph: atlas_morph.morph_id == morph_id, atlas_morphs))
    if len(filtered_morphs) == 0:
        print('Could not find morph %s in atlas annotations' % morph_id)
        return
    if len(filtered_morphs) > 1:
        print('That is weird!')
        return
    morph = filtered_morphs[0]
    ff0_path = '%s/%s/ff0.pdb' % (atlas_directory, morph_id)
    if Path(ff0_path).is_file():
        with open(ff0_path, 'r') as pdb_file:
            ubi, header = parsePDB(ff0_path, subset='calpha', header=True)
            print("Morph: ", morph_id)
            print("Annotated Hinges: ", morph.get_hinges())
            print("LEARNED HINGES: ", get_learned_hinges(model, ubi, header))


def print_all_morphs_results(model):

    directory = os.fsencode(atlas_directory)

    for file in os.listdir(directory):
        morph_filename = os.fsdecode(file)
        print_morph_hinges_results(model, atlas_morphs, atlas_directory, morph_filename)


if __name__ == '__main__':
    model = train_model()
#    print_morph_hinges_results(model, atlas_morphs, atlas_directory, '06487-15304')
    print_all_morphs_results(model)





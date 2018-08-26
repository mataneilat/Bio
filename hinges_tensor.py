
from prody import *
from bio.morphs_atlas_parser import *
from bio.morphs_repository import *
import os
import matplotlib.pyplot as plt
from bio.utils import *
from bio.prediction_score import *
from bio.gnm_utils import *
import tensorflow as tf
from tensorflow import keras

morphs_repository = MorphsRepository(parse_morphs_atlas_from_text('./hingeatlas.txt'),
                                    '/Users/mataneilat/Downloads/hinge_atlas_nonredundant')

sensitivity = 7

def prepare_train_data(training_morphs_list):

    train_data = []
    train_labels = []

    def add_training_data_for_morph(morph, ubi, header):
        nonlocal train_data, train_labels

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

    morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in training_morphs_list,
                                                          add_training_data_for_morph)

    return train_data, train_labels


def train_model(training_morphs_list):

    dim = sensitivity * 2 + 1
    model = keras.Sequential([
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


def predict_confidence_levels(model, ubi, header):
    k_inv = calc_gnm_k_inv(ubi, header, None)
    (m,n) = k_inv.shape
    predictions = [0] * m
    for i in range(sensitivity, m-sensitivity):
        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, sensitivity)
        sub_matrix_wrapped = np.array([sub_matrix])
        result = model.predict(sub_matrix.reshape(sub_matrix_wrapped.shape[0], sub_matrix_wrapped.shape[1],
                                                  sub_matrix_wrapped.shape[2], 1))
        predictions[i] = result[0][0]
    return predictions


def predict_morphs_hinges(the_model, morphs_test_list):
    results = {}

    def collect_predictions(morph, ubi, header, model):
        nonlocal results
        if morph.morph_id not in morphs_test_list:
            return

        predicted_confidence_levels = predict_confidence_levels(model, ubi, header)
        predicted_hinges = predict_hinges(predicted_confidence_levels)
        results[morph.morph_id] = predicted_hinges

    morphs_repository.perform_on_all_morphs_in_directory(collect_predictions, model=the_model)
    return results


def main():
    morphs_ids = list(morphs_repository.atlas_morphs.keys())

    train_morph_ids = morphs_ids[:150]
    test_morph_ids = morphs_ids[150:]
    model = train_model(train_morph_ids)

    morph_to_predicted_hinges = predict_morphs_hinges(model, test_morph_ids)

    total_ml_score = 0
    total_default_score = 0

    def print_prediction_results(morph, ubi, header, hinges_dict):
        nonlocal total_ml_score, total_default_score

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

    morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in test_morph_ids,
            print_prediction_results, hinges_dict=morph_to_predicted_hinges)

    print("ML SCORE IS: ", total_ml_score)
    print("DEFAULT SCORE IS: ", total_default_score)


if __name__ == '__main__':
    main()









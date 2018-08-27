
from prody import *
import numpy as np
from bio.cast import *
from bio.morphs_atlas_parser import *
import os
from pathlib import Path
import operator
from bio.utils import *
from bio.gamma_functions import *
import itertools
from bio.gnm_utils import *
from bio.prediction_score import *
from bio.morphs_repository import *
import math
import matplotlib.pyplot as plt


morphs_repository = MorphsRepository(parse_morphs_atlas_from_text('./hingeatlas.txt'),
                                    '/Users/mataneilat/Downloads/hinge_atlas_nonredundant')

local_sensitivity = 7

def get_my_hinges_with_correlation_simplest(ubi, header, raptor_matrix):
    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    correlations = [0] * m
    for i in range(local_sensitivity, m-local_sensitivity):
        curr_correlations = k_inv[i,i-local_sensitivity:i+local_sensitivity]
        correlations[i] = math.exp(-np.sum(curr_correlations))

    return predict_hinges(correlations)


def get_my_gnm_hinges_with_correlation_distance(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    correlations = [0] * m
    for i in range(m):
        for d, k in itertools.product(range(local_sensitivity), range(local_sensitivity)):
            if i-k >= 0 and i+d < m:
                correlations[i] += np.linalg.norm(k_inv[i-k,:] - k_inv[i+d,:])

    return predict_hinges(correlations)


def get_hinges_using_cross_correlation_avgs(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    prediction_scores = [0] * m

    for i in range(1, m-1):

        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, local_sensitivity)

        left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

        np.fill_diagonal(left_up, 0)
        np.fill_diagonal(right_bottom, 0)

        left_up_avg = np.average(left_up)
        right_bottom_avg = np.average(right_bottom)
        cross_avg = np.average(cross)

        prediction_scores[i] = left_up_avg + right_bottom_avg - 2 * cross_avg

    return predict_hinges(prediction_scores)


def get_hinges_using_cross_correlation_avg(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header)
    (n,m) = k_inv.shape

    prediction_scores = [0] * m

    for i in range(1, m - 1):

        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, local_sensitivity)

        left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

        cross_avg = np.average(cross)

        prediction_scores[i] = math.exp(-cross_avg)

    return predict_hinges(prediction_scores, max_mean_alpha=0.95, total_max_beta=0.95)



def get_hinges_using_cross_correlation_negative(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header)
    (n,m) = k_inv.shape

    expected_negative_percentage = 0.7
    hinges = []

    for i in range(local_sensitivity, m - local_sensitivity):

        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, local_sensitivity)

        left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

        cross_negative_count = len(cross[cross < 0])

        negative_percentage = cross_negative_count / cross.size

        if negative_percentage > expected_negative_percentage:
            hinges.append(i)

    return hinges


def main():
    morphs_ids = list(morphs_repository.atlas_morphs.keys())

    test_morph_ids = morphs_ids[150:]

    total_my_score_1 = 0
    total_my_score_2 = 0
    total_my_score_3 = 0
    total_my_score_4 = 0
    total_my_score_5 = 0
    total_default_score = 0

    def print_prediction_results(morph, ubi, header):
        nonlocal total_my_score_1, total_my_score_2, total_my_score_3, total_my_score_4, total_my_score_5, total_default_score

        total_residue_count = len(morph.residues)

        predicted_hinges_1 = get_hinges_using_cross_correlation_negative(ubi, header, None)
        my_score_1 = score_prediction(predicted_hinges_1, morph.get_hinges(), total_residue_count)
        total_my_score_1 += my_score_1

        predicted_hinges_2 = get_hinges_using_cross_correlation_avg(ubi, header, None)
        my_score_2 = score_prediction(predicted_hinges_2, morph.get_hinges(), total_residue_count)
        total_my_score_2 += my_score_2

        predicted_hinges_3 = get_hinges_using_cross_correlation_avgs(ubi, header, None)
        my_score_3 = score_prediction(predicted_hinges_3, morph.get_hinges(), total_residue_count)
        total_my_score_3 += my_score_3

        predicted_hinges_4 = get_my_gnm_hinges_with_correlation_distance(ubi, header, None)
        my_score_4 = score_prediction(predicted_hinges_4, morph.get_hinges(), total_residue_count)
        total_my_score_4 += my_score_4

        predicted_hinges_5 = get_my_hinges_with_correlation_simplest(ubi, header, None)
        my_score_5 = score_prediction(predicted_hinges_5, morph.get_hinges(), total_residue_count)
        total_my_score_5 += my_score_5

        default_hinges = get_hinges_default(ubi, header)
        default_score = score_prediction(default_hinges, morph.get_hinges(), total_residue_count)
        total_default_score += default_score

        print("Annotated Hinges: ", morph.get_hinges())
        print("DEFAULT HINGES: ", default_hinges, default_score)
        print("MY HINGES 1:", predicted_hinges_1, my_score_1)
        print("MY HINGES 2:", predicted_hinges_2, my_score_2)
        print("MY HINGES 3:", predicted_hinges_3, my_score_3)
        print("MY HINGES 4:", predicted_hinges_4, my_score_4)
        print("MY HINGES 5:", predicted_hinges_5, my_score_5)


    morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in test_morph_ids,
            print_prediction_results)

    print("MY SCORE 1 IS: ", total_my_score_1)
    print("MY SCORE 2 IS: ", total_my_score_2)
    print("MY SCORE 3 IS: ", total_my_score_3)
    print("MY SCORE 4 IS: ", total_my_score_4)
    print("MY SCORE 5 IS: ", total_my_score_5)
    print("DEFAULT SCORE IS: ", total_default_score)

if __name__ == "__main__":
    main()
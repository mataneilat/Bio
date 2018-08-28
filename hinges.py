
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
from bio.contact_map_repository import ContactMapRepository
import math
import matplotlib.pyplot as plt


morphs_repository = MorphsRepository(parse_morphs_atlas_from_text('./hingeatlas.txt'),
                                    '/Users/mataneilat/Downloads/hinge_atlas_nonredundant')

local_sensitivity = 7

def get_my_hinges_with_correlation_simplest(k_inv):

    (m,n) = k_inv.shape

    correlations = [0] * m
    for i in range(local_sensitivity, m-local_sensitivity):
        curr_correlations = k_inv[i,i-local_sensitivity:i+local_sensitivity]
        correlations[i] = math.exp(-np.sum(curr_correlations))

    return predict_hinges(correlations)


def get_my_gnm_hinges_with_correlation_distance(k_inv):

    (m,n) = k_inv.shape

    correlations = [0] * m
    for i in range(m):
        for d, k in itertools.product(range(local_sensitivity), range(local_sensitivity)):
            if i-k >= 0 and i+d < m:
                correlations[i] += np.linalg.norm(k_inv[i-k,:] - k_inv[i+d,:])

    return predict_hinges(correlations)


def get_hinges_using_cross_correlation_avgs(k_inv):

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


def get_hinges_using_cross_correlation_avg(k_inv):

    (n,m) = k_inv.shape

    prediction_scores = [0] * m

    for i in range(1, m - 1):

        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, local_sensitivity)

        left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

        cross_avg = np.average(cross)

        prediction_scores[i] = math.exp(-cross_avg)

    return predict_hinges(prediction_scores, max_mean_alpha=0.95, total_max_beta=0.95)



def get_hinges_using_cross_correlation_negative(k_inv):

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

    contact_map_repository = ContactMapRepository('/Users/mataneilat/Documents/BioInfo/raptor_output/contact_maps')
    morphs_ids = list(morphs_repository.atlas_morphs.keys())

    test_morph_ids = morphs_ids

    methods = [('Negative Count', get_hinges_using_cross_correlation_negative),
               ('Near Cross Correlations Avg', get_hinges_using_cross_correlation_avg),
               ('Near Correlations Score', get_hinges_using_cross_correlation_avgs),
               ('Correlation Distances', get_my_gnm_hinges_with_correlation_distance),
               ('Simplest', get_my_hinges_with_correlation_simplest)]

    total_scores_without_contact_map = [0] * len(methods)
    total_scores_with_contact_map = [0] * len(methods)

    total_default_score = 0

    def collect_prediction_results(morph, file_path, ubi, header):
        nonlocal methods, total_scores_without_contact_map, total_scores_with_contact_map, total_default_score

        total_residue_count = len(morph.residues)

        contact_map = contact_map_repository.get_contact_map_old(morph.morph_id, len(ubi))
        if contact_map is None:
            return

        k_inv = calc_gnm_k_inv(ubi, header)
        k_inv_with_contact = calc_gnm_k_inv(ubi, header, raptor_matrix=contact_map)

        for i, method in enumerate(methods):
            desc = method[0]
            func = method[1]

            predicted_hinges_without_contact_map = func(k_inv)
            score_without_contact_map = score_prediction(predicted_hinges_without_contact_map,
                                                         morph.get_hinges(), total_residue_count)
            print(desc, "without contact map", predicted_hinges_without_contact_map, score_without_contact_map)
            total_scores_without_contact_map[i] += score_without_contact_map

            predicted_hinges_with_contact_map = func(k_inv_with_contact)
            score_with_contact_map = score_prediction(predicted_hinges_with_contact_map,
                                                         morph.get_hinges(), total_residue_count)
            print(desc, "with contact map", predicted_hinges_with_contact_map, score_with_contact_map)
            total_scores_with_contact_map[i] += score_with_contact_map

        default_hinges = get_hinges_default(ubi, header)
        default_score = score_prediction(default_hinges, morph.get_hinges(), total_residue_count)
        total_default_score += default_score



    morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in test_morph_ids,
            collect_prediction_results)

    for i, method in enumerate(methods):
        desc = method[0]
        print("Total Score for", desc, "without contact map", total_scores_without_contact_map[i])
        print("Total Score for", desc, "with contact map", total_scores_with_contact_map[i])
    print("Total default score:", total_default_score)


if __name__ == "__main__":
    main()
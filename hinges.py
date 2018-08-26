
from prody import *
import numpy as np
from bio.cast import *
from bio.morphs_atlas_parser import *
import os
from pathlib import Path
import operator
from bio.utils import *
from bio.gamma_functions import *
from bio.gnm_utils import *
from bio.prediction_score import *
from bio.morphs_repository import *
import matplotlib.pyplot as plt


morphs_repository = MorphsRepository(parse_morphs_atlas_from_text('./hingeatlas.txt'),
                                    '/Users/mataneilat/Downloads/hinge_atlas_nonredundant')

local_sensitivity = 7

def normalize_matrix(M):
    if M.shape[0] == 0:
        return None
    return M / np.abs(M).max()


def find_local_maxima(arr):
    np_arr = np.array(arr)
    diff_arr = np.sign(np.diff(np_arr))
    k = 5
    t = 4
    return [i for i in range(k, len(arr) - k) if (np.sum(diff_arr[i-k:i]) >= t and np.sum(diff_arr[i:i+k]) <= -t)]


def insert_to_ranges(ranges, idx):
    if len(ranges) == 0:
        ranges.append(Range(idx,idx))
        return
    last_range = ranges[-1]
    if last_range.end_idx + 1 == idx:
        ranges[-1] = Range(last_range.start_idx, idx)
        return
    ranges.append(Range(idx,idx))


def create_nested_ranges(last_ranges, current_ranges):
    nested_ranges = []
    did_nest_range = [False] * len(last_ranges)
    for current_range in current_ranges:
        is_within_any = False
        for i, last_range in enumerate(last_ranges):
            if current_range.is_within(last_range):
                did_nest_range[i] = True
                is_within_any = True
        if is_within_any:
            nested_ranges.append(current_range)
    for i in range(len(did_nest_range)):
        if not did_nest_range[i]:
            nested_ranges.append(last_ranges[i])
    sort_key = lambda range : range.start_idx
    nested_ranges.sort(key=sort_key)
    #print(nested_ranges)
    return nested_ranges


def get_my_hinges_with_correlation_simplest(ubi, header, raptor_matrix):
    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    correlations = [0] * m
    for i in range(8, m-8):
        curr_correlations = k_inv[i,i-8:i+8]
        correlations[i] = np.inner(curr_correlations, [1,2,4,8,4,2,1,0,0,1,2,4,8,4,2,1])
        correlations[i] = 1 / np.sum(curr_correlations)
    for j, score in enumerate(correlations):
        print(j, score)
    plt.plot(correlations)
    plt.ylabel('some numbers')
    plt.show()

def get_my_gnm_hinges_with_correlation_distance(ubi, header, raptor_matrix):
    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    correlations = [0] * m
    for i in range(5, m-5):
        pp_correlations = k_inv[i-5,:]
        prev_correlations = k_inv[i-5,:]
        curr_correlations = k_inv[i,:]
        next_correlations = k_inv[i+4,:]
        nn_correlations = k_inv[i+5,:]
        correlations[i] = np.linalg.norm(prev_correlations - next_correlations) + np.linalg.norm(nn_correlations - pp_correlations) - \
                          np.linalg.norm(prev_correlations - pp_correlations) - np.linalg.norm(next_correlations - nn_correlations)

    for j, score in enumerate(correlations):
        print(j, score)
    plt.plot(correlations)
    plt.ylabel('some numbers')
    plt.show()



def get_my_gnm_hinges_with_rigidity_normalization(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    maxima_count = [0] * m
    scores_sum = [0] * m

    for current_sensitivity in range(5, 30, 2):
        sensitivity_scores = [0] * m
        for i in range(m):

            # Calc hinge score parts
            cross_correlation_checked = 0
            cross_correlation_sum = 0

            auto_rigid_checked = 0
            auto_rigid_sum = 0

            sub_matrix_size = min(min(i ,current_sensitivity), min(m-i-1, current_sensitivity))

            normalized_sub_matrix = k_inv[i - sub_matrix_size:i + sub_matrix_size + 1, i - sub_matrix_size:i + sub_matrix_size + 1]
      #      normalized_sub_matrix = normalize_matrix(normalized_sub_matrix)

            (r1,r2) = normalized_sub_matrix.shape

            pivot = int((r1 - 1) / 2)

            left_up = normalized_sub_matrix[0:pivot,0:pivot].copy()
            right_bottom = normalized_sub_matrix[pivot+1:r1,pivot+1:r1].copy()
            cross = normalized_sub_matrix[0:pivot,pivot+1:r1].copy()

            np.fill_diagonal(left_up, 0)
            np.fill_diagonal(right_bottom, 0)

            left_up_sum = np.average(left_up)
            right_bottom_sum = np.average(right_bottom)
            cross_sum = np.average(cross)

            score = left_up_sum + right_bottom_sum - 2 * cross_sum
           # score = 1 if (cross_sum < 0 and (left_up_sum > 0 or right_bottom_sum > 0)) else 0
            sensitivity_scores[i] = score
            scores_sum[i] += score

        local_maximas = find_local_maxima(sensitivity_scores)
        print(local_maximas)
        for local_maxima in local_maximas:
            for j in range(int(local_maxima - current_sensitivity / 5), int(local_maxima + current_sensitivity / 5)):
                if j >= 0 and j < m:
                    maxima_count[j] += 1
        # for i in range(m):
        #     residues_to_score_parts[i] += sensitivity_scores[i]

    alpha = 0.5
    combined_scores = (alpha * np.array(scores_sum) + (1-alpha) * np.array(maxima_count)).tolist()
    for j, score in enumerate(combined_scores):
        print(j, score)
    plt.plot(combined_scores)
    plt.ylabel('some numbers')
    plt.show()



def get_hinges_using_cross_correlation_avgs(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    prediction_scores = [0] * m

    for i in range(m):

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

    for i in range(local_sensitivity, m - local_sensitivity):

        sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, local_sensitivity)

        left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

        cross_avg = np.average(cross)

        prediction_scores[i] = -cross_avg

    return predict_hinges(prediction_scores, max_mean_alpha=0.6, total_max_beta=0.5)



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


# def get_my_gnm_hinges(ubi, header, raptor_matrix):
#
#     k_inv = calc_gnm_k_inv(ubi, header)
#     (n,m) = k_inv.shape
#
#     negative_correlation_sensitivity = 5
#     hinge_ranges = None
#     min_percentage = 0.9
#     max_percentage = 0.98
#     percentage_delta = 0.01
#     current_percentage = min_percentage
#
#     residue_to_score = {}
#     # calculate score
#     alpha = 0
#     while alpha <= 1:
#
#         for i in range(m):
#             negative_correlation_count = 0
#             total_checked = 0
#             correlation_sum = 0
#             for d_minus in range(negative_correlation_sensitivity):
#                 for d_plus in range(negative_correlation_sensitivity):
#                     if i > d_minus and i + d_plus + 1 < m:
#                         total_checked += 1
#                         correlation = k_inv[i - d_minus - 1, i + d_plus + 1]
#                         correlation_sum += correlation
#                         if correlation < 0:
#                             negative_correlation_count += 1
#
#             if total_checked > 0:
#                 checked)
#                 residue_to_score[i] = score
#                 # if total_checked > 0:
#                     # if correlation_sum / float(total_checked) < 0.01 and negative_correlation_count / float(total_checked) > current_percentage:
#                       #  print(i)
#         scores = np.array(list(residue_to_score.values()))
#         a = np.r_[True, scores[1:] > scores[:-1]] & np.r_[scores[:-1] > scores[1:], True]
#         print([i for i, x in enumerate(a) if x])
#         print(dict(sorted(residue_to_score.items(), key=operator.itemgetter(1), reverse=True)[:20]))
#         print(residue_to_score[12])
#         alpha += 0.1
#
#     while current_percentage <= max_percentage:
#         current_ranges = []
#         for i in range(m):
#             negative_correlation_count = 0
#             total_checked = 0
#             correlation_sum = 0
#             for d_minus in range(negative_correlation_sensitivity):
#                 for d_plus in range(negative_correlation_sensitivity):
#                     if i > d_minus and i + d_plus + 1 < m:
#                         total_checked += 1
#                         correlation = k_inv[i - d_minus - 1, i + d_plus + 1]
#                         correlation_sum += correlation
#                         if correlation < 0.01:
#                             negative_correlation_count += 1
#
#             if total_checked > 0:
#                 alpha = 1
#                 score = alpha * (- correlation_sum / float(total_checked)) + (1-alpha) * negative_correlation_count / float(total_checked)
#                 if score > 0.5:
#                     insert_to_ranges(current_ranges, i)
#             # if total_checked > 0:
#                 # if correlation_sum / float(total_checked) < 0.01 and negative_correlation_count / float(total_checked) > current_percentage:
#                   #  print(i)
#
#         if hinge_ranges is None:
#             hinge_ranges = current_ranges
#         else:
#             hinge_ranges = create_nested_ranges(hinge_ranges, current_ranges)
#         current_percentage += percentage_delta
#     return hinge_ranges

# def main_old():
# #    ubi, header = parsePDB('1ggg', chain='A', subset='calpha', header=True)
#
#  #   pdb_path = '/tmp/77117-31304/ff0.pdb'
#     pdb_path = '/tmp/663817-9353/ff0.pdb'
#
#     raptor_path = '/tmp/373539.rr'
#     ubi, header = parsePDB(pdb_path, subset='calpha', header=True)
#     get_my_gnm_hinges(ubi, header, parse_raptor_file(len(ubi), raptor_path))
#     print(ubi._n_atoms)
#     mode = 0
#     print("DEFAULT:", get_hinges_default(ubi, header, cutoff=10, modeIndex = mode))
#     # print("STRUCTURE:", get_hinges_using_structure(ubi, header, modeIndex = mode))
#    # print("DISTANCE:", get_hinges_using_distance(ubi, header, modeIndex = mode))
#     print("RAPTOR:" , get_hinges_using_raptor(ubi, header, parse_raptor_file(len(ubi), raptor_path), modeIndex = mode))


# def print_morph_hinges_results(atlas_morphs, atlas_directory, morph_id, raptor_file=None):
#     filtered_morphs = list(filter(lambda atlas_morph: atlas_morph.morph_id == morph_id, atlas_morphs))
#     if len(filtered_morphs) == 0:
#         print('Could not find morph %s in atlas annotations' % morph_id)
#         return
#     if len(filtered_morphs) > 1:
#         print('That is weird!')
#         return
#     morph = filtered_morphs[0]
#     ff0_path = '%s/%s/ff0.pdb' % (atlas_directory, morph_id)
#     if Path(ff0_path).is_file():
#         with open(ff0_path, 'r') as pdb_file:
#             ubi, header = parsePDB(ff0_path, subset='calpha', header=True)
#             print("Morph: ", morph_id)
#             #print("My Hinges: ", get_my_gnm_hinges(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
#             print("Annotated Hinges: ", morph.get_hinges())
#             # print("ANM with rigid: " , get_my_anm_hinges_with_rigidity(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
#             print("GNM with rigid: " , get_my_hinges_with_correlation_simplest(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
#             print("GNM with rigid: " , get_my_gnm_hinges_with_rigidity_normalization(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
#             #print("ANM: ", get_my_anm_hinges(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
#             print("Default Hinges: ", get_hinges_default(ubi, header))
#


# def print_all_morphs_results():
#     atlas_morphs = parse_morphs_atlas_from_text('./hingeatlas.txt')
#     atlas_directory = '/Users/mataneilat/Downloads/hinge_atlas_nonredundant'
#
#     directory = os.fsencode(atlas_directory)
#
#     for file in os.listdir(directory):
#         morph_filename = os.fsdecode(file)
#         print_morph_hinges_results(atlas_morphs, atlas_directory, morph_filename)


def main():
    morphs_ids = list(morphs_repository.atlas_morphs.keys())

    test_morph_ids = morphs_ids[150:]

    total_my_score_1 = 0
    total_my_score_2 = 0
    total_my_score_3 = 0
    total_default_score = 0

    def print_prediction_results(morph, ubi, header):
        nonlocal total_my_score_1, total_my_score_2, total_my_score_3, total_default_score

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

        default_hinges = get_hinges_default(ubi, header)
        default_score = score_prediction(default_hinges, morph.get_hinges(), total_residue_count)
        total_default_score += default_score

        print("Annotated Hinges: ", morph.get_hinges())
        print("DEFAULT HINGES: ", default_hinges, default_score)
        print("MY HINGES 1:", predicted_hinges_1, my_score_1)
        print("MY HINGES 2:", predicted_hinges_2, my_score_2)
        print("MY HINGES 3:", predicted_hinges_3, my_score_3)

    morphs_repository.perform_on_some_morphs_in_directory(lambda morph_id : morph_id in test_morph_ids,
            print_prediction_results)

    print("MY SCORE 1 IS: ", total_my_score_1)
    print("MY SCORE 2 IS: ", total_my_score_2)
    print("MY SCORE 3 IS: ", total_my_score_3)
    print("DEFAULT SCORE IS: ", total_default_score)

if __name__ == "__main__":
    main()
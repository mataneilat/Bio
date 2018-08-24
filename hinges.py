
from prody import *
import numpy as np
from bio.cast import *
from bio.hinge_atlas_parser import *
import os
from pathlib import Path
import operator
import matplotlib.pyplot as plt


# def calc_anm_hinges(anm):
#     if anm._array is None:
#         raise ValueError('Modes are not calculated.')
#     # obtain the eigenvectors
#     V = anm._array
#     print(V.shape)
#     (m, n) = V.shape
#     hinges = []
#     for i in range(n):
#         v = V[:,i]
#         # obtain the signs of eigenvector
#         s = np.sign(v)
#         # obtain the relative magnitude of eigenvector
#         mag = np.sign(np.diff(np.abs(v)))
#         # obtain the cross-overs
#         torf = np.diff(s)!=0
#         indices = np.where(torf)[0]
#         print("indices: %s" % indices)
#         # find which side is more close to zero
#         for i in range(len(indices)):
#             idx = indices[i]
#             if mag[idx] < 0:
#                 indices[i] += 1
#         hinges.append(indices)
#     return np.array(hinges)
#
# def get_anm_hinges_using_dist(ubi, header):
#     anm = ANM('Ubiquitin')
#     anm.buildHessian(ubi)
#     anm.calcModes()
#     print('ANM: %s' % calc_anm_hinges(anm))


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


class Range:

    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __repr__(self):
        return "<%s:%s>" % (self.start_idx, self.end_idx)

    def is_within(self, other_range):
        return self.start_idx >= other_range.start_idx and self.end_idx <= other_range.end_idx

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



def calc_gnm_k_inv(ubi, header, raptor_matrix):
    gnm = GNM('Ubiquitin')
    if raptor_matrix is None:
        gnm.buildKirchhoff(ubi, gamma=GammaDistance())
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


def calc_anm_k_inv(ubi, header, raptor_matrix):
    anm = ANM('Ubiquitin')
    if raptor_matrix is None:
        anm.buildHessian(ubi, gamma=GammaDistance())
    else:
        anm.buildHessian(ubi, gamma=GammaRaptor(raptor_matrix))

    anm.calcModes()
    V = anm._array
    eigvals = anm._eigvals
    (m, n) = V.shape

    k_inv = np.zeros((m,m))
    for i in range(2):
        eigenvalue = eigvals[i]
        #print("EV: " , eigenvalue)
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)
    return k_inv


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



def get_my_gnm_hinges_with_rigidity(ubi, header, raptor_matrix):

    k_inv = calc_gnm_k_inv(ubi, header, raptor_matrix)

    (m,n) = k_inv.shape

    residues_segment_length = 5

    residues_to_score_parts = {}

    for i in range(m):

        # Calc hinge score parts
        cross_correlation_checked = 0
        cross_correlation_sum = 0

        auto_rigid_checked = 0
        auto_rigid_sum = 0

        for d1 in range(1, residues_segment_length):
            for d2 in range(1, residues_segment_length):
                if i - d1 >= 0 and i + d2 < m:
                    cross_correlation_checked += 1
                   # cross_correlation_sum += (d1 + d2) * k_inv[i - d1, i + d2]
                    cor = k_inv[i - d1, i + d2]
                    if cor < 0.1:
                        cross_correlation_sum += (d1 + d2) * cor

        for d1 in range(1, residues_segment_length):
            for k in range(1, residues_segment_length - d1):
                d2 = d1 + k
                if i - d1 >= 0 and i - d2 >= 0:
                    auto_rigid_checked += 1
                    cor = k_inv[i - d1, i - d2]
                    if cor > 0.5:
                        auto_rigid_sum += cor

                if i + d1 < m and i + d2 < m:
                    # auto_rigid_checked += 1
                    # auto_rigid_sum += k_inv[i + d1, i + d2]
                    auto_rigid_checked += 1
                    cor = k_inv[i + d1, i + d2]
                    if cor > 0.5:
                        auto_rigid_sum += cor

        cross_correlation_avg = None
        if cross_correlation_checked > 0:
            cross_correlation_avg = float(cross_correlation_sum) / cross_correlation_checked
        auto_rigid_avg = None
        if auto_rigid_checked > 0:
            auto_rigid_avg = float(auto_rigid_sum) / auto_rigid_checked

        residues_to_score_parts[i] = (cross_correlation_avg, auto_rigid_avg)

    for residue_to_score_parts in residues_to_score_parts.items():
        print(residue_to_score_parts)


def get_my_anm_hinges_with_rigidity(ubi, header, raptor_matrix):

    k_inv = calc_anm_k_inv(ubi, header, raptor_matrix)
    (m,n) = k_inv.shape
    residues_segment_length = 10

    correlations_dict = {}

    for i in range(0, m, 3):
        xyz_dict = {}
        for j in range(3):

            curr_idx = i + j

            cross_correlation_checked = 0
            cross_correlation_sum = 0

            auto_rigid_checked = 0
            auto_rigid_sum = 0

            for d1 in range(residues_segment_length):
                for d2 in range(residues_segment_length):
                    d_1 = 3 * (d1 + 1)
                    d_2 = 3 * (d2 + 1)

                    if curr_idx - d_1 >= 0 and curr_idx + d_2 < m:
                        cross_correlation_checked += 1
                        cross_correlation_sum += k_inv[curr_idx - d_1, curr_idx + d_2]

                    if curr_idx - d_1 >= 0 and curr_idx - d_2 >= 0 and d_1 != d_2:
                        auto_rigid_checked += 1
                        auto_rigid_sum += k_inv[curr_idx - d_1, curr_idx - d_2]

                    if curr_idx + d_1 < m and curr_idx + d_2 < m and d_1 != d_2:
                        auto_rigid_checked += 1
                        auto_rigid_sum += k_inv[curr_idx + d_1, curr_idx + d_2]

            cross_correlation_avg = None
            if cross_correlation_checked > 0:
                cross_correlation_avg = float(cross_correlation_sum) / cross_correlation_checked
            auto_rigid_avg = None
            if auto_rigid_checked > 0:
                auto_rigid_avg = float(auto_rigid_sum) / auto_rigid_checked
            xyz_dict[j] = (cross_correlation_avg, auto_rigid_avg)

        # correlations_dict[i / 3.0] = xyz_dict

        # calculate score
        score = 0
        for j in range(3):
            auto_rigid_avg = xyz_dict[j][1]
            cross_correlation_avg = xyz_dict[j][0]
            if auto_rigid_avg is None or cross_correlation_avg is None:
                score = None
                break
            s = auto_rigid_avg - cross_correlation_avg
            score += (s * s * s)
        correlations_dict[i / 3.0] = score
        if score is not None:
            correlations_dict[i / 3.0] = score * 1000

    for correlation in correlations_dict.items():
        print(correlation)



def get_my_anm_hinges(ubi, header, raptor_matrix):

    k_inv = calc_anm_k_inv(ubi, header, raptor_matrix)
    (m,n) = k_inv.shape

    negative_correlation_sensitivity = 10
    correlations_dict = {}

    for i in range(0, m, 3):
        xyz_dict = {}
        for j in range(3):
            negative_correlation_count = 0
            total_checked = 0
            correlation_sum = 0
            curr_idx = i + j

            # Check for x,y,z separately
            val = []
            for d_minus in range(negative_correlation_sensitivity):
                for d_plus in range(negative_correlation_sensitivity):
                    if curr_idx - 3 * (d_minus + 1) >= 0 and curr_idx + 3 * (d_plus + 1) < m:
                        total_checked += 1
                        correlation = k_inv[curr_idx - 3 * (d_minus + 1), curr_idx + 3 * (d_plus + 1)]
                       # val.append(correlation)
                        correlation_sum += correlation

                        if correlation < 0:
                            negative_correlation_count += (2 * negative_correlation_sensitivity - (d_minus + d_plus))
           # xyz_dict[j] = val
            if total_checked > 0:
                xyz_dict[j] = negative_correlation_count / float(total_checked)
        correlations_dict[i / 3.0] = (xyz_dict, sum(s * s for s in xyz_dict.values()))
    for correlation in correlations_dict.items():
        print(correlation)



def get_my_gnm_hinges(ubi, header, raptor_matrix):

    gnm = GNM('Ubiquitin')
    if raptor_matrix is None:
        gnm.buildKirchhoff(ubi, gamma=GammaDistance())
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
        #print("EV: " , eigenvalue)
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)

    negative_correlation_sensitivity = 5
    hinge_ranges = None
    min_percentage = 0.9
    max_percentage = 0.98
    percentage_delta = 0.01
    current_percentage = min_percentage

    residue_to_score = {}
    # calculate score
    alpha = 0
    while alpha <= 1:

        for i in range(m):
            negative_correlation_count = 0
            total_checked = 0
            correlation_sum = 0
            for d_minus in range(negative_correlation_sensitivity):
                for d_plus in range(negative_correlation_sensitivity):
                    if i > d_minus and i + d_plus + 1 < m:
                        total_checked += 1
                        correlation = k_inv[i - d_minus - 1, i + d_plus + 1]
                        correlation_sum += correlation
                        if correlation < 0:
                            negative_correlation_count += 1

            if total_checked > 0:
                score = alpha * (- correlation_sum / float(total_checked)) + (1-alpha) * negative_correlation_count / float(total_checked)
                residue_to_score[i] = score
                # if total_checked > 0:
                    # if correlation_sum / float(total_checked) < 0.01 and negative_correlation_count / float(total_checked) > current_percentage:
                      #  print(i)
        scores = np.array(list(residue_to_score.values()))
        a = np.r_[True, scores[1:] > scores[:-1]] & np.r_[scores[:-1] > scores[1:], True]
        print([i for i, x in enumerate(a) if x])
        print(dict(sorted(residue_to_score.items(), key=operator.itemgetter(1), reverse=True)[:20]))
        print(residue_to_score[12])
        alpha += 0.1

    while current_percentage <= max_percentage:
        current_ranges = []
        for i in range(m):
            negative_correlation_count = 0
            total_checked = 0
            correlation_sum = 0
            for d_minus in range(negative_correlation_sensitivity):
                for d_plus in range(negative_correlation_sensitivity):
                    if i > d_minus and i + d_plus + 1 < m:
                        total_checked += 1
                        correlation = k_inv[i - d_minus - 1, i + d_plus + 1]
                        correlation_sum += correlation
                        if correlation < 0.01:
                            negative_correlation_count += 1

            if total_checked > 0:
                alpha = 1
                score = alpha * (- correlation_sum / float(total_checked)) + (1-alpha) * negative_correlation_count / float(total_checked)
                if score > 0.5:
                    insert_to_ranges(current_ranges, i)
            # if total_checked > 0:
                # if correlation_sum / float(total_checked) < 0.01 and negative_correlation_count / float(total_checked) > current_percentage:
                  #  print(i)

        if hinge_ranges is None:
            hinge_ranges = current_ranges
        else:
            hinge_ranges = create_nested_ranges(hinge_ranges, current_ranges)
        current_percentage += percentage_delta
    return hinge_ranges

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



def get_hinges_using_structure(ubi, header, modeIndex=0):
    assignSecstr(header, ubi)
    gnm = GNM('Ubiquitin')
    gnm.buildKirchhoff(ubi, gamma=GammaStructureBased(ubi))
    gnm.calcModes()
    return gnm.getHinges(modeIndex)


def get_hinges_default(ubi, header, cutoff=10, modeIndex=0):
    gnm = GNM('Ubiquitin')
    gnm.buildKirchhoff(ubi, cutoff=cutoff)
    gnm.calcModes()
    return gnm.getHinges(modeIndex)

def get_hinges_using_distance(ubi, header, modeIndex=0):
    gnm = GNM('Ubiquitin')
    gnm.buildKirchhoff(ubi, gamma=GammaDistance(ubi))
    gnm.calcModes()
    return gnm.getHinges(modeIndex)


def get_hinges_using_raptor(ubi, header, raptor_matrix, modeIndex=0):
    gnm = GNM('Ubiquitin')
    gnm.buildKirchhoff(ubi, gamma=GammaRaptor(raptor_matrix), kd_tree=False)
    gnm.calcModes()
    return gnm.getHinges(modeIndex)


def parse_raptor_file(atoms_count, rr_file):
    if rr_file is None:
        return None
    mat = np.zeros(shape=(atoms_count, atoms_count))
    for i in range(atoms_count):
        mat[i,i] = 1
    with open(rr_file) as handle:
        content = handle.readlines()
        for line in content:
            tokens = line.split(' ')
            if len(tokens) == 5:
                i = int(tokens[0]) - 1
                j = int(tokens[1]) - 1
                p = float(tokens[4].strip('\n'))
                mat[i,j] = p
                mat[j,i] = p
    return mat


def main_old():
#    ubi, header = parsePDB('1ggg', chain='A', subset='calpha', header=True)

 #   pdb_path = '/tmp/77117-31304/ff0.pdb'
    pdb_path = '/tmp/663817-9353/ff0.pdb'

    raptor_path = '/tmp/373539.rr'
    ubi, header = parsePDB(pdb_path, subset='calpha', header=True)
    get_my_gnm_hinges(ubi, header, parse_raptor_file(len(ubi), raptor_path))
    print(ubi._n_atoms)
    mode = 0
    print("DEFAULT:", get_hinges_default(ubi, header, cutoff=10, modeIndex = mode))
    # print("STRUCTURE:", get_hinges_using_structure(ubi, header, modeIndex = mode))
   # print("DISTANCE:", get_hinges_using_distance(ubi, header, modeIndex = mode))
    print("RAPTOR:" , get_hinges_using_raptor(ubi, header, parse_raptor_file(len(ubi), raptor_path), modeIndex = mode))


def print_morph_hinges_results(atlas_morphs, atlas_directory, morph_id, raptor_file=None):
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
            #print("My Hinges: ", get_my_gnm_hinges(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
            print("Annotated Hinges: ", morph.get_hinges())
            # print("ANM with rigid: " , get_my_anm_hinges_with_rigidity(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
            print("GNM with rigid: " , get_my_hinges_with_correlation_simplest(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
            print("GNM with rigid: " , get_my_gnm_hinges_with_rigidity_normalization(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
            #print("ANM: ", get_my_anm_hinges(ubi, header, raptor_matrix=parse_raptor_file(len(ubi), raptor_file)))
            print("Default Hinges: ", get_hinges_default(ubi, header))



def print_all_morphs_results():
    atlas_morphs = parse_hinge_atlas_text('./hingeatlas.txt')
    atlas_directory = '/Users/mataneilat/Downloads/hinge_atlas_nonredundant'

    directory = os.fsencode(atlas_directory)

    for file in os.listdir(directory):
        morph_filename = os.fsdecode(file)
        print_morph_hinges_results(atlas_morphs, atlas_directory, morph_filename)


if __name__ == "__main__":
    # np.set_printoptions(precision=2, suppress=True)
    # atlas_morphs = parse_hinge_atlas_text('./hingeatlas.txt')
    # atlas_directory = '/Users/mataneilat/Downloads/hinge_atlas_nonredundant'
    # print_morph_hinges_results(atlas_morphs, atlas_directory, '021200-13968',
    #                            raptor_file=None)
    print_all_morphs_results()
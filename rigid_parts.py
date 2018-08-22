
from prody import *
import numpy as np
from bio.cast import *


def my_get_hinges(gnm, modeIndex=None, flag=False):
    calc_hinges = my_calc_hinges(gnm)
    if calc_hinges is None:
        LOGGER.info('Warning: hinges are not calculated, thus None is returned. '
                    'Please call GNM.calcHinges() to calculate the hinge sites first.')
        return None
    if modeIndex is None:
        hinges = calc_hinges
    else:
        hinges = calc_hinges[:, modeIndex]

    if flag:
        return hinges
    else:
        hinge_list = np.where(hinges)[0]
        return sorted(set(hinge_list))

def my_calc_hinges(gnm):
    if gnm._array is None:
        raise ValueError('Modes are not calculated.')
    # obtain the eigenvectors
    V = gnm._array
    (m, n) = V.shape
    hinges = []
    for i in range(n):
        v = V[:, 1]
        # obtain the signs of eigenvector
        s = np.sign(v)
        # obtain the relative magnitude of eigenvector
        mag = np.sign(np.diff(np.abs(v)))
        # obtain the cross-overs
        torf = np.diff(s)!=0
        torf = np.append(torf, [False], axis=0)
        hinges.append(torf)
        # print(torf)
        # hing = np.array(torf, copy=True)
        # # find which side is more close to zero
        # for j, m in enumerate(mag):
        #     if torf[j] and m < 0:
        #         torf[j+1] = True
        #         torf[j] = False

        hinges.append(torf)

    return np.stack(hinges).T


# def build_rigid_parts(atoms_count, hinges):
#     start = 0
#     rigid_parts = []
#     for hinge in hinges:
#         rigid_parts.append(RigidPart(start, hinge))
#         start = hinge + 1
#
#     rigid_parts.append(RigidPart(start, atoms_count-1))
#
#     print("RIGG:", rigid_parts)
#     big_rigid_parts = [rigid_parts[0]]
#     for rigid_part in rigid_parts[:-1]:
#         if len(rigid_part) >= 15:
#             big_rigid_parts.append(rigid_part)
#     big_rigid_parts.append(rigid_parts[-1])
#
#     print("BIGG:", big_rigid_parts)
#     final_rigid_parts = []
#     start_idx = 0
#     end_idx = big_rigid_parts[0].end_idx
#     for big_rigid_part in big_rigid_parts[1:]:
#         if big_rigid_part.start_idx == end_idx + 1:
#             final_rigid_parts.append(RigidPart(start_idx, end_idx))
#             start_idx = big_rigid_part.start_idx
#             end_idx = big_rigid_part.end_idx
#         else:
#             end_idx = big_rigid_part.end_idx
#
#     final_rigid_parts.append(RigidPart(start_idx, atoms_count-1))
#
#     return final_rigid_parts

def calc_dist(a, b):
    return np.linalg.norm(a-b)

def build_similarity_matrix(ubi, rigid_parts, correlation_signs):
    rigid_parts_count = len(rigid_parts)
    similarity_matrix = np.zeros((rigid_parts_count, rigid_parts_count))
    for i in range(rigid_parts_count):
        for j in range(rigid_parts_count):
            if i == j:
                similarity_matrix[i,j] = 1
                continue
            if correlation_signs[i] != correlation_signs[j]:
                continue
            i_start = rigid_parts[i].start_idx
            i_end = rigid_parts[i].end_idx
            j_start = rigid_parts[j].start_idx
            j_end = rigid_parts[j].end_idx
            dist_1 = calc_dist(ubi[i_start].getCoords(), ubi[j_start].getCoords())
            dist_2 = calc_dist(ubi[i_end].getCoords(), ubi[j_end].getCoords())
            if dist_1 < 16 and dist_2 < 16:
                similarity_matrix[i,j] = 1
                continue
            dist_1 = calc_dist(ubi[i_start].getCoords(), ubi[j_end].getCoords())
            dist_2 = calc_dist(ubi[i_end].getCoords(), ubi[j_start].getCoords())
            if dist_1 < 25 and dist_2 < 25:
                similarity_matrix[i,j] = 1


    return similarity_matrix


def get_correlations_signs(rigid_parts, gnm, mode=0):
    V = gnm._array
    (m, n) = V.shape
    print(m)
    v = V[:,mode]
    sign_vector = np.sign(v)
    correlation_signs = []
    for rigid_part in rigid_parts:
        positive_count = 0
        for s in sign_vector[rigid_part.start_idx:rigid_part.end_idx]:
            if s > 0:
                positive_count += 1
        print(rigid_part, positive_count)
        if positive_count * 2 > len(rigid_part):
            correlation_signs.append(1)
        else:
            correlation_signs.append(-1)
    return correlation_signs


def build_rigid_parts(atoms_count, hinges):
    print(atoms_count)
    rigid_start = 0
    rigid_parts = []
    for hinge in hinges:
        rigid_end = hinge
        if rigid_end - rigid_start >= 20:
            rigid_parts.append(RigidPart(rigid_start, rigid_end))
            rigid_start = rigid_end + 1
    if atoms_count == rigid_start:
        return rigid_parts
    # if atoms_count - rigid_start < 15:
    #     rigid_parts[-1].end_idx = atoms_count - 1
    #     return rigid_parts
    rigid_parts.append(RigidPart(rigid_start, atoms_count - 1))
    return rigid_parts

def main():
    ubi, header = parsePDB('1aar', chain='A', subset='calpha', header=True)

    gnm = GNM('Ubiquitin')

    print("NOW")
    mode = 1

    gnm.buildKirchhoff(ubi)
    gnm.calcModes()

    hinges = my_get_hinges(gnm, modeIndex=mode)
    print(hinges)
    rigid_parts = build_rigid_parts(ubi._n_atoms, hinges)
    print(rigid_parts)

    # rigid_parts = [RigidPart(0,7),RigidPart(8, 37), RigidPart(38,151), RigidPart(152, 172),
    #               RigidPart(173, 192), RigidPart(193, 219)]

    correlation_signs = get_correlations_signs(rigid_parts, gnm, mode=mode)
    print(correlation_signs)
    similiarity_matrix = build_similarity_matrix(ubi, rigid_parts, correlation_signs)
    print(similiarity_matrix)
    clusters = CAST(similiarity_matrix, 0.5)
    rigid_part_clusters = []
    for c in clusters:
        rigid_part_clusters.append([rigid_parts[x] for x in c])
    print(rigid_part_clusters)

   # get_anm_hinges_using_dist(ubi, header)

  #  get_hinges_using_structure(ubi, header)
  #  get_hinges_using_distance(ubi, header)
  #  get_hinges_using_raptor(ubi, header, parse_raptor_file(len(ubi), '/tmp/372846.rr'))



if __name__ == "__main__":
    main()

class RigidPart:

    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __repr__(self):
        return "<%s:%s>" % (self.start_idx, self.end_idx)

    def __len__(self):
        return self.end_idx - self.start_idx + 1


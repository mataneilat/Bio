
from prody import *
import numpy as np
from bio.cast import *

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
    print(nested_ranges)
    return nested_ranges



def print_correlation(ubi, header, raptor_matrix):

    # gnm = GNM('Ubiquitin')
    # gnm.buildKirchhoff(ubi)
    # gnm.calcModes()

    gnm = GNM('Ubiquitin')
    gnm.buildKirchhoff(ubi, gamma=GammaRaptor(raptor_matrix))
    gnm.calcModes()

    if gnm._array is None:
        raise ValueError('Modes are not calculated.')
    # obtain the eigenvectors
    V = gnm._array
    eigvals = gnm._eigvals
    (m, n) = V.shape

    k_inv = np.zeros((m,m))
    for i in range(3):
        eigenvalue = eigvals[i]
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)

    print(m)
    negative_correlation_sensitivity = 20
    hinge_ranges = None
    min_percentage = 0.9
    max_percentage = 0.98
    percentage_delta = 0.01
    current_percentage = min_percentage
    while current_percentage <= max_percentage:
        current_ranges = []
        for i in range(m):
            negative_correlation_count = 0
            total_checked = 0
            for d_minus in range(negative_correlation_sensitivity):
                for d_plus in range(negative_correlation_sensitivity):
                    if i > d_minus and i + d_plus + 1 < m:
                        total_checked += 1
                        if k_inv[i - d_minus - 1, i + d_plus + 1] < 0:
                            negative_correlation_count += 1

            if total_checked > 0 and negative_correlation_count / float(total_checked) > current_percentage:
                insert_to_ranges(current_ranges, i)

        if hinge_ranges is None:
            hinge_ranges = current_ranges
        else:
            hinge_ranges = create_nested_ranges(hinge_ranges, current_ranges)
        current_percentage += percentage_delta
    print("MINE:", hinge_ranges)

class GammaRaptor(Gamma):

    def __init__(self, raptor_matrix):
        self.raptor_matrix = raptor_matrix

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        raptor_score = self.raptor_matrix[i, j]
        alpha = 0.5
        return alpha * raptor_score + (1 - alpha) * (1 - dist2 / 100.0)


class GammaDistance(Gamma):

    def __init__(self, identifiers, gamma=1., default_radius=7.5, **kwargs):
        pass

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        return 1 - dist2 / 100.0



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


def main():
#    ubi, header = parsePDB('1ggg', chain='A', subset='calpha', header=True)

 #   pdb_path = '/tmp/77117-31304/ff0.pdb'
    pdb_path = '/tmp/663817-9353/ff0.pdb'

    raptor_path = '/tmp/373539.rr'
    ubi, header = parsePDB(pdb_path, subset='calpha', header=True)
    print_correlation(ubi, header, parse_raptor_file(len(ubi), raptor_path))
    print(ubi._n_atoms)
    mode = 0
    print("DEFAULT:", get_hinges_default(ubi, header, cutoff=10, modeIndex = mode))
    # print("STRUCTURE:", get_hinges_using_structure(ubi, header, modeIndex = mode))
   # print("DISTANCE:", get_hinges_using_distance(ubi, header, modeIndex = mode))
    print("RAPTOR:" , get_hinges_using_raptor(ubi, header, parse_raptor_file(len(ubi), raptor_path), modeIndex = mode))



if __name__ == "__main__":
    main()
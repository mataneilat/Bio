import numpy as np
from prody import GNM

from nma.gamma_functions import *
from benchmark import Benchmark
import time

def get_hinges_default(ubi, header, cutoff=8, modeIndex=0):
    gnm = GNM()
    gnm.buildKirchhoff(ubi, cutoff=cutoff)
    gnm.calcModes()
    return gnm.getHinges(modeIndex)


def calc_gnm_k_inv(ubi, header, contact_map=None, cutoff=8, raptor_alpha=0.5, number_of_modes=2):

    benchmark = Benchmark()
    gnm = GNM()

    before_kirchhoff = time.time()
    if contact_map is None:
        gnm.buildKirchhoff(ubi, cutoff=cutoff, gamma=SquaredDistanceGamma(cutoff))
    else:
        gnm.buildKirchhoff(ubi, cutoff=cutoff, gamma=ContactMapAndDistanceGamma(contact_map, cutoff, raptor_alpha))
    after_kirchhoff = time.time()

    benchmark.update(len(ubi), 'Springs Setup', after_kirchhoff - before_kirchhoff)

    before_calc_modes = time.time()
    gnm.calcModes()
    after_calc_modes = time.time()

    benchmark.update(len(ubi), 'Eigenvalues Find', after_calc_modes - before_calc_modes)

    if gnm._array is None:
        raise ValueError('Modes are not calculated.')

    V = gnm._array

    eigvals = gnm._eigvals

    (m, n) = V.shape
    k_inv = np.zeros((m,m))

    before_k_inv = time.time()
    for i in range(number_of_modes):
        eigenvalue = eigvals[i]
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)
    after_k_inv = time.time()

    benchmark.update(len(ubi), 'Gamma Inversion', after_k_inv - before_k_inv)

    return k_inv

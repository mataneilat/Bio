import numpy as np
from prody import GNM

from bio.nma.gamma_functions import *


def get_hinges_default(ubi, header, cutoff=8, modeIndex=0):
    gnm = GNM()
    gnm.buildKirchhoff(ubi, cutoff=cutoff)
    gnm.calcModes()
    return gnm.getHinges(modeIndex)


def calc_gnm_k_inv(ubi, header, contact_map=None, cutoff=8, raptor_alpha=0.7, number_of_modes=2):

    gnm = GNM()

    if contact_map is None:
        gnm.buildKirchhoff(ubi, cutoff=cutoff, gamma=SquaredDistanceGamma(cutoff))
    else:
        gnm.buildKirchhoff(ubi, cutoff=cutoff, gamma=ContactMapAndDistanceGamma(contact_map, cutoff, raptor_alpha))

    gnm.calcModes()

    if gnm._array is None:
        raise ValueError('Modes are not calculated.')

    V = gnm._array

    eigvals = gnm._eigvals

    (m, n) = V.shape
    k_inv = np.zeros((m,m))

    for i in range(number_of_modes):
        eigenvalue = eigvals[i]
        eigenvector = V[:,i]
        k_inv += (np.outer(eigenvector, eigenvector) / eigenvalue)

    return k_inv

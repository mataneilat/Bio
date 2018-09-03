
"""
    This module contains the hinge predictors which do not use machine learning mechanisms.
"""

from prediction.prediction_commons import *
from utils import *
import math
from benchmark import Benchmark
import time
import scipy.spatial as spt

class CorrelationVectorsDistancePredictor(HingePredictor):
    """
    This predictor follows the following reasoning in order to predict a hinge:
    For every residue [i] , this predictor calculates the squared distance between the correlations of preceding and
    succeeding residues with every other residue in the protein.
    In case [i] is a hinge residue, we expect that preceding residues will have high correlation with the residues that
    have low correlation with succeeding residues of [i], and vice versa.
    Hence the squared distance between these correlation vectors should be high.
    """
    def __init__(self, local_sensitivity):
        """
        Constructs the predictor with the local sensitivity factor.
        :param  local_sensitivity: An integer representing the number of preceding and succeeding residues we wish to
                consider
        """
        self.local_sensitivity = local_sensitivity

    def predict_hinges(self, k_inv):
        """
        Predicts the hinge residues.
        :param  k_inv: The inverted matrix produced by the GNM analysis which is interpreted as a correlation matrix.
        :return:    The predicted hinge residues
        """
        (m,n) = k_inv.shape

        confidence_levels = [0] * m

        before_cvd = time.time()
        for i in range(m):
            forward = min(m-i, self.local_sensitivity)
            backward = min(i, self.local_sensitivity - 1)

            distances = spt.distance.squareform(spt.distance.pdist(k_inv[i - backward:i + forward,:],
                                                                   'sqeuclidean'))

            confidence_levels[i] = np.sum(distances[:backward,backward+1:])

        after_cvd = time.time()
        Benchmark().update(m, 'CVD confidence', after_cvd - before_cvd)
        return predict_hinges(confidence_levels, self.local_sensitivity, 90, 95, 0.2)


class NearCorrelationAvgsPredictor(HingePredictor):
    """
    This predictor follows the following reasoning in order to predict a hinge:
    In order for a residue [i] to be a hinge, we expect that preceding residues will be in high correlation, as well as
    succeeding residues. However, we also expect that the cross-correlations between the preceding and succeeding
    residues will be low.
    """
    def __init__(self, local_sensitivity, alpha=0.6):
        """
        Constructs the predictor with the local sensitivity factor.
        :param  local_sensitivity: An integer representing the number of preceding and succeeding residues we wish to
                consider
        :param  alpha: The weigh parameter used for arithmetic average. Defaults to 0.6
        """
        self.local_sensitivity = local_sensitivity
        self.alpha = alpha

    def predict_hinges(self, k_inv):
        """
        Predicts the hinge residues.
        :param  k_inv: The inverted matrix produced by the GNM analysis which is interpreted as a correlation matrix.
        :return:    The predicted hinge residues
        """
        (m,n) = k_inv.shape
        confidence_levels = [0] * m

        before_nca = time.time()
        for i in range(1, m-1):
            forward = min(m-i-1, self.local_sensitivity)
            backward = min(i, self.local_sensitivity)

            cross = k_inv[i+1:i+forward+1, i-backward:i]
            left_up = k_inv[i-backward:i, i-backward:i]
            right_bottom = k_inv[i+1:i+forward+1, i+1:i+forward+1]

            inter_avg = np.average(np.append(upper_triangular_no_diagonal(left_up),
                                             upper_triangular_no_diagonal(right_bottom)))

            cross_avg = np.average(cross)

            confidence_levels[i] = (1 - self.alpha) * inter_avg - self.alpha * cross_avg

        after_nca = time.time()
        Benchmark().update(m, 'NCA confidence', after_nca - before_nca)

        return predict_hinges(confidence_levels, self.local_sensitivity, 80, 0, 0.7)


class CrossCorrelationAvgPredictor(HingePredictor):
    """
    This predictor follows the following reasoning in order to predict a hinge:
    In order for a residue [i] to be a hinge, we expect that the cross-correlations between the preceding and succeeding
    residues will be low.
    """
    def __init__(self, local_senstivity):
        """
        Constructs the predictor with the local sensitivity factor.
        :param  local_sensitivity: An integer representing the number of preceding and succeeding residues we wish to
                    consider
        """
        self.local_sensitivity = local_senstivity

    def predict_hinges(self, k_inv):
        """
        Predicts the hinge residues.
        :param  k_inv: The inverted matrix produced by the GNM analysis which is interpreted as a correlation matrix.
        :return:    The predicted hinge residues
        """
        (n,m) = k_inv.shape

        confidence_levels = [0] * m

        before_cca = time.time()
        for i in range(1, m - 1):

            forward = min(m-i-1, self.local_sensitivity)
            backward = min(i, self.local_sensitivity)

            cross = k_inv[i+1:i+forward+1, i-backward:i]

            cross_avg = np.average(cross)

            confidence_levels[i] = math.exp(-cross_avg)
        after_cca = time.time()
        Benchmark().update(m, 'CCA confidence', after_cca - before_cca)
        return predict_hinges(confidence_levels, self.local_sensitivity, 90, 95, 0.3)
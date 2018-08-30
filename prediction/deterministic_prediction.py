
"""
    This module contains the hinge predictors which do not use machine learning mechanisms.
"""

from prediction.prediction_commons import *
from utils import *
import math
import itertools


class CorrelationDistancePredictor(HingePredictor):
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
        for i in range(m):
            for d, k in itertools.product(range(self.local_sensitivity), range(self.local_sensitivity)):
                if i-k >= 0 and i+d < m:
                    confidence_levels[i] += np.sum(np.square(k_inv[i-k,:] - k_inv[i+d,:]))

        return predict_hinges(confidence_levels, self.local_sensitivity, 90, 95, 0)


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
        prediction_scores = [0] * m

        for i in range(1, m-1):

            sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, self.local_sensitivity)

            left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

            inter_avg = np.average(np.append(upper_triangular_no_diagonal(left_up),
                                             upper_triangular_no_diagonal(right_bottom)))

            cross_avg = np.average(cross)

            prediction_scores[i] = (1 - self.alpha) * inter_avg - self.alpha * cross_avg

        return predict_hinges(prediction_scores, self.local_sensitivity, 80, 0, 0.7)


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

        prediction_scores = [0] * m

        for i in range(1, m - 1):

            sub_matrix = get_maximum_sub_matrix_around_diagonal_element(k_inv, i, self.local_sensitivity)

            left_up, right_bottom, cross = get_block_metrics_around_center(sub_matrix)

            cross_avg = np.average(cross)

            prediction_scores[i] = math.exp(-cross_avg)

        return predict_hinges(prediction_scores, self.local_sensitivity, 90, 95, 0.3)
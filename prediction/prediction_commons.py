"""
    This module contains methods and classes which are commonly used around the prediction package
"""
import numpy as np

import abc

class HingePredictor(metaclass=abc.ABCMeta):
    """
    Interface for hinge predictor
    """
    @abc.abstractmethod
    def predict_hinges(self, k_inv):
        """
        Predicts the hinge residues.
        :param  k_inv: The inverted matrix produced by the GNM analysis which is interpreted as a correlation matrix.
        :return:    The predicted hinge residues
        """
        pass


def predict_hinges(predicted_confidence_levels, local_sensitivity, local_percentile, global_percentile, alpha_max):
    """
    Method for hinges prediction using the predicted confidence levels
    The hinge prediction uses local confidence as well as global confidence to predict the hinge using the given
    parameters.
    :param predicted_confidence_levels: List of confidence levels for each residue for being a hinge
    :param local_sensitivity:   A parameter that sets the local range taken into account for the prediction
    :param local_percentile:    A hinge residue should be at the top local_percentile of its local neighborhood's
                                confidence levels.
    :param global_percentile:   A hinge residue should be at the top global_percentile of all residues' confidence
                                levels.
    :param alpha_max:           A hinge residue should have confidence level which is higher than
                                alpha_max * <max confidence level>
    :return:    A list of residues that are predicted as hinges.
    """

    m = len(predicted_confidence_levels)
    hinges = []

    total_max = max(predicted_confidence_levels)
    top_percentile = np.percentile(predicted_confidence_levels, global_percentile)

    for i in range(local_sensitivity, m-local_sensitivity):

        residue_confidence_level = predicted_confidence_levels[i]

        if residue_confidence_level < top_percentile:
            continue
        if residue_confidence_level < alpha_max * total_max:
            continue

        prediction_subset = predicted_confidence_levels[max(0, i - local_sensitivity):min(m, i + local_sensitivity)]

        if residue_confidence_level < np.percentile(prediction_subset, local_percentile):
            continue

        hinges.append(i)

    return hinges

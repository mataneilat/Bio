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


def predict_hinges(predicted_confidence_levels, local_sensitivity=7, max_average_alpha=0.9, total_max_beta=0.7):
    """
    Method for hinges prediction using the predicted confidence levels
    The hinge prediction uses local confidence as well as global confidence to predict the hinge using the given
    parameters.
    :param predicted_confidence_levels: List of confidence levels for each residue for being a hinge
    :param local_sensitivity:   A parameter that sets the local range taken into account for the prediction
    :param max_average_alpha:   A weight parameter that determines how close we expect a predicted hinge to be closer
                                to a local maximum than to a local average.
    :param total_max_beta:  A parameter that gives a threshold for how far can a predicted hinge be from the global
                            maximum.
    :return:    A list of residues that are predicted as hinges.
    """

    m = len(predicted_confidence_levels)
    hinges = []

    total_max = max(predicted_confidence_levels)
    for i in range(local_sensitivity, m-local_sensitivity):

        prediction_subset = predicted_confidence_levels[max(0, i - local_sensitivity):min(m, i + local_sensitivity)]
        subset_max = max(prediction_subset)
        subset_average = np.average(np.array(prediction_subset))

        residue_confidence_level = predicted_confidence_levels[i]
        if residue_confidence_level < max_average_alpha * subset_max + (1 - max_average_alpha) * subset_average:
            continue
        if residue_confidence_level < total_max_beta * total_max:
            continue
        hinges.append(i)
    return hinges

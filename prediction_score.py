
import numpy as np
import math

def score_prediction(predicted_hinges, annotated_hinges, total_residue_count, match_range=2, beta=1):
    """
    Assigns score to a given prediction with respected to the given annotated hinges

    Arguments:
        predicted_hinges:       List of predicted hinges
        annotated_hinges:       List of annotated hinges (The "ground truth")
        total_residue_count:    Integer representing the number of total number of residues.
        match_range:            The range for which which a prediction is considered as match.
        score_alpha:            The alpha parameters used to weight the score.

    Returns:
        A score for the prediction
    """

    annotated_hinges_considered_predicted = set()
    predicted_hinges_considered_annotated = set()


    for annotated_hinge in annotated_hinges:
        for predicted_hinge in predicted_hinges:
            if annotated_hinge in range(predicted_hinge - match_range, predicted_hinge + match_range + 1):
                annotated_hinges_considered_predicted.add(annotated_hinge)
                predicted_hinges_considered_annotated.add(predicted_hinge)


    all_residues = set(range(total_residue_count))
    condition_positives = set(annotated_hinges)
    condition_negatives = all_residues - condition_positives

    predicted_condition_positives = set(predicted_hinges)
    predicted_condition_negatives = all_residues - predicted_condition_positives

    true_positives = condition_positives & annotated_hinges_considered_predicted
    TP = len(true_positives)

    false_positives = (condition_negatives & predicted_condition_positives) - predicted_hinges_considered_annotated
    FP = len(false_positives)

    false_negatives = (condition_positives & predicted_condition_negatives) - annotated_hinges_considered_predicted
    FN = len(false_negatives)

    true_negatives = condition_negatives & predicted_condition_negatives
    TN = len(true_negatives)

    nom = float(TP * TN - FP * FN)
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if denom == 0:
        denom = 1
    return nom / denom


def predict_hinges(predicted_confidence_levels, local_sensitivity=7, max_mean_alpha=0.9, total_max_beta=0.7):
    """
    Method for hinges prediction using the predicted probabilities
    The hinge prediction uses local confidence as well as global confidence to predict the hinge using the given
    parameters.

    Arguments:
        predicted_confidence_levels: List of confidence levels for each residue for being a hinge
        local_sensitivity:  A parameter that sets the local range taken into account for the prediction
        max_mean_alpha:     A weight parameter that determines how close we expect a predicted hinge to be closer to a
                            local maximum than to a local mean.
        total_max_beta:     A parameter that gives a threshold for how far can a predicted hinge be from the global
                            maximum.

    Returns:
        A list of residues that are predicted as hinges.

    """
    m = len(predicted_confidence_levels)
    hinges = []

    total_max = max(predicted_confidence_levels)
    for i in range(local_sensitivity, m-local_sensitivity):

        prediction_subset = predicted_confidence_levels[max(0, i - local_sensitivity):min(m, i + local_sensitivity)]
        subset_max = max(prediction_subset)
        subset_mean = np.mean(np.array(prediction_subset))

        residue_confidence_level = predicted_confidence_levels[i]
        if residue_confidence_level < max_mean_alpha * subset_max + (1-max_mean_alpha) * subset_mean:
            continue
        if residue_confidence_level < total_max_beta * total_max:
            continue
        hinges.append(i)
    return hinges


# def score_prediction(predicted_hinges, annotated_hinges, total_residue_count, match_range=2, beta=1):
#     """
#     Assigns score to a given prediction with respected to the given annotated hinges
#
#     Arguments:
#         predicted_hinges:       List of predicted hinges
#         annotated_hinges:       List of annotated hinges (The "ground truth")
#         total_residue_count:    Integer representing the number of total number of residues.
#         match_range:            The range for which which a prediction is considered as match.
#         score_alpha:            The alpha parameters used to weight the score.
#
#     Returns:
#         A score for the prediction
#     """
#
#     condition_positive = annotated_hinges
#     condition_positive_count = len(condition_positive)
#
#     predicted_annotated_list = [False] * len(predicted_hinges)
#
#     true_positive_count = 0
#
#     for annotated_hinge in annotated_hinges:
#         was_predicted = False
#         for i, predicted_hinge in enumerate(predicted_hinges):
#             if annotated_hinge in range(predicted_hinge - match_range, predicted_hinge + match_range + 1):
#                 was_predicted = True
#                 predicted_annotated_list[i] = True
#         if was_predicted:
#             true_positive_count += 1
#
#     true_positive_rate = float(true_positive_count) / condition_positive_count
#
#     condition_negative_count = total_residue_count - condition_positive_count
#     true_negative_count = sum([1 if x else 0 for x in predicted_annotated_list])
#
#     true_negative_rate = float(true_negative_count) / condition_negative_count
#     one_minus_true_negative_rate = 1 - true_negative_rate
#
#     beta2 = beta * beta
#     return (1 + beta2) * ((true_positive_rate * one_minus_true_negative_rate) / (beta2 * true_positive_rate + one_minus_true_negative_rate))
#

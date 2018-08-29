"""
    This module contains different scores to measure prediction results
"""

import math


def calculate_mcc(predicted_hinges, annotated_hinges, total_residue_count, match_range=2):
    """
    Assigns a relaxed version of the Matthews correlation coefficient measure to a given prediction with respected to
    the annotated hinges

    Arguments:
        predicted_hinges:       List of predicted hinges
        annotated_hinges:       List of annotated hinges (The "ground truth")
        total_residue_count:    Integer representing the number of total number of residues.
        match_range:            The range for which which a prediction is considered as match.

    Returns:
        The MCC score.
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
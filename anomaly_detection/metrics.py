import numpy as np
from itertools import groupby
from operator import itemgetter


def _get_discords_errors(annotation: list, predicted_discords: list, m: int | list, full_report: bool = False) -> dict | tuple:
    """ 
    Calculate TruePositive (TP), FalsePositive (FP), and FalseNegative (FN) examples.

    Parameters
    --------
    true_discords : list
        The array of true discord indices. 

    predicted_discords : list
        The array of predicted discord indices.
    
    m : int | list
        The length(s) of discords. 

    full_report: bool, default False
        If f

    Returns
    -------
    errors : dict
        Calculated TP, FP, FN.

    discord_errors: dict
        Indices of TP, FP, FN discords. 
    """

    true_discords_annotation = np.where(annotation == 1)[0]

    true_discords = []
    true_discords_parts = []
    for k, g in groupby(enumerate(true_discords_annotation), lambda x: x[0] - x[1]):
        true_discords_part = list(map(itemgetter(1), g))
        true_discords_parts.append(true_discords_part)
        true_discords.append(true_discords_part[0])
    
    errors = {}
    discord_errors = {}

    top_k = len(predicted_discords)
    
    if type(m) is int:
        m = [m]*top_k

    # True Positive errors (number of true discords)
    TP_discords_true = []
    discord_errors['TP'] = []
    for i in range(top_k): # predicted discords
        predicted_idx = predicted_discords[i]
        discord_m = m[i]
        for true_idx in true_discords: # true discords
            if (true_idx in np.arange(predicted_idx, predicted_idx+discord_m)):
                discord_errors['TP'].append((predicted_idx, discord_m))
                TP_discords_true.append(true_idx)
                break
    errors['TP'] = len(discord_errors['TP'])

    # False Positive errors (number of false discords that are normal in annotation))
    discord_errors['FP'] = []
    FP_discords = set(predicted_discords) - set([tp[0] for tp in discord_errors['TP']])
    for FP_discord_idx in FP_discords:
        i = predicted_discords.index(FP_discord_idx)
        discord_errors['FP'].append((FP_discord_idx, m[i]))
    errors['FP'] = len(discord_errors['FP'])
 
    # False Negative errors (number of true discords that are not detected by algorithm)
    discord_errors['FN'] = []
    FN_discords = list(set(true_discords) - set(TP_discords_true))
    for FN_discord_idx in FN_discords:
        discords_idx_in_part = true_discords.index(FN_discord_idx)
        discord_part_len = len(true_discords_parts[discords_idx_in_part])
        if (discord_part_len == 1):
            discord_errors['FN'].append((FN_discord_idx, min(m)))     
        else:
            discord_errors['FN'].append((FN_discord_idx, discord_part_len))

    errors['FN'] = len(discord_errors['FN']) 

    if full_report:
        return errors, discord_errors
    else:
        return errors


def get_metrics(annotation: list, predicted_discords: list, m: int | list) -> dict:
    """
    Calculate accuracy metrics (Precision, Recall, and F1-measure) for discord discovery.
    
    Parameters
    --------
    true_discords : list
        The array of true discord indices. 

    predicted_discords : list
        The array of predicted discord indices.
    
    m : int | list
        The length(s) of discords. 

    Returns
    -------
    metrics : dict
        Calculated accuracy metrics.
    """
    
    metrics = {}

    errors = _get_discords_errors(annotation, predicted_discords, m)

    metrics['Precision'] = errors['TP'] / (errors['TP'] + errors['FP'])
    metrics['Recall'] = errors['TP'] / (errors['TP'] + errors['FN'])
    metrics['F1-measure'] = (2*metrics['Precision']*metrics['Recall'])/(metrics['Precision']+metrics['Recall'])

    return metrics
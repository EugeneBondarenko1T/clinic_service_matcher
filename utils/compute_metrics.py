"""Функции для расчёта метрик mapk, mark взятые из репозитория 
https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py"""

from typing import List

import numpy as np
import pandas as pd


def _ark(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average recall at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def _apk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average precision at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average precision at k.
    """
    if not predicted or not actual:
        return 0.0
    
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    true_positives = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], actual)
            true_positives += 1
    
    if score == 0.0:
        return 0.0
    
    return score / true_positives


def mark(actual: List[list], predicted: List[list], k=10) -> float:
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average recall at k (mar@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")

    return np.mean([_ark(a,p,k) for a,p in zip(actual, predicted)])


def mapk(actual: List[list], predicted: List[list], k: int=10) -> float:
    """
    Computes the mean average precision at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average precision at k (map@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")
    
    return np.mean([_apk(a,p,k) for a,p in zip(actual, predicted)])


def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec


def f1(value_mapk, value_mark):

    return 2 * (value_mapk * value_mark) / (value_mark + value_mapk)

def get_pivot_table(actual_list, predict_list, top_k):
    results = {'k': [], 'mark': [], 'mapk': [], 'f1_score': []}

    for k in range(1, top_k + 1):
        mark_value = mark(actual_list, predict_list, k=k)
        mapk_value = mapk(actual_list, predict_list, k=k)
        f1_value = f1(mark_value, mapk_value)
        
        results['k'].append(k)
        results['mark'].append(mark_value)
        results['mapk'].append(mapk_value)
        results['f1_score'].append(f1_value)

    return pd.DataFrame(results)
import numpy as np
### https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

def apk(actual, predicted, k=10):
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

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def recall_at_k(actual, predicted, k=5):

    if len(predicted) > k:
        predicted = predicted[:k]

    num_relevant = len(set(actual) & set(predicted))
    return num_relevant / len(actual)

def mark(actual_list, predicted_list, k=5):

    return np.mean([recall_at_k(a, p, k) for a, p in zip(actual_list, predicted_list)])
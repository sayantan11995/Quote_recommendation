import numpy as np


def get_accuracy(preds, labels):
    """Compute the accuracy of binary predictions.

    Returns:
        accuracy: float
    -----------------
    Arguments:
        preds: Numpy list with two columns of probabilities for each label
        labels: List of labels
    """
    # Get the label (column) with the higher probability
    predictions = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    # Compute accuracy
    accuracy = np.sum(predictions == labels) / len(labels)

    return accuracy

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
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

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def top_k_acc(true_labels, pred_labels, k=1):
    top_k_acc_score = 0

    for id, labels in pred_labels.items():
        if len(list(set(labels[:k]).intersection(set(true_labels[id])))) > 0:
            top_k_acc_score += 1

    return top_k_acc_score/len(pred_labels)



# import ml_metrics

def mAP_k(true_labels, pred_labels, k=5):

    true_label_list = list(dict(sorted(true_labels.items())).values())
    pred_label_list = list(dict(sorted(pred_labels.items())).values())

    return mapk(true_label_list, pred_label_list, k) 

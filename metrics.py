import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc(gt:list, pred:list, class_num:int)->list:
    """compute the auc score and roc score

    Args:
        gt (list): ground truth
        pred (list): prediction
        class_num (int): the number of classes

    Returns:
        list: auc and roc scores
    """    
    auc_roc_scores = []
    gt = np.array(gt)
    pred = np.array(pred)
    for i in range(class_num):
        try:
            auc_roc_scores.append(roc_auc_score(gt[:, i], pred[:, i]))
        except:
            auc_roc_scores.append(0.)
    return auc_roc_scores



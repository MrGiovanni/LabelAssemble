import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import scipy.stats as st
from random import sample



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




def compute_conf(gt, pred, threshold:int=0.5, sample_time:int=100, sample_ratio:int=0.5)->dict:
    def sampling_dataset(gt, predict):
        metrics = {'accuracy': [], 'recall': [], 'specificity': []}

        for _ in (range(sample_time)):
            rand_index = sample([i for i in range(len(gt))], int(sample_ratio*len(gt)))
            tn, fp, fn, tp = confusion_matrix(gt[rand_index, 0], predict[rand_index, 0]>threshold).ravel()
            metrics['recall'].append(tp/(tp+fn))
            metrics['specificity'].append(tn/(tn+fp))
            metrics['accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        return metrics
    metrics = sampling_dataset(gt, pred)
    results = {}
    for metric_name in metrics.keys():
        lower, upper = st.t.interval(alpha=0.95, df=len(metrics[metric_name])-1, 
                                    loc=np.mean(metrics[metric_name]), 
                                    scale=st.sem(metrics[metric_name])) 
        results[metric_name] = [metrics[metric_name], lower, upper] 
    return results

from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(true_labels, cluster_labels):
    # Create a confusion matrix
    cm = confusion_matrix(true_labels, cluster_labels)
    
    # Use the Hungarian Algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize the assignment (use -cm for maximization)
    
    # Calculate the accuracy
    total_correct = cm[row_ind, col_ind].sum()
    accuracy = total_correct / len(true_labels)
    
    return accuracy, cm, (row_ind, col_ind)

def compute_auc_crr(scores, y_test, max_known_label, return_lists=False, is_rotation=False, initial_n_known=None):
    # Label outliers as 1 and known class samples as 0
    if not is_rotation:
        y_outlier_true = (y_test > max_known_label).astype(int)  # 1 for true outliers, 0 for known classes
    else:
        y_outlier_true = ((y_test >= initial_n_known) & (y_test != max_known_label)).astype(int)  # 1 for true outliers, 0 for known classes
    outlier_scores = np.max(scores, axis=1)  # Use max decision score for each sample
    # Flip the scores to treat higher values as more likely to be outliers
    negated_outlier_scores = -outlier_scores

    # Compute the AUC of CRR (TNR) by passing true labels and negated scores directly
    if not return_lists:
        auc_crr = roc_auc_score(y_outlier_true, negated_outlier_scores)
        return auc_crr
    else:
        fpr, tpr, thresholds = roc_curve(y_outlier_true, negated_outlier_scores)
        return auc(fpr, tpr), fpr, tpr, thresholds

def compute_oscr_ours(scores, y_test, num=100000, return_lists=False):
    known_scores = scores[y_test > -1]  # Known samples
    unknown_scores = scores[y_test <= -1]  # Unknown samples
    known_max_scores = known_scores.max(axis=1)  # Max score across classes for each known sample
    unknown_max_scores = unknown_scores.max(axis=1)  # Max score across classes for each unknown sample
    y_test_known = y_test[y_test > -1]  # True labels for known samples

    thresholds = np.linspace(scores.min(), scores.max(), num=num)
    ccr_list = []
    fpr_list = []
    for threshold in thresholds:
        ccr = np.sum((known_max_scores >= threshold) & (np.argmax(known_scores, axis=1) == y_test_known)) / len(known_max_scores)
        ccr_list.append(ccr)
        
        fpr = np.sum(unknown_max_scores >= threshold) / len(unknown_max_scores)
        fpr_list.append(fpr)

    oscr_auc = auc(fpr_list, ccr_list)  # FPR on x-axis, CCR on y-axis

    if return_lists:
        return oscr_auc, fpr_list, ccr_list

    return oscr_auc


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1) # pred_k和pred_u预测概率的最大值
    pred = np.argmax(pred_k, axis=1) #known样本的预测值
    correct = (pred == labels) 
    m_x1 = np.zeros(len(x1)) 
    m_x1[pred == labels] = 1
    
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0) #known样本的预测目标值
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0) #unkown的预测目标
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict) #unknown+known样本数量

    # Cutoffs are of prediction values
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort() #从小到大排序
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1): # k是阈值，概率大于k位置的样本判断为know,否则为unknown
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR
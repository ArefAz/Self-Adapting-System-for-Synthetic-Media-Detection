import numpy as np
import torch
import warnings

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, roc_curve, auc
from eval import compute_auc_crr


class OpenSetModel:

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        is_bayesian=False,
        min_ood_tpr=0.90,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm_dict = {}
        self.thresholds = []
        self.is_bayesian = is_bayesian
        self.known_sources = []
        self.min_ood_tpr = min_ood_tpr
        self.autoencoder = None

    @property
    def n_known_sources(self):
        return len(self.gmm_dict)
    
    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder

    def train_gmms(self, X, y):
        for i in np.unique(y):
            if self.is_bayesian:
                gmm = BayesianGaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    # n_init=10,
                    reg_covar=1e-5,
                )
            else:
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    # n_init=10,
                    reg_covar=1e-5,
                )

            gmm.fit(X[y == i])
            self.gmm_dict[i] = gmm

    def fit(self, X, y):
        self.gmm_dict = {}
        self.train_gmms(X, y)

    def update(self, X, y):
        # make sure that the new source labels are all larger than the known sources
        new_sources_labels = np.unique(y)
        cond = np.all(new_sources_labels == self.n_known_sources)
        if not cond:
            print(f"New source labels: {new_sources_labels}, Known sources: {self.n_known_sources}")
            raise ValueError("New source labels must be larger than the known sources")

        if len(new_sources_labels) == 0:
            warnings.warn("No new sources to update")
            return

        self.train_gmms(X, y)

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.gmm_dict)))
        for i, gmm in self.gmm_dict.items():
            scores[:, i] = gmm.score_samples(X)
        return scores

    def classify(self, X):
        scores = self.predict(X)
        src_preds = np.argmax(scores, axis=1)
        if len(self.thresholds):
            rejections = self.ood_detect(X)
            src_preds[rejections] = -1
        else:
            warnings.warn("No threshold set")

        return src_preds
    
    def get_cm_labels(self, X, y, add_new_source=False):
        preds = self.classify(X)
        if add_new_source:
            labels_range = np.arange(-1, self.n_known_sources + 1)
        else:
            labels_range = np.arange(-1, self.n_known_sources)
        # print(f"Labels range: {labels_range}")
        # print(f"Unique labels: {np.unique(y)}, {np.unique(preds)}")
        # raise ValueError("Stop here")
        cm = confusion_matrix(y, preds, labels=labels_range)
        return cm, labels_range

    def evaluate(self, X, y):
        preds = self.classify(X)
        # print(f"Unique labels: {np.unique(y, return_counts=True)}")
        # print(f"Unique preds: {np.unique(preds, return_counts=True)}")
        cm = confusion_matrix(y, preds, labels=np.arange(-1, self.n_known_sources))
        b_acc = round(balanced_accuracy_score(y, preds), 4)

        y_current = y[y == y.max()]
        preds_current = preds[y == y.max()]
        b_acc_current = round(balanced_accuracy_score(y_current, preds_current), 4)

        y_detection = (y > 0).astype(int)
        preds_detection = (preds > 0).astype(int)
        acc_detection = round(balanced_accuracy_score(y_detection, preds_detection), 4)

        y_detection_current = y_detection[y == y.max()]
        preds_detection_current = preds_detection[y == y.max()]
        acc_detection_current = round(balanced_accuracy_score(y_detection_current, preds_detection_current), 4)

        results = {
            "confusion_matrix": cm,
            "balanced_accuracy": b_acc,
            "balanced_accuracy_current": b_acc_current,
            "detection_accuracy": acc_detection,
            "detection_accuracy_current": acc_detection_current
        }
        return results
    
    def evaluate_ood(self, ood_decisions, ood_labels):
        cm = confusion_matrix(ood_labels, ood_decisions)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        results = {
            "ood_cm": cm,
            "ood_tpr": round(tpr, 4),
            "ood_fpr": round(fpr, 4),
            "ood_acc": round(acc, 4),
        }
        return results
        

    def ood_detect(self, X):
        # a sample is detected ood if all GMMs have a score lower than the threshold
        scores = self.predict(X)
        rejections = np.zeros(scores.shape, dtype=bool)
        for i, threshold in enumerate(self.thresholds):
            rejections[:, i] = scores[:, i] < threshold
        
        return np.all(rejections, axis=1).astype(int)
    
    def roc_curve(self, truths, preds, range_, num_points):
        thresholds = np.linspace(range_[0], range_[1], num=num_points)
        tprs = []
        fprs = []
        for threshold in thresholds:
            preds_ = preds > threshold
            tn, fp, fn, tp = confusion_matrix(truths, preds_).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tprs.append(tpr)
            fprs.append(fpr)
        return np.array(fprs), np.array(tprs), thresholds


    def find_best_thresholds(self, X, y):
        
        for i, gmm in self.gmm_dict.items():
            scores = gmm.score_samples(X)
            y_ind = (y == i).astype(int)
            range_ = (scores.min(), scores.max())
            fprs, tprs, thresholds = self.roc_curve(y_ind, scores, range_, 5000)
            max_j = np.max(tprs - fprs)
            acceptable_tuples = [
                (fpr, tpr, threshold) for fpr, tpr, threshold in zip(fprs, tprs, thresholds) if np.abs((tpr - fpr) - max_j) < 0.001 # and tpr >= self.min_ood_tpr
            ]
            if len(acceptable_tuples):
                best_tuple = min(acceptable_tuples, key=lambda x: (x[0], )) 
                best_threshold = best_tuple[2]
            else:
                best_threshold = np.max(scores)
                warnings.warn(f"No acceptable threshold found for GMM {i}, using max score as fallback")
            # from IPython import embed; embed()
            # best_threshold = thresholds[np.argmax(tprs - fprs)]
            # max_j = np.max(tprs - fprs)
            # best_thresholds = thresholds[np.abs((tprs - fprs) - max_j) < 0.025]
            # best_threshold = np.min(best_thresholds)
            # best_threshold_idx = thresholds.tolist().index(best_threshold)
            # best_tuple = (fprs[best_threshold_idx], tprs[best_threshold_idx], best_threshold)
            # best_threshold = thresholds[np.argmax(tprs - fprs)]
            # if best_threshold < 0:
            #     from ipdb import set_trace; set_trace()
            #     pass
            # best_threshold_idx = np.argmax(tprs - fprs)
            # best_threshold = thresholds[best_threshold_idx]
            # best_tuple = (fprs[best_threshold_idx], tprs[best_threshold_idx], best_threshold)
            # if len(np.unique(y_ind)) != 1:
            auc_value = round(roc_auc_score(y_ind, scores), 4)
            # else:
            #     auc_value = 0.0
            best_tuple = [round(x, 4) for x in best_tuple]
            # print(f"GMM {i} Best threshold: {best_threshold}, auc: {auc_value}, fpr: {best_tuple[0]}, tpr: {best_tuple[1]}")
            # print(f"GMM {i} Best threshold: {best_threshold}, auc: {auc_value}")
            
            self.thresholds.append(best_threshold)       
            
    def set_threshold(self, threshold):
        self.threshold = threshold

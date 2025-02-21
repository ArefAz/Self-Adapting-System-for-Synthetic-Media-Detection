import numpy as np
import torch
import warnings

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from eval import compute_auc_crr


class OpenSetModel:

    def __init__(self, n_components=1, covariance_type="full", is_bayesian=False, min_ood_tpr=0.90):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm_list = []
        self.threshold = None
        self.is_bayesian = is_bayesian
        self.known_sources = []
        self.min_ood_tpr = min_ood_tpr

    
    def train_gmms(self, X, y):
        for i in np.unique(y):
            if self.is_bayesian:
                gmm = BayesianGaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                )
            else:
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                )

            gmm.fit(X[y == i])
            self.gmm_list.append(gmm)

    def fit(self, X, y):
        self.gmm_list = []
        self.train_gmms(X, y)
        
    def update(self, X, y):
        # make sure that the new source labels are all larger than the known sources
        new_sources = np.unique(y)
        cond = np.all(new_sources > len(self.gmm_list))
        if not cond:
            raise ValueError("New source labels must be larger than the known sources")

        if len(new_sources) == 0:
            warnings.warn("No new sources to update")
            return
        
        self.known_sources += new_sources
        self.train_gmms(X, y)

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.gmm_list)))
        for i, gmm in enumerate(self.gmm_list):
            scores[:, i] = gmm.score_samples(X)
        return scores
    
    def classify(self, X):
        scores = self.predict(X)
        src_preds = np.argmax(scores, axis=1)
        if self.threshold is not None:
            src_preds[src_preds < self.threshold] = -1
        else:
            warnings.warn("No threshold set")
        return src_preds
    
    def ood_detect(self, X):
        classification = self.classify(X)
        return (classification == -1).astype(int)
    
    def find_best_threshold(self, X, y):
        scores = self.predict(X)
        auc_crr, fprs, tprs, thresholds = compute_auc_crr(scores, y, return_lists=True)
        acceptable_thresholds = []
        for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
            if tpr >= self.min_ood_tpr:
                acceptable_thresholds.append((fpr, tpr, threshold))

        # Step 2: Select the threshold with the lowest FPR
        if acceptable_thresholds:
            best_threshold_tuple = min(acceptable_thresholds, key=lambda x: x[0])  # x[0] is the FPR
            selected_fpr, selected_tpr, best_threshold = best_threshold_tuple
        else:
            raise ValueError("No acceptable thresholds found")

        return best_threshold

    def set_threshold(self, threshold):
        self.threshold = threshold
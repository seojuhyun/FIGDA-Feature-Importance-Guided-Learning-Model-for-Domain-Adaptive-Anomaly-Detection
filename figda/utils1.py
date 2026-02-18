#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is based on the original DevNet implementation by:
Guansong Pang, Chunhua Shen, and Anton van den Hengel.
Deep Anomaly Detection with Deviation Networks.
KDD 2019. https://doi.org/10.1145/3292500.3330871
Original implementation author: Guansong Pang

Modifications by:
Juhyun Seo (2026)
- Modified dataLoading function to accept pandas DataFrame
- Minor compatibility adjustments
This implementation is distributed for research and academic purposes.
The original DevNet license terms apply to the inherited portions of the code.
"""


import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from joblib import Memory  
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os

def dataLoading(path):

    df= path
    labels = df['class']   # 내가 수정함    
    x_df = df.drop(['class'], axis=1)
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    
    return x, labels, df;


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


def plot_roc_curve(scores, labels, save_path=None, dataset_name=None):
    """
    Plot ROC curve for anomaly detection.
    Returns
    -------
    roc_auc : float
    """

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "ROC Curve"
    if dataset_name:
        title += f" ({dataset_name})"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        filename = os.path.join(save_path, f"roc_curve_{dataset_name}.svg")
        plt.savefig(filename, format="svg", bbox_inches="tight")

    plt.close()

    return roc_auc

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(scores, labels, save_path=None, dataset_name=None):
    """
    Plot Precision-Recall curve for anomaly detection.
    Returns
    -------
    aupr : float
    """

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = average_precision_score(labels, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPR = {aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    title = "Precision-Recall Curve"
    if dataset_name:
        title += f" ({dataset_name})"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        filename = os.path.join(save_path, f"pr_curve_{dataset_name}.svg")
        plt.savefig(filename, format="svg", bbox_inches="tight")

    plt.close()

    return aupr

def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap, train_time, test_time, path = "./results/auc_performance_cl0.5.csv"):    
    csv_file = open(path, 'a') 
    row = name + "," + str(n_samples)+ ","  + str(dim) + ',' + str(n_samples_trn) + ','+ str(n_outliers_trn) + ','+ str(n_outliers)  + ',' + str(depth)+ "," + str(rauc) +"," + str(std_auc) + "," + str(ap) +"," + str(std_ap)+"," + str(train_time)+"," + str(test_time) + "\n"
    csv_file.write(row)
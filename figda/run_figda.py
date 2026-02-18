#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGDA training script (FI-guided DevNet).

Based on the DevNet implementation:
Pang et al., KDD 2019.

Implemented by
Juhyun Seo, 2026.
(Associated with FIGDA, PAKDD 2026)
"""

import sys
import os
import time
import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost.callback import EvaluationMonitor
from devnet_base import *
from utils1 import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(int(seed))  # ensure it's scalar
    import random
    random.seed(seed)

def fi_regularized_loss(model, fi_weights, lambda_fi):
    fi_weights_tf = tf.convert_to_tensor(fi_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        base_loss = deviation_loss(y_true, y_pred)

        # 학습 중 현재 가중치 가져오기 (첫 Dense 레이어)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                current_weights = layer.kernel  # 학습 도중 계속 바뀜
                break

        # FI 기반 정규화 항 계산
        reg_term = tf.reduce_mean(tf.square(current_weights - fi_weights_tf))
        return base_loss + lambda_fi * reg_term

    return loss_fn


class FIGDAClassifier(torch.nn.Module):
    def __init__(self, data, dataset_name, output_path, num_layers, num_layers_boosted=1,
                 batch_size=512, nb_batch=20, epochs=100,
                 alpha_first_layer=0.873,
                 lambda_fi= 5.25
                 ): 
        super().__init__()
        self.data = data
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.epochs = epochs
        self.output_path = output_path
        self.alpha_first_layer = alpha_first_layer
        self.lambda_fi = lambda_fi
        self.first_layer_fi = None
        self.seeds = [971, 879, 796, 623,  18, 135,  66, 933, 200, 805]

    def forward(self, train=True):
        seeds = self.seeds
        all_metrics = []
        all_val_losses = []
        all_diff_norms = []
        P_list, R_list, F1_list = [], [], []
        # forward마다 초기화 (누적 방지)
        mid_devs, final_devs = [], []
        dist_per_seed, aupr_per_seed, auroc_per_seed = [], [], []


        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        line_styles = ['-', '--', '-.', ':'] * 3

        for run_idx, seed in enumerate(seeds):
            print(f"\n🔁 Run {run_idx + 1} with seed {seed}")
            set_seed(seed)
            x, labels, data = dataLoading(self.data)
            X_train, X_val, y_train, y_val, X_test, y_test, outlier_indices = prepare_data(data)

            rng = np.random.RandomState(seed)
            
            if run_idx == 0:
                print("🔍 Running RandomizedSearchCV to find best XGBClassifier for FI extraction...")
                param_dist = {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.05],
                            'max_depth': [3, 5, 6, 8],
                            'subsample': [0.6, 0.8, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0]
                                }
                search = RandomizedSearchCV(
                estimator = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    #use_label_encoder=False,
                    random_state=seed),
                    param_distributions=param_dist,
                    n_iter=20,  # 조합 개수 (시간 여유 있으면 더 늘려도 좋음)
                    scoring='roc_auc',
                    cv=3,
                    verbose=0,
                    random_state=seed,
                    n_jobs=-1
                )

                search.fit(X_train, y_train.astype(int).ravel())

                best_xgb = search.best_estimator_
                print(f"✅ Best XGBoost params: {search.best_params_}")
                fi = best_xgb.feature_importances_
                self.first_layer_fi = np.asarray(fi, dtype=np.float32)
                
            fi = self.first_layer_fi
          
            devnet_model = deviation_network(
                input_shape=(X_train.shape[1],),
                network_depth=6,
                feature_importance_weights=self.first_layer_fi,
                alpha_dict={1: self.alpha_first_layer} 
            )
            
            fi_weights = np.tile(
                self.first_layer_fi.reshape(-1, 1),
                (1, devnet_model.layers[1].units)
            )

            class FIDeviationCallback(tf.keras.callbacks.Callback):
                def __init__(self, fi_w, x_val, y_val):
                    super().__init__()
                    self.fi_w = fi_w
                    self.x_val = x_val
                    self.y_val = y_val
                    
                    self.epoch_devs = []
                    self.epoch_aupr = []
                    self.epoch_auroc = []

                def on_epoch_end(self, epoch, logs=None):
                    W1, _ = self.model.layers[1].get_weights()
                    diff = np.abs(W1 - self.fi_w)
                    dev_l1 = diff.mean()
                    self.epoch_devs.append(dev_l1)

                    # 🔥 validation performance for trajectory curve
                    scores = self.model.predict(self.x_val, verbose=0).flatten()
                    aupr  = average_precision_score(self.y_val, scores)
                    auroc = roc_auc_score(self.y_val, scores)

                    self.epoch_aupr.append(aupr)
                    self.epoch_auroc.append(auroc)
                    
            fi_callback = FIDeviationCallback(fi_weights, X_val, y_val)
            
           # Feature Importance 텐서 준비 (고정값)
            fi_weights_tf = tf.convert_to_tensor(
                np.tile(self.first_layer_fi.reshape(-1, 1), (1, devnet_model.layers[1].units)),
                dtype=tf.float32
            )
         

            # Custom loss 함수 정의 (학습 중 실시간 kernel 참조)
            custom_loss = fi_regularized_loss(
                model=devnet_model,
                fi_weights=fi_weights_tf,
                lambda_fi= self.lambda_fi
            )
                        
            optimizer = Adam(learning_rate=0.001)
            devnet_model.compile(optimizer=optimizer, loss=custom_loss)

            checkpointer = ModelCheckpoint(filepath=os.path.join(self.output_path, f"run_{run_idx + 1}_best_model.h5"), monitor='loss', verbose=0, save_best_only=True)
            early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, restore_best_weights=True)

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            generator = batch_generator_sup(X_train, outlier_indices, inlier_indices, self.batch_size, self.nb_batch, rng)

            history = devnet_model.fit(
                x=generator,
                steps_per_epoch=self.nb_batch,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=[checkpointer, early_stop, fi_callback],  
                verbose=0
            )

            all_val_losses.append((history.history['val_loss'], colors[run_idx], line_styles[run_idx]))

            scores = devnet_model.predict(X_test)
            scores_flat = np.asarray(scores).reshape(-1)

            # PR-curve → Best-F1 임계값
            prec_curve, rec_curve, thr = precision_recall_curve(y_test, scores_flat)
            f1_arr = 2 * prec_curve[:-1] * rec_curve[:-1] / (prec_curve[:-1] + rec_curve[:-1] + 1e-12)
            best_idx = int(np.argmax(f1_arr))
            best_thr = float(thr[best_idx])

            # 임계값 적용 (>= 권장)
            eps = np.finfo(scores_flat.dtype).eps if np.issubdtype(scores_flat.dtype, np.floating) else 1e-12
            y_pred = (scores_flat > (best_thr + eps)).astype(int)

            
            precision_val = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall_val    = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1_val        = f1_score(y_test, y_pred, average='binary', zero_division=0)
            accuracy_val  = accuracy_score(y_test, y_pred)

            P_list.append(precision_val)
            R_list.append(recall_val)
            F1_list.append(f1_val)

            rauc, ap = aucPerformance(scores, y_test)

            all_metrics.append([rauc, ap, accuracy_val, precision_val, recall_val, f1_val])
                    
            epoch_devs = fi_callback.epoch_devs  # 길이 = 실제 학습된 epoch 수

            # 10 epoch 근처 값 (실제로는 min(10, 마지막 epoch) 사용)
            mid_idx = min(4, len(epoch_devs) - 1)   # 0-based index라 9가 10번째 epoch
            mid_dev = epoch_devs[mid_idx]

            # 최종(epoch_end) 값
            final_dev = epoch_devs[-1]

            mid_devs.append(mid_dev)
            final_devs.append(final_dev)
            dist_per_seed.append(fi_callback.epoch_devs)
            aupr_per_seed.append(fi_callback.epoch_aupr)
            auroc_per_seed.append(fi_callback.epoch_auroc)


            learned_weights, _ = devnet_model.layers[1].get_weights()
            fi_weights = np.tile(self.first_layer_fi.reshape(-1, 1), (1, learned_weights.shape[1]))
            diff_norm = np.abs(learned_weights - fi_weights)
            all_diff_norms.append(diff_norm)

        avg_diff_norm = np.mean(np.stack(all_diff_norms), axis=0)
        avg_deviation = np.mean(avg_diff_norm)
        # 새 방식: 에폭 기준 scalar 편차
        avg_mid_dev   = np.mean(mid_devs)
        avg_final_dev = np.mean(final_devs)

        
        print(f"\n📏 Average FI-Weight Deviation (L1 norm): {avg_deviation:.6f}")
        print(f"\n📏 Mid (≈10 epoch) FI-Weight Deviation:  {avg_mid_dev:.6f}")
        print(f"📏 Final FI-Weight Deviation:            {avg_final_dev:.6f}")
        
        # 평균 & 표준편차 계산 추가
        metrics_array = np.array(all_metrics)
        metric_names = ["AUC-ROC", "AUC-PR", "Accuracy", "Precision", "Recall", "F1"]

        avg_metrics = np.mean(metrics_array, axis=0)
        std_metrics = np.std(metrics_array, axis=0, ddof=0)

        print("\n📊 Average over 10 runs:")
        for name, mean, std in zip(metric_names, avg_metrics, std_metrics):
            print(f"🔹 {name}: {mean:.3f} ± {std:.3f}")

        # CSV 저장 (mean ± std)
        df_results = pd.DataFrame([{
            "DataSet": self.dataset_name,
            "AUC-ROC(mean)": avg_metrics[0],
            "AUC-ROC(std)": std_metrics[0],
            "AUC-PR(mean)": avg_metrics[1],
            "AUC-PR(std)": std_metrics[1],
            "Accuracy(mean)": avg_metrics[2],
            "Accuracy(std)": std_metrics[2],
            "Precision(mean)": avg_metrics[3],
            "Precision(std)": std_metrics[3],
            "Recall(mean)": avg_metrics[4],
            "Recall(std)": std_metrics[4],
            "F1-Score(mean)": avg_metrics[5],
            "F1-Score(std)": std_metrics[5]
        }])

        results_path = os.path.join(self.output_path, "figdaSC_results_avg_std_10runs.csv")
        file_exists = os.path.isfile(results_path)
        df_results.to_csv(results_path, mode='a', index=False, header=not file_exists)



# ============================================================
# FIGDA Hyperparameters (PAKDD 2026 Final Experimental Setting)
# alpha_first_layer: controls FI-based weight initialization strength
# lambda_fi: FI regularization coefficient
# ============================================================
#
# Dataset      | alpha_first_layer | lambda_fi
# ------------------------------------------------------------
# Census       | 0.206             | 112
# Bank         | 0.15255           | 0.50825
# Fraud        | 0.873             | 5.25
# CelebA       | 0.09873           | 0.58341
#
# These values were selected via validation experiments.
# ============================================================
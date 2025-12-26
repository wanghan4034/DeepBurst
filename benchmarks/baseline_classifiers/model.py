
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def train_and_eval_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_name: str,
    split_id: int,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    在给定的 train/test 集上，分别训练：
      - 逻辑回归 LogisticRegression
      - 随机森林分类 RandomForestClassifier
    不划分验证集。
    """
    print(f"\n========== Task: {task_name}, Split {split_id} ==========")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 如果训练集中只有一个类别，直接跳过这一折
    unique_train = np.unique(y_train)
    if unique_train.size < 2:
        print(f"[Warn] only one class ({unique_train}) in training set for {task_name}, split {split_id}, skip.")
        return {
            "logit": {"acc": np.nan, "auc": np.nan},
            "rf": {"acc": np.nan, "auc": np.nan},
        }

    # ---------- 1) Logistic Regression ----------
    logit_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                multi_class="auto",
                n_jobs=-1,
            )),
        ]
    )
    logit_pipe.fit(X_train, y_train)
    y_pred_logit = logit_pipe.predict(X_test)

    acc_logit = accuracy_score(y_test, y_pred_logit)

    print("\n--- Logistic Regression ---")
    print(f"Accuracy: {acc_logit:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_logit, digits=3))

    # 二分类才算 ROC AUC
    unique_all = np.unique(np.concatenate([y_train, y_test]))
    if unique_all.size == 2:
        y_proba_logit = logit_pipe.predict_proba(X_test)[:, 1]
        try:
            auc_logit = roc_auc_score(y_test, y_proba_logit)
        except ValueError:
            auc_logit = np.nan
        print(f"ROC AUC: {auc_logit:.3f}")
    else:
        auc_logit = np.nan

    # ---------- 2) Random Forest Classifier ----------
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("\n--- Random Forest Classifier ---")
    print(f"Accuracy: {acc_rf:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_rf, digits=3))

    if unique_all.size == 2:
        y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
        try:
            auc_rf = roc_auc_score(y_test, y_proba_rf)
        except ValueError:
            auc_rf = np.nan
        print(f"ROC AUC: {auc_rf:.3f}")
    else:
        auc_rf = np.nan

    return {
        "logit": {"acc": acc_logit, "auc": auc_logit},
        "rf": {"acc": acc_rf, "auc": auc_rf},
    }

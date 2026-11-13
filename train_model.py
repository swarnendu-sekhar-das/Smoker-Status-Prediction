"""
========================================================================================
AIT 511 Machine Learning - Course Project II: Smoker Status Prediction
Standard ML & DL Coursework Models Pipeline
Author: Swarnendu Sekhar Das (M.Tech CSE, IIIT Bangalore)
----------------------------------------------------------------------------------------
Description:
This script builds a standard machine learning and deep learning pipeline
to predict smoker status using bio-signal indicators.

Standard Coursework Models Evaluated:
1. Logistic Regression (Linear baseline with StandardScaler)
2. Multi-Layer Perceptron (Standard Deep Learning Feedforward Neural Network)
3. Random Forest Classifier (Standard Decision Tree Bagging Ensemble)

Key Features:
- 6 intuitive clinical features (BMI, Pulse Pressure, De Ritis Ratio, Lipid Ratio, etc.).
- 5-Fold Stratified Cross-Validation (prevents data leakage).
- Decision threshold calibration on the best model for optimal clinical metrics.
- Export of test predictions and publication-ready evaluation charts.
========================================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn standard ML tools
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# PyTorch for Deep Learning (Multi-Layer Perceptron)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------------------------------------------
# 1. Reproducibility & Seed Configuration
# --------------------------------------------------------------------------------------
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Info] Running on compute device: {device}")

# --------------------------------------------------------------------------------------
# 2. Paths Configuration
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train_dataset.csv")
TEST_PATH = os.path.join(BASE_DIR, "test_dataset.csv")
PRED_OUT_PATH = os.path.join(BASE_DIR, "predictions.csv")

print(f"[Info] Loading datasets from: {BASE_DIR}")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"[Info] Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"[Info] Class balance: {train_df['smoking'].value_counts().to_dict()} (0: Non-smoker, 1: Smoker)")

# --------------------------------------------------------------------------------------
# 3. Clinically Grounded Feature Engineering
# --------------------------------------------------------------------------------------
def add_clinical_features(df):
    """
    Adds 6 standard, intuitive clinical features grounded in medical literature:
    1. BMI (Body Mass Index): weight(kg) / height(m)^2. Standard metabolic indicator.
    2. Pulse Pressure: systolic - relaxation blood pressure. Measures arterial stiffness.
    3. De Ritis Ratio: AST / ALT. Classic diagnostic ratio for liver stress.
    4. Cholesterol Risk Ratio: (LDL + triglyceride) / HDL. Atherogenic dyslipidemia index.
    5. Hemoglobin-Age Ratio: Compensatory oxygen response in chronic smokers.
    6. Log(GTP): Log-transform variance stabilization for heavily right-skewed liver enzyme.
    """
    df_feat = df.copy()

    # 1. Body Mass Index (BMI)
    height_m = df_feat['height(cm)'] / 100.0
    df_feat['BMI'] = df_feat['weight(kg)'] / (height_m ** 2)

    # 2. Pulse Pressure (arterial stiffness indicator)
    df_feat['Pulse_Pressure'] = df_feat['systolic'] - df_feat['relaxation']

    # 3. De Ritis Ratio (AST/ALT ratio for liver function)
    df_feat['De_Ritis_Ratio'] = df_feat['AST'] / (df_feat['ALT'] + 0.1)

    # 4. Cholesterol Risk Ratio (atherogenic index)
    df_feat['Cholesterol_Risk_Ratio'] = (df_feat['LDL'] + df_feat['triglyceride']) / (df_feat['HDL'] + 0.1)

    # 5. Hemoglobin to Age Ratio (physiological adaptation to hypoxia)
    df_feat['Hemoglobin_Age_Ratio'] = df_feat['hemoglobin'] / (df_feat['age'] + 1)

    # 6. Log-GTP (dampens extreme positive outliers)
    if 'Gtp' in df_feat.columns:
        df_feat['Log_Gtp'] = np.log1p(df_feat['Gtp'])

    return df_feat

train_feat = add_clinical_features(train_df)
test_feat = add_clinical_features(test_df)

X = train_feat.drop(columns=['smoking'])
y = train_feat['smoking'].values
X_test = test_feat.copy()

feature_names = X.columns.tolist()
print(f"[Info] Total features after engineering: {len(feature_names)} (Original 22 + 6 Clinical)")

# --------------------------------------------------------------------------------------
# 4. Standard Deep Learning Architecture: Multi-Layer Perceptron (MLP)
# --------------------------------------------------------------------------------------
class SmokerMLP(nn.Module):
    """
    Standard Feedforward Deep Learning Neural Network:
    - 3 Hidden layers with ReLU activations
    - Batch Normalization for stable gradient propagation
    - Dropout (0.15, 0.10) for gentle regularization
    - Sigmoid output for calibrated probability prediction
    """
    def __init__(self, input_dim):
        super(SmokerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_pytorch_mlp(X_tr, y_tr, X_val, y_val, input_dim, epochs=30, batch_size=64, lr=5e-4):
    """Trains a standard PyTorch MLP with Adam optimizer and Binary Cross-Entropy loss."""
    model = SmokerMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_tensor_x = torch.tensor(X_tr, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_preds = model(val_tensor_x).cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_preds)

        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = model.state_dict()

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        final_val_preds = model(val_tensor_x).cpu().numpy().flatten()

    return model, final_val_preds

# --------------------------------------------------------------------------------------
# 5. 5-Fold Stratified Cross-Validation Pipeline
# --------------------------------------------------------------------------------------
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_lr = np.zeros(len(y))
oof_mlp = np.zeros(len(y))
oof_rf = np.zeros(len(y))

test_preds_lr = np.zeros(len(X_test))
test_preds_mlp = np.zeros(len(X_test))
test_preds_rf = np.zeros(len(X_test))

print("\n" + "="*60)
print(f"Starting {N_SPLITS}-Fold Stratified Cross-Validation")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n>>> [Fold {fold+1}/{N_SPLITS}] Training Models...")

    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    # Preprocessing: StandardScaler fitted only on train fold
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train_fold)
    X_va_sc = scaler.transform(X_val_fold)
    X_te_sc = scaler.transform(X_test)

    # 1. Model 1: Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver='lbfgs')
    lr.fit(X_tr_sc, y_train_fold)
    oof_lr[val_idx] = lr.predict_proba(X_va_sc)[:, 1]
    test_preds_lr += lr.predict_proba(X_te_sc)[:, 1] / N_SPLITS

    # 2. Model 2: Multi-Layer Perceptron (Neural Network)
    mlp_model, mlp_val_probs = train_pytorch_mlp(
        X_tr_sc, y_train_fold, X_va_sc, y_val_fold,
        input_dim=X_tr_sc.shape[1], epochs=30, batch_size=64
    )
    oof_mlp[val_idx] = mlp_val_probs

    with torch.no_grad():
        test_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        test_preds_mlp += mlp_model(test_tensor).cpu().numpy().flatten() / N_SPLITS

    # 3. Model 3: Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=14,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        n_jobs=-1,
        random_state=SEED
    )
    rf.fit(X_train_fold, y_train_fold)
    oof_rf[val_idx] = rf.predict_proba(X_val_fold)[:, 1]
    test_preds_rf += rf.predict_proba(X_test)[:, 1] / N_SPLITS

    fold_lr_acc = accuracy_score(y_val_fold, (oof_lr[val_idx] > 0.5).astype(int))
    fold_mlp_acc = accuracy_score(y_val_fold, (oof_mlp[val_idx] > 0.5).astype(int))
    fold_rf_acc = accuracy_score(y_val_fold, (oof_rf[val_idx] > 0.5).astype(int))
    print(f"Fold {fold+1} Validation Accuracy -> LR: {fold_lr_acc*100:.2f}% | MLP (NN): {fold_mlp_acc*100:.2f}% | Random Forest: {fold_rf_acc*100:.2f}%")

# --------------------------------------------------------------------------------------
# 6. Comprehensive Model Benchmarking
# --------------------------------------------------------------------------------------
models_dict = {
    'Logistic Regression (Baseline)': oof_lr,
    'Neural Network (PyTorch MLP)': oof_mlp,
    'Random Forest Classifier': oof_rf
}

benchmark_rows = []

print("\n" + "="*60)
print("FINAL 5-FOLD OUT-OF-FOLD (OOF) EVALUATION RESULTS")
print("="*60)

for name, probs in models_dict.items():
    binary_preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y, binary_preds)
    roc_auc = roc_auc_score(y, probs)
    f1 = f1_score(y, binary_preds)
    prec = precision_score(y, binary_preds)
    rec = recall_score(y, binary_preds)

    benchmark_rows.append({
        'Model': name,
        'Accuracy (%)': round(acc * 100, 2),
        'ROC-AUC': round(roc_auc, 4),
        'F1-Score': round(f1, 4),
        'Precision (%)': round(prec * 100, 2),
        'Recall (%)': round(rec * 100, 2)
    })

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")

results_df = pd.DataFrame(benchmark_rows).sort_values(by='ROC-AUC', ascending=False)
print("\n" + "="*60)
print("SUMMARY COMPARISON TABLE:")
print("="*60)
print(results_df.to_string(index=False))

# --------------------------------------------------------------------------------------
# 7. Decision Threshold Optimization (On Best Performing Model)
# --------------------------------------------------------------------------------------
print("\n" + "="*60)
print("DECISION THRESHOLD TUNING (ON BEST MODEL: RANDOM FOREST)")
print("="*60)

thresholds = np.arange(0.35, 0.65, 0.01)
best_th = 0.50
best_acc = 0.0
best_f1 = 0.0

for th in thresholds:
    preds = (oof_rf > th).astype(int)
    cur_acc = accuracy_score(y, preds)
    cur_f1 = f1_score(y, preds)
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_th = th
        best_f1 = cur_f1

print(f"Optimal Decision Cutoff: {best_th:.2f}")
print(f"Calibrated Accuracy    : {best_acc*100:.2f}%")
print(f"Calibrated F1-Score    : {best_f1:.4f}")

# --------------------------------------------------------------------------------------
# 8. Visualization: Confusion Matrices, Feature Importances, ROC Curves
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, probs) in enumerate(models_dict.items()):
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(y, preds)
    acc = accuracy_score(y, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{name}\nAcc: {acc*100:.2f}%", fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted Label (0: Non, 1: Smoker)")
    axes[idx].set_ylabel("True Label")

plt.tight_layout()
cm_plot_path = os.path.join(BASE_DIR, "5_confusion_matrices.png")
plt.savefig(cm_plot_path, dpi=200)
plt.close()
print(f"[Info] Saved confusion matrices plot to: {cm_plot_path}")

# Feature importances from Random Forest
rf_final = RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=8, random_state=SEED, n_jobs=-1)
rf_final.fit(X, y)
feat_imp = pd.Series(rf_final.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
feat_imp.head(15).plot(kind='barh', color='teal', edgecolor='black')
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight='bold')
plt.xlabel("Gini Importance Score")
plt.tight_layout()
feat_plot_path = os.path.join(BASE_DIR, "6_feature_importances.png")
plt.savefig(feat_plot_path, dpi=200)
plt.close()
print(f"[Info] Saved feature importances plot to: {feat_plot_path}")

# ROC Curves
plt.figure(figsize=(8, 6))
colors = {'Logistic Regression (Baseline)': 'gray', 'Neural Network (PyTorch MLP)': 'blue', 'Random Forest Classifier': 'green'}
for name, probs in models_dict.items():
    fpr, tpr, _ = roc_curve(y, probs)
    auc_score = roc_auc_score(y, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=colors[name], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5000)')
plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
roc_plot_path = os.path.join(BASE_DIR, "7_roc_curves.png")
plt.savefig(roc_plot_path, dpi=200)
plt.close()
print(f"[Info] Saved ROC curves plot to: {roc_plot_path}")

# --------------------------------------------------------------------------------------
# 9. Generating Final Test Predictions
# --------------------------------------------------------------------------------------
final_binary_preds = (test_preds_rf > best_th).astype(int)

submission = pd.DataFrame({
    'id': test_df.index,
    'prediction': final_binary_preds,
    'probability': np.round(test_preds_rf, 4)
})

submission[['id', 'prediction']].to_csv(PRED_OUT_PATH, index=False)
print(f"\n[Info] Saved test predictions to: {PRED_OUT_PATH}")
print(f"[Info] Test prediction preview:")
print(submission.head(10))
print(f"[Info] Predicted smokers in test set: {(final_binary_preds == 1).sum()} / {len(final_binary_preds)} ({(final_binary_preds == 1).mean()*100:.2f}%)")

print("\n" + "="*60)
print("EXECUTION COMPLETE SUCCESSFULLY!")
print("="*60)


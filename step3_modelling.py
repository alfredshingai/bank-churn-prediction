# ============================================================
# BANK CHURN PROJECT — Step 3: Feature Engineering & ML Modelling
# ============================================================
# Run after step1_data_cleaning.py
# Input : bank_churn_clean.csv
# Output: 3 evaluation charts + printed metrics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve,
)
from sklearn.pipeline import Pipeline

# ── Load clean data ──────────────────────────────────────────
df = pd.read_csv("bank_churn_clean.csv")


# ════════════════════════════════════════════════════════════
# 1. Feature Engineering
# ════════════════════════════════════════════════════════════

# Ratio of account balance to estimated salary
# → captures whether a customer is "parking" a lot of money
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

# Group age into bands — helps tree-based models find non-linear patterns
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 30, 45, 60, 100],
    labels=["18-30", "31-45", "46-60", "60+"],
)

# Flag customers with a zero balance (a possible disengagement signal)
df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["Geography", "Gender", "AgeGroup"], drop_first=False)

# Convert bool columns (pandas get_dummies output) to int
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

print("Dataset shape after feature engineering:", df.shape)


# ════════════════════════════════════════════════════════════
# 2. Train / Test Split
# ════════════════════════════════════════════════════════════

X = df.drop(columns=["Exited"])
y = df["Exited"]

# stratify=y preserves the 80/20 churn ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Churn rate — Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")


# ════════════════════════════════════════════════════════════
# 3. Define Models
# ════════════════════════════════════════════════════════════
# class_weight='balanced' compensates for the 80/20 class imbalance
# without needing to over/undersample the data.

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),   # LR needs features on the same scale
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        random_state=42,
    ),
}


# ════════════════════════════════════════════════════════════
# 4. Train, Evaluate, Cross-Validate
# ════════════════════════════════════════════════════════════

results = {}
print("\n" + "="*60)
for name, model in models.items():
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    # 5-fold stratified CV on the training set (checks for overfitting)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
    )

    results[name] = {
        "model":   model,
        "y_prob":  y_prob,
        "y_pred":  y_pred,
        "auc":     auc,
        "ap":      ap,
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
    }

    print(f"\n── {name} ──")
    print(f"  Test AUC-ROC       : {auc:.4f}")
    print(f"  Avg Precision (AP) : {ap:.4f}")
    print(f"  CV AUC (5-fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

print("="*60)


# ════════════════════════════════════════════════════════════
# 5. Visualisations
# ════════════════════════════════════════════════════════════

BG     = "#F8F9FB"
COLORS = ["#4C9BE8", "#E8A44C", "#5DBE7A"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CDD3DA", "grid.color": "#E2E6EA",
    "axes.titleweight": "bold", "axes.titlesize": 13,
})

# ── Fig 5: ROC | PR curves | AUC comparison ─────────────────
fig5, axes = plt.subplots(1, 3, figsize=(18, 6))
fig5.patch.set_facecolor(BG)
fig5.suptitle("Model Evaluation", fontsize=16, fontweight="bold")

# ROC curves
ax = axes[0]
for (name, res), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} ({res['auc']:.3f})")
ax.plot([0,1],[0,1], "--", color="#AAAAAA", lw=1.5, label="Random (0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(fontsize=9)

# Precision-Recall curves
ax = axes[1]
for (name, res), color in zip(results.items(), COLORS):
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ax.plot(rec, prec, color=color, lw=2.5, label=f"{name} (AP={res['ap']:.3f})")
ax.axhline(y_test.mean(), linestyle="--", color="#AAAAAA", lw=1.5,
           label=f"Baseline ({y_test.mean():.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(fontsize=9)

# Test AUC vs CV AUC bar chart
ax = axes[2]
short_names = ["Log. Reg.", "Rand. Forest", "Grad. Boost"]
aucs = [results[n]["auc"]     for n in results]
cvs  = [results[n]["cv_mean"] for n in results]
x, w = np.arange(3), 0.35
b1 = ax.bar(x - w/2, aucs, w, color=COLORS, alpha=0.9, edgecolor="white", label="Test AUC")
b2 = ax.bar(x + w/2, cvs,  w, color=COLORS, alpha=0.5, edgecolor="white",
            hatch="//", label="CV AUC (5-fold)")
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylim(0.7, 0.92)
ax.set_ylabel("AUC-ROC")
ax.set_title("Test AUC vs Cross-Val AUC")
ax.legend(fontsize=9)
for bar, val in zip(list(b1) + list(b2), aucs + cvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

fig5.tight_layout()
fig5.savefig("model_fig5_evaluation.png", dpi=150, bbox_inches="tight")
print("Saved → model_fig5_evaluation.png")


# ── Fig 6: Confusion Matrices ────────────────────────────────
fig6, axes = plt.subplots(1, 3, figsize=(16, 5))
fig6.patch.set_facecolor(BG)
fig6.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")

for ax, (name, res), color in zip(axes, results.items(), COLORS):
    cm     = confusion_matrix(y_test, res["y_pred"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(rv, rp)]
                       for rv, rp in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=labels, fmt="", ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                linewidths=2, linecolor="white", cbar=False,
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"],
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig6.tight_layout()
fig6.savefig("model_fig6_confusion.png", dpi=150, bbox_inches="tight")
print("Saved → model_fig6_confusion.png")


# ── Fig 7: Feature Importance (Gradient Boosting) ────────────
fig7, ax = plt.subplots(figsize=(10, 7))
fig7.patch.set_facecolor(BG)
ax.set_facecolor(BG)

gb_model    = results["Gradient Boosting"]["model"]
feat_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": gb_model.feature_importances_})
    .sort_values("Importance", ascending=True)
    .tail(15)
)
bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
               color="#5DBE7A", edgecolor="white", linewidth=1)
ax.set_title("Feature Importances — Gradient Boosting (Top 15)")
ax.set_xlabel("Importance Score")
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlim(0, feat_df["Importance"].max() * 1.18)

fig7.tight_layout()
fig7.savefig("model_fig7_feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved → model_fig7_feature_importance.png")

# ============================================================
# BANK CHURN PROJECT — Step 2: EDA & Visualizations
# ============================================================
# Run after step1_data_cleaning.py
# Input : bank_churn_clean.csv
# Output: 4 PNG charts saved to working directory
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load clean data ──────────────────────────────────────────
df = pd.read_csv("bank_churn_clean.csv")

# ── Quick summary ────────────────────────────────────────────
print("Shape:", df.shape)
print(f"\nOverall churn rate: {df['Exited'].mean()*100:.1f}%")
print("\nChurn rate by Geography:")
print(df.groupby("Geography")["Exited"].mean().mul(100).round(1).astype(str) + "%")
print("\nChurn rate by Gender:")
print(df.groupby("Gender")["Exited"].mean().mul(100).round(1).astype(str) + "%")
print("\nChurn rate by NumOfProducts:")
print(df.groupby("NumOfProducts")["Exited"].mean().mul(100).round(1).astype(str) + "%")

# ── Colour palette ───────────────────────────────────────────
STAY  = "#4C9BE8"   # blue  → stayed
CHURN = "#E8604C"   # red   → churned
BG    = "#F8F9FB"
GRID  = "#E2E6EA"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#CDD3DA",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "font.size": 11,
})

palette   = {0: STAY, 1: CHURN}
label_map = {0: "Stayed", 1: "Churned"}


# ════════════════════════════════════════════════════════════
# Figure 1 — Overview (2 × 2)
# ════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Bank Churn — Overview", fontsize=16, fontweight="bold", y=1.01)

# 1a. Churn donut chart
ax = axes[0, 0]
counts = df["Exited"].value_counts().sort_index()
wedges, texts, autotexts = ax.pie(
    counts,
    labels=["Stayed", "Churned"],
    colors=[STAY, CHURN],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    textprops={"fontsize": 12},
)
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title("Overall Churn Distribution")

# 1b. Churn rate by Geography
ax = axes[0, 1]
geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False) * 100
bars = ax.bar(
    geo_churn.index, geo_churn.values,
    color=[CHURN if g == "Germany" else STAY for g in geo_churn.index],
    edgecolor="white", linewidth=1.2, width=0.5,
)
ax.set_title("Churn Rate by Geography")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 45)
for bar, val in zip(bars, geo_churn.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

# 1c. Churn rate by Gender
ax = axes[1, 0]
gen_churn = df.groupby("Gender")["Exited"].mean().sort_values(ascending=False) * 100
bars = ax.bar(
    gen_churn.index, gen_churn.values,
    color=[CHURN, STAY], edgecolor="white", linewidth=1.2, width=0.4,
)
ax.set_title("Churn Rate by Gender")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 35)
for bar, val in zip(bars, gen_churn.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

# 1d. Churn rate by Number of Products
ax = axes[1, 1]
prod_churn = df.groupby("NumOfProducts")["Exited"].mean() * 100
bars = ax.bar(
    prod_churn.index.astype(str), prod_churn.values,
    color=[STAY, STAY, CHURN, CHURN],
    edgecolor="white", linewidth=1.2, width=0.5,
)
ax.set_title("Churn Rate by Number of Products")
ax.set_xlabel("Number of Products")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 115)
for bar, val in zip(bars, prod_churn.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

fig1.tight_layout()
fig1.savefig("eda_fig1_overview.png", dpi=150, bbox_inches="tight")
print("Saved → eda_fig1_overview.png")


# ════════════════════════════════════════════════════════════
# Figure 2 — Numeric Feature Distributions (2 × 3)
# ════════════════════════════════════════════════════════════
numeric_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure"]
fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Numeric Feature Distributions by Churn Status",
              fontsize=16, fontweight="bold")

for i, col in enumerate(numeric_cols):
    ax = axes.flatten()[i]
    for exited, grp in df.groupby("Exited"):
        ax.hist(grp[col], bins=35, alpha=0.65,
                color=palette[exited], label=label_map[exited],
                edgecolor="white", linewidth=0.4)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)

# Bonus: Age box plot in the 6th panel
ax = axes.flatten()[5]
stayed  = df[df["Exited"] == 0]["Age"]
churned = df[df["Exited"] == 1]["Age"]
bp = ax.boxplot([stayed, churned], patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
for patch, color in zip(bp["boxes"], [STAY, CHURN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels(["Stayed", "Churned"])
ax.set_title("Age Distribution (Box Plot)")
ax.set_ylabel("Age")

fig2.tight_layout()
fig2.savefig("eda_fig2_distributions.png", dpi=150, bbox_inches="tight")
print("Saved → eda_fig2_distributions.png")


# ════════════════════════════════════════════════════════════
# Figure 3 — Correlation Heatmap
# ════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(10, 8))
fig3.patch.set_facecolor(BG)
ax.set_facecolor(BG)

corr = df.select_dtypes(include="number").corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

sns.heatmap(
    corr, mask=mask, ax=ax,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    annot=True, fmt=".2f", annot_kws={"size": 10},
    linewidths=0.5, linecolor="white",
    vmin=-1, vmax=1, square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)

fig3.tight_layout()
fig3.savefig("eda_fig3_correlation.png", dpi=150, bbox_inches="tight")
print("Saved → eda_fig3_correlation.png")


# ════════════════════════════════════════════════════════════
# Figure 4 — Active Member & Credit Card Status
# ════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
fig4.patch.set_facecolor(BG)
fig4.suptitle("Churn Rate by Membership & Credit Card Status",
              fontsize=14, fontweight="bold")

configs = [
    ("IsActiveMember", "Active Member Status", ["Inactive", "Active"]),
    ("HasCrCard",      "Has Credit Card",       ["No Card",  "Has Card"]),
]
for ax, (col, title, xticks) in zip(axes, configs):
    churn_rates = df.groupby(col)["Exited"].mean() * 100
    bars = ax.bar(xticks, churn_rates.values,
                  color=[CHURN, STAY], edgecolor="white", linewidth=1.2, width=0.4)
    ax.set_title(title)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 38)
    for bar, val in zip(bars, churn_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)

fig4.tight_layout()
fig4.savefig("eda_fig4_membership.png", dpi=150, bbox_inches="tight")
print("Saved → eda_fig4_membership.png")

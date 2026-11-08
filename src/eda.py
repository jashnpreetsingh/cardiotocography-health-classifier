import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import CLASS_NAMES, FEATURES, REPORT_DIR, TARGET, add_binary_target, load_raw_data

EDA_DIR = REPORT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

df = load_raw_data()
df = add_binary_target(df)

print("shape:", df.shape)
print("missing:", df.isnull().sum().sum())
print("duplicates:", df.duplicated().sum())
print("class balance:")
print(df[TARGET].value_counts())

print("summary stats:")
print(df[FEATURES].describe().T.round(3))

print("skew:")
print(df[FEATURES].skew().round(2))

target_corr = df[FEATURES].corrwith(df[TARGET]).sort_values(ascending=False)
print("corr with target:")
print(target_corr.round(3))

# 1. class balance
plt.figure(figsize=(5, 4))
counts = df[TARGET].value_counts().sort_index()
plt.bar(CLASS_NAMES, counts.values)
plt.title("Class balance")
plt.tight_layout()
plt.savefig(EDA_DIR / "class_balance.png")
plt.close()

# 2. correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df[FEATURES].corr(), cmap="coolwarm", annot=True, fmt=".1f")
plt.title("Feature correlations")
plt.tight_layout()
plt.savefig(EDA_DIR / "correlation_heatmap.png")
plt.close()

# 3. boxplots for top 6 correlated features
top_cols = ["DP", "ALTV", "ASTV", "Mean", "Mode", "Median"]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

sns.boxplot(x=df[TARGET], y=df["DP"], ax=axes[0, 0])
axes[0, 0].set_title("DP")

sns.boxplot(x=df[TARGET], y=df["ALTV"], ax=axes[0, 1])
axes[0, 1].set_title("ALTV")

sns.boxplot(x=df[TARGET], y=df["ASTV"], ax=axes[0, 2])
axes[0, 2].set_title("ASTV")

sns.boxplot(x=df[TARGET], y=df["Mean"], ax=axes[1, 0])
axes[1, 0].set_title("Mean")

sns.boxplot(x=df[TARGET], y=df["Mode"], ax=axes[1, 1])
axes[1, 1].set_title("Mode")

sns.boxplot(x=df[TARGET], y=df["Median"], ax=axes[1, 2])
axes[1, 2].set_title("Median")

plt.tight_layout()
plt.savefig(EDA_DIR / "top_features_by_class.png")
plt.close()

print("saved plots to", EDA_DIR)
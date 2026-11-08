import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (
    CLASS_NAMES, FEATURES, MODEL_DIR, RANDOM_STATE,
    REPORT_DIR, TARGET, TEST_DATA_PATH, TRAIN_DATA_PATH, compute_metrics, df_to_markdown
)

REPORT_DIR.mkdir(exist_ok=True)

train = pd.read_csv(TRAIN_DATA_PATH)
test = pd.read_csv(TEST_DATA_PATH)

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

results = []

def run_model(name, clf, filename, ax):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]
    
    m = compute_metrics(y_test, y_pred, y_score)
    row = {
        "ML Model Name": name,
        "Accuracy": m["Accuracy"],
        "AUC": m["AUC"],
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "F1": m["F1"],
        "MCC": m["MCC"],
    }
    results.append(row)
    
    joblib.dump(pipe, MODEL_DIR / filename)
    
    print("\n---", name, "---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=CLASS_NAMES, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(name)


fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
run_model("Logistic Regression", lr, "logistic_regression.pkl", axes[0])

# 2. Decision Tree
dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
run_model("Decision Tree", dt, "decision_tree.pkl", axes[1])

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=7)
run_model("KNN", knn, "knn.pkl", axes[2])

# 4. Naive Bayes
nb = GaussianNB()
run_model("Naive Bayes", nb, "naive_bayes.pkl", axes[3])

# 5. SVM
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
run_model("Support Vector Machine", svm, "support_vector_machine.pkl", axes[4])

plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrices.png")
plt.close()

table = pd.DataFrame(results)
table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

md_text = df_to_markdown(table)
with open(REPORT_DIR / "model_comparison.md", "w") as f:
    f.write(md_text + "\n")

print("\n", md_text)
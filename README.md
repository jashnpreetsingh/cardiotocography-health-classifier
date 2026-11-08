# Fetal Health Risk Classification

## a. Problem Statement

The goal of this project is to predict whether a fetus is **Normal** or **At risk** using 21 cardiotocogram diagnostic features such as baseline heart rate, accelerations, uterine contractions and variability measurements. Five classification models are trained on this data and deployed in a Streamlit app, where you can upload the test CSV to see predictions and evaluation metrics without retraining the models.

## b. Dataset Description

- **Source:** [Cardiotocography dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography), UCI Machine Learning Repository
- **Rows:** 2,126 records. After dropping 13 duplicate rows, 2,113 rows remain. These are split 80/20 with stratified sampling into **1,690 training rows** and **423 test rows**.
- **Features:** 21 numeric features: LB, AC, FM, UC, DL, DS, DP, ASTV, MSTV, ALTV, MLTV, Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance and Tendency. No missing values.
- **Target:** the original `NSP` column has 3 classes: Normal (1,655), Suspect (295) and Pathologic (176). It was converted into a binary target called `fetal_risk`, where **0 = Normal** and **1 = At risk** (Suspect and Pathologic combined). Around 78% of the data is class 0 and 22% is class 1. The test set has 330 Normal and 93 At-risk rows.
- **Preprocessing:** median imputation for any missing values and `StandardScaler` for feature scaling, both inside a scikit-learn pipeline with each model.

## c. Links

- **GitHub Repository:** https://github.com/jashnpreetsingh/cardiotocography-health-classifier
- **Live Streamlit App:** https://cardiotocography-health-classifier-nbf5ksyvel6e6jdtrjccgh.streamlit.app/

### Repository structure

```
cardiotocography-health-classifier/
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv           # 423-row test split (upload this to the app)
├── data/
│   ├── ctg_raw.csv         # raw dataset from UCI
│   └── train_data.csv      # training split
├── src/
│   ├── common.py           # paths, feature list, target, metric function
│   ├── get_raw_data.py     # downloads the dataset from UCI
│   ├── eda.py              # exploratory data analysis
│   ├── transformation.py   # binary target, duplicate removal, train/test split
│   └── train.py            # trains the 5 models and computes metrics
├── model/                  # saved pipelines for all 5 models (.pkl)
└── files/                  # EDA plots, confusion matrices, metrics table
```

## d. Models Used & Evaluation Metrics Comparison

All five models are trained on the same training split. Each is saved as a pipeline (imputer → scaler → classifier), so the app can predict directly on raw test data.

- **Logistic Regression:** `max_iter=1000`, `class_weight="balanced"`
- **Decision Tree:** `max_depth=6`, `class_weight="balanced"`
- **K-Nearest Neighbors:** `n_neighbors=7`
- **Naive Bayes:** Gaussian
- **Support Vector Machine:** RBF kernel, `class_weight="balanced"`, `probability=True`

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression    | 0.8794   | 0.9579 | 0.6780    | 0.8602 | 0.7583 | 0.6880 |
| Decision Tree          | 0.9220   | 0.9533 | 0.7830    | 0.8925 | 0.8342 | 0.7863 |
| KNN                    | 0.9243   | 0.9759 | 0.8861    | 0.7527 | 0.8140 | 0.7709 |
| Naive Bayes            | 0.8818   | 0.9390 | 0.7471    | 0.6989 | 0.7222 | 0.6478 |
| Support Vector Machine | 0.9196   | 0.9782 | 0.7479    | 0.9570 | 0.8396 | 0.7977 |

## e. Observations on Model Performance

| ML Model Name                         | Observation about model performance                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression                   | Gets good recall (0.8602) because of balanced class weights, and a strong AUC because a few features like ASTV and ALTV separate the classes almost linearly. But precision is the lowest of all models at 0.6780: the simple linear boundary flags too many Normal cases as At risk (38 false alarms).                                                                           |
| Decision Tree                         | Improves precision, F1 and MCC compared to Logistic Regression because the depth-6 tree picks up non-linear splits on features like ASTV, ALTV and DP. It misses only 10 of 93 at-risk cases. Its AUC (0.9533) is a little lower than the other strong models because a tree gives only a few distinct probability values (one per leaf), so it ranks cases more coarsely.        |
| KNN                                   | Has the highest accuracy (0.9243) and precision (0.8861) with k=7, and the fewest false alarms (9). But recall drops to 0.7527, missing 23 at-risk cases, because KNN does not use class weights and the majority Normal class dominates the nearest neighbours.                                                                                                                  |
| Naive Bayes                           | Worst performing model, with the lowest AUC, recall, F1 and MCC. It misses 28 of 93 at-risk cases. Gaussian Naive Bayes assumes all features are independent, which is not true here: the histogram features Mean, Mode, Median, Min and Max are heavily correlated, so the same information is counted several times.                                                            |
| Support Vector Machine                | Best model overall, with the highest AUC, recall, F1 and MCC. It catches 89 of 93 at-risk cases and misses only 4. The RBF kernel on scaled features handles the non-linear boundary, and balanced class weights handle the imbalance. Its accuracy is slightly below KNN and the Decision Tree because it accepts more false alarms (30) in exchange for far fewer missed cases. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** With 78% Normal cases, accuracy alone is misleading, and in fetal health screening missing an at-risk case is much worse than a false alarm. SVM has the highest recall (0.9570), F1 (0.8396), MCC (0.7977) and AUC (0.9782), so it is the most reliable model for this dataset.                                                                      |

## How to Run Locally

```bash
pip install -r requirements.txt
python src/get_raw_data.py      # download the dataset
python src/eda.py               # EDA plots -> files/eda/
python src/transformation.py    # train/test split -> data/, test_data.csv
python src/train.py             # train models -> model/, metrics -> files/
streamlit run app.py
```

The pinned package versions need Python 3.11 (also used for the Streamlit Cloud deployment).

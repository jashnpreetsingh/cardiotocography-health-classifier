# Fetal Health Risk Classification

## a. Problem Statement

The goal of this project is to predict whether a fetus is **Normal** or **At risk** using 21 cardiotocogram (CTG) diagnostic features such as baseline heart rate, accelerations, uterine contractions and variability measurements. Five classification models are trained on this data and deployed in a Streamlit app, where you can upload the test CSV to see predictions and evaluation metrics without retraining the models.

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

| ML Model Name                     | Observation about model performance                                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression               | Good recall (0.86) because of balanced class weights, but the lowest precision (0.68). The linear boundary flags many Normal cases as At risk.                                                             |
| Decision Tree                     | Better precision, F1 and MCC than Logistic Regression, since it picks up non-linear patterns in features like ASTV and ALTV. Its AUC is lower because a tree gives only a few distinct probability values. |
| KNN                               | Highest accuracy (0.92) and precision (0.89), but low recall (0.75). It doesn't use class weights, so the majority Normal class dominates the neighbours.                                                  |
| Naive Bayes                       | Weakest model, with the lowest recall, F1 and MCC. It assumes features are independent, but features like Mean, Mode and Median are highly correlated.                                                     |
| Support Vector Machine            | Best overall, with the highest AUC, recall, F1 and MCC. It misses only 4 of 93 at-risk cases.                                                                                                              |
| Overall, Winner for your dataset? | Support Vector Machine. Missing an at-risk case is worse than a false alarm, and SVM has the best recall (0.96) and MCC (0.80).                                                                            |

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

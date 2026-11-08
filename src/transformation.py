import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    FEATURES, MODEL_DIR, PREPROCESSOR_PATH, RANDOM_STATE, TARGET,
    TEST_DATA_PATH, TEST_SIZE, TRAIN_DATA_PATH, add_binary_target, load_raw_data
)

MODEL_DIR.mkdir(exist_ok=True)

df = load_raw_data()
df = add_binary_target(df)

# drop duplicates before split
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("dropped duplicates:", before - len(df))

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("train size:", len(X_train), "test size:", len(X_test))

train_df = X_train.copy()
train_df[TARGET] = y_train
train_df.to_csv(TRAIN_DATA_PATH, index=False)

test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv(TEST_DATA_PATH, index=False)

preprocessor = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
preprocessor.fit(X_train)
joblib.dump(preprocessor, PREPROCESSOR_PATH)

print("done saving train/test and preprocessor")
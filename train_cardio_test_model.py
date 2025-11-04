# train_cvd.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load cleaned dataset
df = pd.read_csv("cardio_test1.csv")

# 2. Feature engineering
df["age_years"] = (df["age"] / 365.25).round(1)
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

# 3. Features and target
feature_cols = [
    "age_years",
    "gender",        # 1 = female, 2 = male
    "bmi",           # kg/m^2
    "systolic_bp",   # mmHg
    "diastolic_bp",  # mmHg
    "cholesterol",   # mmol/L
    "glucose",       # mmol/L
]
X = df[feature_cols]
y = df["cardio"]

feature_names = X.columns.tolist()
feature_defaults = X.median(numeric_only=True).to_dict()

# 4. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Base learners
base_estimators = [
    ("LR", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])),
    ("SVC", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ])),
    ("KNN", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ])),
    ("RF", RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
    ("DT", DecisionTreeClassifier(
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )),
    ("GNB", GaussianNB()),
    ("MLP", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        )),
    ])),
    ("LDA", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])),
    ("QDA", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", QuadraticDiscriminantAnalysis()),
    ])),
]

stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,
)

# 6. Train & quick evaluation
stacking_clf.fit(X_train, y_train)
y_prob = stacking_clf.predict_proba(X_test)[:, 1]
y_pred = stacking_clf.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test ROC-AUC : {roc_auc_score(y_test, y_prob):.3f}")

# 7. Retrain on full data for deployment
stacking_clf.fit(X, y)

# 8. Save bundle
bundle = {
    "model": stacking_clf,
    "feature_names": feature_names,
    "feature_defaults": feature_defaults,
}
joblib.dump(bundle, "cardio_mmol_stacked_model.pkl")
print("Saved model to cardio_mmol_stacked_model.pkl")

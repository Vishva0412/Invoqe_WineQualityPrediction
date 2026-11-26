
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction System")
st.write(
    "Predict whether a red wine is likely to have a high quality score."
)

# Dataset location
DATA_PATH = os.path.join("data", "winequality-red.csv")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

# Load dataset
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, sep=";")

    return pd.read_csv(DATA_URL, sep=";")


try:
    df = load_data()

except Exception as exc:
    st.error(
        "Dataset could not be loaded. Download the CSV "
        "and save it as data/winequality-red.csv."
    )
    st.exception(exc)
    st.stop()


# Data cleaning
df = df.drop_duplicates().copy()
df = df.dropna()

# Create target variable
df["high_quality"] = (df["quality"] >= 7).astype(int)

# Select features and target
feature_cols = [
    c for c in df.columns
    if c not in ["quality", "high_quality"]
]

X = df[feature_cols]
y = df["high_quality"]


# Dataset overview
st.subheader("Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Rows after cleaning", f"{len(df):,}")
c2.metric("Input Features", len(feature_cols))
c3.metric("High Quality Wines", int(y.sum()))

with st.expander("View Sample Records"):
    st.dataframe(df.head(10), use_container_width=True)


# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="quality",
        ax=ax
    )

    ax.set_title("Original Wine Quality Scores")

    st.pyplot(fig)
    plt.close(fig)


with right:
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        df[feature_cols + ["quality"]].corr(),
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Feature Correlation")

    st.pyplot(fig)
    plt.close(fig)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Machine Learning Models
models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2
        ))
    ])

}


# Model training and evaluation
results = []
trained = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    trained[name] = model

    results.append({

        "Model": name,

        "Accuracy": accuracy_score(y_test, pred),

        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),

        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),

        "F1-score": f1_score(
            y_test, pred, zero_division=0
        )

    })


results_df = pd.DataFrame(results).sort_values(
    "F1-score",
    ascending=False
)

st.subheader("Model Comparison")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-score": "{:.3f}"
    }),
    use_container_width=True
)


# Select best model
best_name = results_df.iloc[0]["Model"]

best_model = trained[best_name]

test_pred = best_model.predict(X_test)


# Classification report
col_a, col_b = st.columns(2)

with col_a:

    st.write(f"Selected Model: {best_name}")

    st.text(
        classification_report(
            y_test,
            test_pred,
            target_names=["Not high", "High"],
            zero_division=0
        )
    )


# Confusion matrix
with col_b:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(y_test, test_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not high", "High"],
        yticklabels=["Not high", "High"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)


# Prediction interface
st.subheader("Try a Prediction")

st.caption(
    "Enter plausible measurements to predict wine quality."
)

input_values = {}

cols = st.columns(3)

for i, feature in enumerate(feature_cols):

    default = float(df[feature].median())

    with cols[i % 3]:

        input_values[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=default,
            format="%.4f"
        )


# Predict button
if st.button("Predict Wine Quality", type="primary"):

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_cols
    )

    prediction = int(best_model.predict(input_df)[0])

    if prediction == 1:

        st.success(
            "Prediction: High Quality (estimated score >= 7)"
        )

    else:

        st.info(
            "Prediction: Not High Quality (estimated score < 7)"
        )

    if hasattr(best_model, "predict_proba"):

        probability = float(
            best_model.predict_proba(input_df)[0][1]
        )

        st.write(
            f"Estimated probability of high quality: {probability:.1%}"
        )


st.caption(
    "Educational ML project. Predictions are estimates."
)

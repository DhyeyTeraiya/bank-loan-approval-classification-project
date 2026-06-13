# Google Colab Code - Bank Loan Approval Prediction
# Copy this code into Google Colab.
# It is written step by step with simple comments.


# Cell 1: Import Libraries

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Cell 2: Step 1 - Loading Dataset From Google Sheet

sheet_url = "https://docs.google.com/spreadsheets/d/1nRJhACNzEu6y5PckkWcgSAiTBB24KFAc5Rg8Bo9kAzc/gviz/tq?tqx=out:csv&sheet=Sheet1"

df = pd.read_csv(sheet_url)

print("Dataset loaded successfully")
df.head()


# Cell 3: Step 2 - Explore Dataset

print("Dataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nFirst 5 Rows:")
display(df.head())

print("Dataset Information:")
df.info()

print("Checking Missing Values:")
print(df.isnull().sum())

print("Statistical Summary:")
display(df.describe())


# Cell 4: Data Cleaning

print("Duplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Cell 5: Step 3 - EDA Target Variable Distribution

target_column = "loan_approved"

print("Target Variable Count:")
print(df[target_column].value_counts())

print("\nTarget Variable Percentage:")
print(df[target_column].value_counts(normalize=True) * 100)

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=target_column, palette="Set2")
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Approved: 0 = Rejected, 1 = Approved")
plt.ylabel("Number of Applications")
plt.show()


# Cell 6: Step 3 - Correlation Heatmap

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Cell 7: Step 3 - Credit Score vs Loan Approval

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="loan_approved", y="credit_score", palette="Set3")
plt.title("Credit Score vs Loan Approval")
plt.xlabel("Loan Approved: 0 = Rejected, 1 = Approved")
plt.ylabel("Credit Score")
plt.show()


# Cell 8: EDA Relationship Analysis

numeric_features = (
    df.select_dtypes(include=["int64", "float64"])
    .drop(columns=[target_column], errors="ignore")
    .columns
)

numeric_relationship = (
    df[list(numeric_features) + [target_column]]
    .corr()[target_column]
    .drop(target_column)
    .sort_values(key=abs, ascending=False)
)

print("Numeric Feature Relationship With Loan Approval:")
print(numeric_relationship)

plt.figure(figsize=(9, 5))
numeric_relationship.sort_values().plot(kind="barh", color="#2f6f8f")
plt.title("Numeric Feature Relationship with Loan Approval")
plt.xlabel("Correlation with Loan Approval")
plt.show()

categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
categorical_features = [col for col in categorical_features if col != "customer_id"]

for column in categorical_features:
    approval_rate = (
        df.groupby(column)[target_column]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    print(f"\nApproval Rate by {column}:")
    display(approval_rate)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=approval_rate, x=column, y=target_column, palette="viridis")
    plt.title(f"Loan Approval Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Approval Rate")
    plt.xticks(rotation=35, ha="right")
    plt.show()


# Cell 9: Step 4 - Feature Engineering

df["loan_to_income_ratio"] = df["loan_amount_lakh"] / df["annual_income_lakh"]

df["financial_stability_score"] = (
    df["credit_score"] / 100
    + df["years_employed"]
    + df["savings_balance_lakh"]
)

df["debt_burden_score"] = (
    df["debt_to_income_ratio"]
    + df["existing_loans"] * 5
    + df["missed_payments"] * 3
)

print("New Features Created:")
print("1. loan_to_income_ratio")
print("2. financial_stability_score")
print("3. debt_burden_score")

display(df[[
    "loan_amount_lakh",
    "annual_income_lakh",
    "loan_to_income_ratio",
    "financial_stability_score",
    "debt_burden_score",
]].head())


# Cell 10: Separate Features and Target

X = df.drop(columns=["loan_approved", "customer_id"], errors="ignore")
y = df["loan_approved"]

print("Input Features Shape:", X.shape)
print("Target Shape:", y.shape)


# Cell 11: Step 5 - Creating Preprocessing Pipeline

numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

print("Preprocessing pipeline created successfully")


# Cell 12: Step 6 - Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Cell 13: Common Function To Evaluate Model

def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print(model_name)
    print("=" * 60)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


# Cell 14: Train Logistic Regression

logistic_model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

logistic_model.fit(X_train, y_train)

logistic_result = evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_test,
    y_test,
)


# Cell 15: Train KNN Classifier

knn_model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", KNeighborsClassifier(n_neighbors=5)),
])

knn_model.fit(X_train, y_train)

knn_result = evaluate_model(
    "KNN Classifier",
    knn_model,
    X_test,
    y_test,
)


# Cell 16: Train Naive Bayes

naive_bayes_model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", GaussianNB()),
])

naive_bayes_model.fit(X_train, y_train)

naive_bayes_result = evaluate_model(
    "Naive Bayes",
    naive_bayes_model,
    X_test,
    y_test,
)


# Cell 17: Train Decision Tree

decision_tree_model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42)),
])

decision_tree_model.fit(X_train, y_train)

decision_tree_result = evaluate_model(
    "Decision Tree",
    decision_tree_model,
    X_test,
    y_test,
)


# Cell 18: Compare Models

model_results = [
    logistic_result,
    knn_result,
    naive_bayes_result,
    decision_tree_result,
]

comparison_df = pd.DataFrame(model_results)
comparison_df = comparison_df.sort_values(by="F1 Score", ascending=False)

display(comparison_df)

plt.figure(figsize=(9, 5))
sns.barplot(data=comparison_df, x="Model", y="F1 Score", palette="Set2")
plt.title("Model Comparison by F1 Score")
plt.xticks(rotation=20)
plt.show()


# Cell 19: Final Conclusion

best_model = comparison_df.iloc[0]

print("Best Model:")
print(best_model["Model"])

print("\nBest F1 Score:")
print(round(best_model["F1 Score"], 4))

print("\nConclusion:")
print("The best model is selected based on F1 Score because it balances Precision and Recall.")

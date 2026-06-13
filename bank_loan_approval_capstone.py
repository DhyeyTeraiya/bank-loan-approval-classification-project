"""
Bank Loan Approval Prediction - Classification Capstone Project

Goal:
Predict whether a bank loan application will be approved:
    1 = Approved
    0 = Rejected

This file is written step-by-step so it is easy to understand and present.
Run:
    python bank_loan_approval_capstone.py
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "bank_loan_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET_COLUMN = "loan_approved"


def print_section(title):
    """Print a clean section title in the terminal."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_dataset():
    """Step 1: Load the dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Download the Google Sheet as CSV and "
            f"save it as {DATA_FILE.name} in this folder."
        )

    df = pd.read_csv(DATA_FILE)
    return df


def explore_dataset(df):
    """Step 2: Basic dataset exploration."""
    print_section("Step 2: Explore Dataset")

    print("\nDataset Shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nDataset Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nChecking Missing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistical Summary:")
    print(df.describe())


def clean_dataset(df):
    """
    Data Cleaning:
    - Remove duplicate rows
    - Keep missing value handling inside the preprocessing pipeline

    In this dataset, missing values are already 0, but the pipeline still handles
    missing values safely for real project practice.
    """
    print_section("Data Cleaning")
    cleaned_df = df.copy()

    duplicate_count = cleaned_df.duplicated().sum()
    print(f"\nDuplicate Rows Found: {duplicate_count}")

    cleaned_df = cleaned_df.drop_duplicates()
    print(f"Rows After Removing Duplicates: {cleaned_df.shape[0]}")

    print("\nMissing Values After Cleaning:")
    print(cleaned_df.isnull().sum())

    return cleaned_df


def create_eda_charts(df):
    """
    Step 3: Exploratory Data Analysis.

    Required EDA:
    - Target Variable Distribution
    - Correlation Heatmap
    - Credit Score vs Loan Approval
    """
    print_section("Step 3: Exploratory Data Analysis (EDA)")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nTarget Variable Distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print("\nTarget Variable Percentage:")
    print(df[TARGET_COLUMN].value_counts(normalize=True) * 100)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, palette="Set2")
    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan Approved: 0 = Rejected, 1 = Approved")
    plt.ylabel("Number of Applications")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
    plt.close()

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="credit_score", palette="Set3")
    plt.title("Credit Score vs Loan Approval")
    plt.xlabel("Loan Approved: 0 = Rejected, 1 = Approved")
    plt.ylabel("Credit Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "credit_score_vs_loan_approval.png", dpi=160)
    plt.close()

    print(f"\nEDA charts saved inside: {OUTPUT_DIR.resolve()}")


def analyze_feature_relationships(df):
    """
    Relationship Analysis inside EDA.

    In every data science project, we should ask:
    - Which columns have a strong relationship with the target?
    - Which groups have higher or lower approval rates?
    - Which financial variables look important?
    """
    print_section("EDA Relationship Analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    numeric_features = (
        df.select_dtypes(include=["int64", "float64"])
        .drop(columns=[TARGET_COLUMN], errors="ignore")
        .columns
        .tolist()
    )
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_features = [col for col in categorical_features if col != "customer_id"]

    print("\nRelationship of Numeric Features with Loan Approval:")
    numeric_relationship = (
        df[numeric_features + [TARGET_COLUMN]]
        .corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=abs, ascending=False)
    )
    print(numeric_relationship)
    numeric_relationship.to_csv(OUTPUT_DIR / "numeric_feature_relationships.csv")

    plt.figure(figsize=(9, 5))
    numeric_relationship.sort_values().plot(kind="barh", color="#2f6f8f")
    plt.title("Numeric Feature Relationship with Loan Approval")
    plt.xlabel("Correlation with Loan Approval")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_feature_relationships.png", dpi=160)
    plt.close()

    print("\nApproval Rate by Categorical Features:")
    categorical_summary_tables = []
    for column in categorical_features:
        approval_rate = (
            df.groupby(column)[TARGET_COLUMN]
            .mean()
            .sort_values(ascending=False)
            .rename("approval_rate")
            .reset_index()
        )
        approval_rate["feature"] = column
        categorical_summary_tables.append(approval_rate)

        print(f"\n{column}:")
        print(approval_rate[[column, "approval_rate"]])

        plt.figure(figsize=(8, 4))
        sns.barplot(data=approval_rate, x=column, y="approval_rate", palette="viridis")
        plt.title(f"Loan Approval Rate by {column}")
        plt.xlabel(column)
        plt.ylabel("Approval Rate")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        file_name = f"approval_rate_by_{column}.png"
        plt.savefig(OUTPUT_DIR / file_name, dpi=160)
        plt.close()

    if categorical_summary_tables:
        pd.concat(categorical_summary_tables, ignore_index=True).to_csv(
            OUTPUT_DIR / "categorical_feature_relationships.csv", index=False
        )

    print("\nSimple Data Science Interpretation:")
    strongest_numeric = numeric_relationship.index[0]
    strongest_value = numeric_relationship.iloc[0]
    print(
        f"- Strongest numeric relationship: {strongest_numeric} "
        f"({strongest_value:.4f})"
    )
    print("- Positive correlation means approval tends to increase as the value increases.")
    print("- Negative correlation means approval tends to decrease as the value increases.")
    print("- Categorical approval-rate charts show which groups are approved more often.")


def add_feature_engineering(df):
    """Step 4: Create new useful financial features."""
    print_section("Step 4: Feature Engineering")
    engineered_df = df.copy()

    # Loan amount compared with annual income. Lower value is usually safer.
    engineered_df["loan_to_income_ratio"] = (
        engineered_df["loan_amount_lakh"] / engineered_df["annual_income_lakh"]
    )

    # A simple positive score combining credit score, work experience, and savings.
    engineered_df["financial_stability_score"] = (
        engineered_df["credit_score"] / 100
        + engineered_df["years_employed"]
        + engineered_df["savings_balance_lakh"]
    )

    # A simple risk score combining existing debt pressure and missed payments.
    engineered_df["debt_burden_score"] = (
        engineered_df["debt_to_income_ratio"]
        + engineered_df["existing_loans"] * 5
        + engineered_df["missed_payments"] * 3
    )

    print("\nNew Features Created:")
    print("- loan_to_income_ratio")
    print("- financial_stability_score")
    print("- debt_burden_score")

    print("\nSample of New Features:")
    print(
        engineered_df[
            [
                "loan_amount_lakh",
                "annual_income_lakh",
                "loan_to_income_ratio",
                "financial_stability_score",
                "debt_burden_score",
            ]
        ].head()
    )

    return engineered_df


def create_preprocessing_pipeline(X):
    """
    Step 5: Create preprocessing pipeline.

    The pipeline will:
    - Fill missing numerical values with the median
    - Scale numerical values
    - Fill missing categorical values with the most common value
    - Convert categorical values into numeric columns using one-hot encoding
    """
    print_section("Step 5: Creating Preprocessing Pipeline")

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumerical Features:")
    print(numeric_features)

    print("\nCategorical Features:")
    print(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def create_features_and_target(df):
    """Separate input features X and target variable y."""
    X = df.drop(columns=[TARGET_COLUMN, "customer_id"], errors="ignore")
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test_data(X, y):
    """Step 6: Train Test Split."""
    print_section("Step 6: Train Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining Data Shape:")
    print(X_train.shape)

    print("\nTesting Data Shape:")
    print(X_test.shape)

    return X_train, X_test, y_train, y_test


def evaluate_model(model_name, model, X_test, y_test):
    """
    Create a common function to evaluate every model.

    Evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    """
    y_pred = model.predict(X_test)

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }

    print_section(f"Model Evaluation: {model_name}")
    for metric, value in results.items():
        if metric == "Model":
            continue
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nConfusion Matrix Meaning:")
    print("[[True Rejected, False Approved],")
    print(" [False Rejected, True Approved]]")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    file_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(OUTPUT_DIR / file_name, dpi=160)
    plt.close()

    return results


def train_models(preprocessor, X_train, X_test, y_train, y_test):
    """
    Train classification models:
    - Logistic Regression
    - KNN Classifier
    - Naive Bayes
    - Decision Tree
    """
    print_section("Train Classification Models")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    all_results = []

    for model_name, classifier in models.items():
        model_pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", classifier),
            ]
        )

        model_pipeline.fit(X_train, y_train)
        result = evaluate_model(model_name, model_pipeline, X_test, y_test)
        all_results.append(result)

    return all_results


def compare_models(all_results):
    """Compare models using Accuracy, Precision, Recall, and F1 Score."""
    print_section("Model Comparison")

    comparison_df = pd.DataFrame(all_results).sort_values(
        by="F1 Score", ascending=False
    )

    print(comparison_df.to_string(index=False))

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison saved to: {comparison_path.resolve()}")

    best_model = comparison_df.iloc[0]
    print("\nBest Model Based on F1 Score:")
    print(f"{best_model['Model']} with F1 Score = {best_model['F1 Score']:.4f}")


def main():
    print_section("Bank Loan Approval Prediction Project")
    print("Project Objective:")
    print("Predict whether a customer's loan application will be approved or rejected.")
    print("Approved = 1")
    print("Rejected = 0")

    print("Step 1: Loading Dataset")
    df = load_dataset()

    explore_dataset(df)
    cleaned_df = clean_dataset(df)

    create_eda_charts(cleaned_df)
    analyze_feature_relationships(cleaned_df)

    engineered_df = add_feature_engineering(cleaned_df)

    X, y = create_features_and_target(engineered_df)
    preprocessor = create_preprocessing_pipeline(X)

    X_train, X_test, y_train, y_test = split_train_test_data(X, y)
    all_results = train_models(preprocessor, X_train, X_test, y_train, y_test)
    compare_models(all_results)

    print_section("Project Completed")
    print("All steps are complete.")
    print("Check the outputs folder for charts, confusion matrices, and model comparison.")


if __name__ == "__main__":
    main()

# Bank Loan Approval Prediction - Classification Capstone Project

## Project Objective

The objective of this project is to build a machine learning classification model
that predicts whether a customer's bank loan application will be:

- `1` = Approved
- `0` = Rejected

The prediction is made using customer financial and demographic information such
as income, credit score, employment type, loan amount, missed payments, savings,
and debt-to-income ratio.

## Dataset

The dataset was downloaded from the provided Google Sheet and saved as:

```text
bank_loan_data.csv
```

The dataset contains:

- 22,000 rows
- 18 columns
- Target column: `loan_approved`

## Project Files

```text
bank_loan_approval_capstone.py
```

Main Python file. It contains the full step-by-step machine learning project.

```text
bank_loan_data.csv
```

Dataset used for training and testing the models.

```text
requirements.txt
```

Python libraries required for the project.

```text
outputs/
```

Folder where charts, confusion matrices, and model comparison results are saved.

## How To Run The Project

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
python bank_loan_approval_capstone.py
```

If `python` does not work, try:

```bash
py bank_loan_approval_capstone.py
```

## Libraries Used

The project uses:

- `pandas` for loading and working with data
- `matplotlib` for creating charts
- `seaborn` for better data visualization
- `scikit-learn` for preprocessing, model training, and evaluation

## Complete Project Steps

### Step 1: Loading Dataset

The Python code loads the CSV file using pandas:

```python
df = pd.read_csv(DATA_FILE)
```

This creates a dataframe named `df`, which stores the full dataset.

### Step 2: Explore Dataset

The code checks the basic details of the dataset.

It shows:

- Dataset shape
- Column names
- First 5 rows
- Dataset information
- Missing values
- Statistical summary

This step helps us understand what type of data we have before building a model.

### Data Cleaning

The code checks for duplicate rows and removes them if found.

It also checks missing values again after cleaning.

In this dataset:

- Duplicate rows found: `0`
- Missing values found: `0`

Even though there are no missing values, the preprocessing pipeline still handles
missing values. This is good practice for real data science projects.

### Step 3: Exploratory Data Analysis

EDA helps us understand patterns in the data.

The code creates:

- Target variable distribution chart
- Correlation heatmap
- Credit score vs loan approval chart

The generated chart files are saved in the `outputs/` folder.

### Relationship Analysis

In every data science project, we should find relationships between input
features and the target variable.

This project checks:

- Which numeric columns are related to loan approval
- Which categorical groups have higher approval rates
- Which features may be important for prediction

Important finding:

```text
debt_to_income_ratio
```

has the strongest relationship with loan approval.

It has a negative relationship, which means:

```text
When debt-to-income ratio increases, loan approval chance usually decreases.
```

### Step 4: Feature Engineering

Feature engineering means creating new useful columns from existing columns.

The code creates 3 new features.

#### 1. Loan to Income Ratio

```python
loan_to_income_ratio = loan_amount_lakh / annual_income_lakh
```

This shows how large the loan amount is compared with the customer's income.

#### 2. Financial Stability Score

```python
financial_stability_score = credit_score / 100 + years_employed + savings_balance_lakh
```

This combines credit score, employment experience, and savings.

A higher score means the customer may be financially more stable.

#### 3. Debt Burden Score

```python
debt_burden_score = debt_to_income_ratio + existing_loans * 5 + missed_payments * 3
```

This combines debt ratio, existing loans, and missed payments.

A higher score means the customer may have more financial risk.

### Step 5: Creating Preprocessing Pipeline

Machine learning models need clean numeric input.

The preprocessing pipeline does 3 important things:

- Handles missing values
- Encodes categorical data
- Scales numerical data

#### Numeric Data

For numeric columns:

- Missing values are filled using median
- Values are scaled using `StandardScaler`

#### Categorical Data

For categorical columns:

- Missing values are filled using the most frequent value
- Categories are converted into numbers using `OneHotEncoder`

Examples of categorical columns:

- `gender`
- `region`
- `marital_status`
- `education_level`
- `employment_type`
- `loan_purpose`

### Step 6: Train Test Split

The dataset is split into two parts:

- Training data: used to train the model
- Testing data: used to check model performance

The project uses:

```text
80% training data
20% testing data
```

The split is done using:

```python
train_test_split()
```

### Common Model Evaluation Function

The project uses one common function to evaluate every model.

This avoids repeating the same evaluation code again and again.

The function calculates:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Models Used

The project trains and compares 4 classification models.

### 1. Logistic Regression

Logistic Regression is a simple and powerful classification algorithm.

It works well when the relationship between features and target is mostly linear.

### 2. KNN Classifier

KNN means K-Nearest Neighbors.

It predicts based on nearby similar data points.

### 3. Naive Bayes

Naive Bayes is based on probability.

It is fast and works well for many classification problems.

### 4. Decision Tree

Decision Tree makes decisions using if-else style rules.

It is easy to understand and interpret.

## Evaluation Metrics Explained

### Accuracy

Accuracy tells how many total predictions were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Precision tells how many predicted approved applications were actually approved.

High precision means fewer false approvals.

### Recall

Recall tells how many actual approved applications the model correctly found.

High recall means fewer approved customers are missed.

### F1 Score

F1 Score balances precision and recall.

It is useful when the dataset is not perfectly balanced.

### Confusion Matrix

The confusion matrix shows correct and wrong predictions in table form.

```text
[[True Rejected, False Approved],
 [False Rejected, True Approved]]
```

## Model Comparison Result

After running the project, the best model was:

```text
Logistic Regression
```

Best model score:

```text
F1 Score = 0.9951
```

This means Logistic Regression performed best overall on this dataset.

## Output Files Created

After running the Python file, these outputs are created inside the `outputs/`
folder:

- `target_distribution.png`
- `correlation_heatmap.png`
- `credit_score_vs_loan_approval.png`
- `numeric_feature_relationships.png`
- `numeric_feature_relationships.csv`
- `categorical_feature_relationships.csv`
- Confusion matrix image for each model
- `model_comparison.csv`

## Final Conclusion

This project successfully builds a complete classification machine learning
system for bank loan approval prediction.

The project includes:

- Data cleaning
- EDA
- Relationship analysis
- Encoding
- Feature engineering
- Feature scaling
- Model training
- Model evaluation
- Model comparison

The final selected model is Logistic Regression because it achieved the highest
F1 Score.

# Titanic Survival Prediction Using Support Vector Classifier

## Project Overview

This project explores the Titanic passenger dataset to understand the factors associated with passenger survival and develop a machine learning classification model to predict survival outcomes.

The project follows an end-to-end introductory machine learning workflow, covering data exploration, preprocessing, feature selection, categorical encoding, feature scaling, model development, performance evaluation, and data visualization.

A Support Vector Classifier (SVC) is implemented as the baseline predictive model. The project also examines survival patterns across passenger demographics and ticket classes.

## Project Objectives

- Explore and understand the Titanic passenger dataset.
- Identify patterns and relationships between passenger characteristics and survival.
- Handle missing values and prepare data for machine learning.
- Select relevant features and transform categorical variables.
- Develop an SVC classification model.
- Evaluate predictive performance using standard classification metrics.
- Visualize important patterns in passenger survival.

## Dataset Overview

The project uses the Titanic passenger dataset (`titanic.csv`).

| Dataset Information | Details |
|---|---|
| Total records | 891 |
| Original features | 12 |
| Target variable | Survived |
| Training samples | 712 |
| Testing samples | 179 |
| Train-test ratio | 80:20 |

**Target Variable:**
- `0` – Did not survive
- `1` – Survived

### Selected Features

The following seven original features were selected for model development:

| Feature | Description |
|---|---|
| Pclass | Passenger ticket class |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Passenger ticket fare |
| Embarked | Port of embarkation |

The target variable `Survived` was used to train and evaluate the classifier.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Methodology

### 1. Data Loading and Exploration

- Loaded the Titanic dataset using Pandas.
- Examined dataset dimensions, structure, and data types.
- Generated descriptive statistics.
- Explored unique values and frequency distributions of categorical features.
- Examined passenger distributions across gender and ticket classes.

### 2. Data Preprocessing

- Identified missing values across dataset features.
- Removed the `Cabin` column due to its high proportion of missing values.
- Handled missing values in `Age` using median imputation based on the training set (performed after the train-test split to avoid data leakage).
- Handled missing values in `Embarked` using mode imputation (calculated on the full dataset; only 2 values were missing, so the impact is negligible).
- Inspected the dataset after preprocessing.

### 3. Feature Selection and Encoding

Selected relevant passenger characteristics for survival prediction.

- Converted gender into numerical values (`male = 0`, `female = 1`).
- Applied one-hot encoding to the `Embarked` feature.
- Separated independent features from the target variable.

### 4. Data Splitting and Feature Scaling

- Divided the dataset into training and testing subsets using an 80:20 split.
- Used `random_state=42` to ensure reproducibility.
- Standardized numerical features using Scikit-learn's `StandardScaler`.
- Fitted the scaler on training data and applied the transformation to testing data.

### 5. Model Development

Implemented a Support Vector Classifier (SVC) using Scikit-learn.

The model was trained on the prepared training dataset and used to generate predictions on unseen test samples.

### 6. Model Evaluation

Evaluated model performance using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### 7. Exploratory Data Visualization

Created visualizations to investigate passenger characteristics and survival patterns, including:

- Survival rates across passenger classes and gender.
- Passenger fare distribution using histograms.
- Fare distribution and outliers using box plots.
- Survival counts by gender.
- Survival rates by gender.

## Model Performance

The Support Vector Classifier achieved the following results on the held-out test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 82.12% |
| Precision (Survived) | 83.87% |
| Recall (Survived) | 70.27% |
| F1-score (Survived) | 76.47% |

### Confusion Matrix

| Actual / Predicted | Did Not Survive (0) | Survived (1) |
|---|---:|---:|
| Did Not Survive (0) | 95 | 10 |
| Survived (1) | 22 | 52 |

### Performance Interpretation

- The model correctly classified 147 out of 179 test samples.
- It achieved 83.87% precision for the survived class.
- Its recall of 70.27% indicates that it identified approximately 70% of actual survivors.
- The F1-score of 76.47% reflects the balance between precision and recall for the survived class.

These results provide a baseline for further experimentation and model improvement.

## Key Observations

- The dataset contains 577 male and 314 female passengers.
- Third-class passengers form the largest passenger group, with 491 records.
- The overall survival rate in the dataset is approximately 38.38%.
- Female passengers had a higher observed survival rate (74.20%) than male passengers (18.89%).
- Survival rates were highest in 1st class and lowest in 3rd class, for both men and women.
- The fare distribution is positively skewed, with a skewness value of approximately 4.79.

## Future Improvements

The current implementation establishes a baseline for Titanic survival prediction. Further work will focus on improving model evaluation, feature representation, and predictive performance.

1. **Comparative Model Evaluation:** Implement and compare multiple classification algorithms, including Logistic Regression, Decision Tree, Random Forest, and Support Vector Classifier, to investigate differences in predictive performance.

2. **Cross-Validation:** Introduce K-Fold Stratified Cross-Validation to obtain more reliable performance estimates and assess model stability across different data splits.

3. **Feature Engineering:** Extract passenger titles (Mr., Mrs., Miss., etc.) from the `Name` feature and create a `FamilySize` feature using `SibSp` and `Parch` to capture additional passenger information.

4. **Hyperparameter Optimization:** Apply `GridSearchCV` to explore suitable model parameters and improve classification performance through systematic tuning.

5. **Pipeline-Based Preprocessing:** Develop an integrated Scikit-learn `Pipeline` and `ColumnTransformer` to combine missing-value imputation, categorical encoding, feature scaling, and model training while preventing data leakage during cross-validation.

## How to Run the Project

**1. Clone the repository**

```bash
 git clone https://github.com/areebamubarik10-spec/Titanic-Survival-Prediction.git
 cd Titanic-Survival-Prediction
```

**2. Install the required libraries**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Ensure the dataset is available**

Place `titanic.csv` in the same directory as the Jupyter Notebook.

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open `titanic.ipynb` and run the cells sequentially.

## Project Status

**Completed – Baseline Implementation**

The current project demonstrates data exploration, preprocessing, feature selection, SVC model development, performance evaluation, and visualization. Comparative modeling, cross-validation, advanced feature engineering, hyperparameter tuning, and integrated pipeline development are identified as future extensions.

---

*This project is part of an ongoing effort to strengthen practical skills in data analysis, predictive modeling, and machine learning.*

# E-commerce Customer Spending Prediction

## Overview
This project involves predicting the yearly amount spent by customers on an e-commerce platform. Using a dataset containing various customer attributes, we aim to build a machine learning model that accurately forecasts how much a customer will spend annually.

The primary objective of this project is to use the given dataset, perform exploratory data analysis (EDA), handle data quality issues, and build a predictive model using **Linear Regression**. This model will help businesses predict customer spending, identify high-value customers, and make data-driven decisions.

---

## Dataset

The dataset consists of the following features for each customer:

| Column                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Email**             | Unique identifier for each user (email address)                            |
| **Address**           | Residential address of the customer                                         |
| **Avatar**            | Profile picture representation (categorical feature)                       |
| **Avg. Session Length** | Average time spent with a stylist in a clothing store (in minutes)           |
| **Time on App**       | Time spent by the user on the mobile/desktop app (in minutes)               |
| **Time on Website**   | Time spent on the website before purchasing (in minutes)                    |
| **Length of Membership** | Duration of the membership with the clothing store (in years)              |
| **Yearly Amount Spent** | Target variable: Annual spending on the clothing store (in dollars)         |

### Sample Data (Top 3 Rows)

| Email                           | Address                                                           | Avatar      | Avg. Session Length | Time on App | Time on Website | Length of Membership | Yearly Amount Spent |
|---------------------------------|-------------------------------------------------------------------|-------------|----------------------|-------------|------------------|----------------------|---------------------|
| mstephenson@fernandez.com        | 835 Frank Tunnel\nWrightmouth, MI 82180-9605                      | Violet      | 34.497268            | 12.655651   | 39.577668        | 4.082621             | 587.951054          |
| hduke@hotmail.com                | 4547 Archer Common\nDiazchester, CA 06566-8576                    | DarkGreen   | 31.926272            | 11.109461   | 37.268959        | 2.664034             | 392.204933          |
| pallen@yahoo.com                | 24645 Valerie Unions Suite 582\nCobbborough, D...                 | Bisque      | 33.000915            | 11.330278   | 37.110597        | 4.104543             | 487.547505          |

---

## Approach: 
1. Importing Necessary Libraries && Loading the Dataset
2. Understanding the Dataset
3. Exploratory Data Analysis (EDA) --> [Shape and Basic Information, Statistical Summary]
4. Feature Selection --> Through domain knowledge, we identified which features are relevant for predicting the target variable, Yearly Amount Spent. Features like Email, Address, and Avatar were deemed irrelevant and dropped from the dataset.
5. Data Preprocessing --> [Missing Values and Duplicates, Renaming Columns]
6. Exploratory Data Analysis (Continued) --> [Univariate Analysis, Bivariate Analysis, Multivariate Analysis, Linear Regression Analysis]
7. Model Building and Evaluation

## Deep flow of Approach: 
### 1. Importing Necessary Libraries
We began by importing essential Python libraries for data manipulation, visualization, and machine learning. These libraries include:

NumPy and Pandas for handling and analyzing the data.

Matplotlib and Seaborn for visualizations and exploratory data analysis (EDA).

Scikit-learn for building and evaluating the predictive model.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Loading the Dataset
The dataset, named Ecommerce Customers.csv, was loaded using Pandas into a DataFrame. The first few rows were examined to get a sense of the structure of the data.

```
df = pd.read_csv('Ecommerce Customers.csv')
df.head(3)
```

### 3. Understanding the Dataset
The dataset contains the following columns:

Email: Unique identifier for each user.

Address: The user’s address.

Avatar: Profile picture of the user.

Avg. Session Length: Time the customer spent interacting with a stylist in the store.

Time on App: Time spent on the mobile or desktop app before placing an order.

Time on Website: Time spent on the website before making a purchase.

Length of Membership: Duration of membership at the store.

Yearly Amount Spent: Total annual spending by the customer (target variable).

### 4. Exploratory Data Analysis (EDA)
Shape and Basic Information: We examined the shape and column data types of the dataset to understand its structure.

Statistical Summary: A summary of the numerical features helped in identifying data distribution, such as the average, minimum, and maximum values.

```
df.shape
df.info()
df.describe()
```

### 5. Feature Selection
Through domain knowledge, we identified which features are relevant for predicting the target variable, Yearly Amount Spent. Features like Email, Address, and Avatar were deemed irrelevant and dropped from the dataset.

```
rel_df = df.drop(columns=['Email', 'Address', 'Avatar'], axis=1)
```

### 6. Data Preprocessing
Missing Values and Duplicates: We checked for missing values and duplicates. Since there were none, we proceeded to the next step.

Renaming Columns: For better readability, we renamed the columns.

```
rel_df.rename(columns={
    'Avg. Session Length' : 'avg_session_length',
    'Time on App' : 'time_spent_on_app',
    'Time on Website' : 'time_spent_on_website', 
    'Length of Membership' : 'length_of_membership', 
    'Yearly Amount Spent' : 'yearly_amount_spent'
}, inplace=True)
```

### 7. Exploratory Data Analysis (Continued)
Univariate Analysis: We performed basic statistical analysis to gain insights into the distribution of the data.

Bivariate Analysis: We examined relationships between features like avg_session_length, time_spent_on_app, time_spent_on_website, length_of_membership, and yearly_amount_spent through various plots (scatter plots, joint plots, and line plots).

Multivariate Analysis: A pairplot was generated to analyze interactions between multiple variables at once.

Linear Regression Analysis: We checked for a linear relationship between the features and the target variable using seaborn’s lmplot.

```
sns.pairplot(rel_df)
sns.lmplot(x='length_of_membership', y='yearly_amount_spent', data=rel_df)
```

### 8. Model Building and Evaluation
We built a Linear Regression model to predict the Yearly Amount Spent based on the selected features.

Feature and Target Split: We separated the independent variables (features) and dependent variable (target).

Training and Test Split: The dataset was split into 80% for training and 20% for testing.

Model Training: We trained a Linear Regression model on the training data.

```
X = rel_df.drop('yearly_amount_spent', axis=1)
y = rel_df['yearly_amount_spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 9. Model Performance Evaluation
After training the model, we evaluated its performance using several metrics:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared (R2) score

These metrics helped us assess the prediction accuracy and how well the model generalizes to unseen data.

```
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
```

### 10. Residual Analysis
We examined the residual errors to ensure the model's predictions are unbiased. A probability plot was used to check the normality of the residuals, which is important for the assumptions of linear regression.

```
residuals = y_test - pred
import pylab 
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pylab)
```

## Technologies and Libraries Used
Python: Programming language used for data analysis and modeling.

Pandas: Data manipulation and analysis library.

NumPy: Library for numerical computations.

Matplotlib / Seaborn: Libraries for data visualization.

Scikit-learn: Machine learning library used for model building and evaluation.

Jupyter Notebook: Used for documenting the process and results.

## How to Run the Project
Clone the repository:

```
git clone https://github.com/sameeratanveer/learning_projects_assignment.git
```

Navigate to the project directory:

```
cd learning_projects_assignment
```

Install the required libraries:

```
pip install -r requirements.txt
```

Run the Jupyter notebook or Python script to execute the analysis and model:
```
jupyter notebook linear_reg_ecommerce_data.ipynb
```


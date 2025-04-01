# Boston Housing Price Prediction Using Linear Regression

## Overview
This project focuses on predicting housing prices in the Boston area using a machine learning regression model. The dataset used is the **Boston Housing Dataset**, which contains various features related to housing characteristics, such as crime rate (`CRIM`), average number of rooms per dwelling (`RM`), property tax rate (`TAX`), and more. The target variable is the median value of owner-occupied homes (`MEDV`), measured in thousands of dollars.

The goal of this assessment is to build a predictive model for `MEDV` and evaluate its performance using metrics like **Mean Squared Error (MSE)**. Additionally, the project explores the impact of **feature engineering** on model performance, demonstrating how creating new features or transforming existing ones can improve the model's ability to capture underlying patterns in the data.

---

## Dataset Description
The dataset consists of 14 columns:
1. **CRIM**: Per capita crime rate by town.
2. **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
3. **INDUS**: Proportion of non-retail business acres per town.
4. **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
5. **NOX**: Nitric oxide concentration (parts per 10 million).
6. **RM**: Average number of rooms per dwelling.
7. **AGE**: Proportion of owner-occupied units built prior to 1940.
8. **DIS**: Weighted distances to five Boston employment centers.
9. **RAD**: Index of accessibility to radial highways.
10. **TAX**: Full-value property tax rate per $10,000.
11. **PTRATIO**: Pupil-teacher ratio by town.
12. **B**: Proportion of people of African American descent by town.
13. **LSTAT**: Percentage of lower status of the population.
14. **MEDV**: Median value of owner-occupied homes in $1000s (target variable).

---

## Approach

### 1. Baseline Model
The baseline model was trained using only the original features from the dataset. A **Linear Regression** model was employed to predict `MEDV`. The performance of this model was evaluated using **Mean Squared Error (MSE)**, which quantifies the average squared difference between predicted and actual values.

- **Baseline Model MSE**: 24.999

### 2. Feature Engineering
To improve the model's performance, several feature engineering techniques were applied:
- **Interaction Terms**: Created new features by combining existing ones (e.g., `RM_ZN`, `CRIM_TAX`) to capture relationships between variables.
- **Polynomial Features**: Added squared or cubed terms for certain features (e.g., `RM_squared`, `LSTAT_cubed`) to account for non-linear relationships.
- **Derived Metrics**: Introduced domain-specific features, such as the median house price grouped by whether the property is near the Charles River (`MEDV_median_by_CHAS`).
- **Logarithmic Transformations**: Applied logarithmic transformations to skewed features (e.g., `log_CRIM`, `log_DIS`) to stabilize variance and reduce skewness.

These engineered features allowed the model to better capture complex patterns in the data.

### 3. New Model Performance
After incorporating the engineered features, the model's performance improved significantly:
- **New Model MSE**: 12.398

This represents a **50.4% reduction in MSE**, highlighting the effectiveness of feature engineering in enhancing the model's predictive power.

---

## Key Takeaways
1. **Feature Engineering Matters**: Thoughtfully designed features can significantly improve a model's ability to generalize and make accurate predictions.
2. **Non-Linearity**: Adding polynomial and interaction terms helps capture non-linear relationships that linear models cannot detect with raw features alone.
3. **Domain Knowledge**: Incorporating domain-specific insights (e.g., grouping by categorical variables) can lead to more meaningful features.

---

## Future Work
While the current results are promising, there are opportunities for further improvement:
- Experiment with advanced models like **Random Forest**, **Gradient Boosting**, or **Neural Networks**.
- Perform hyperparameter tuning to optimize model performance.
- Conduct additional feature selection to identify the most impactful features.

---

## Author Information
This project was completed by:

**Name**: Sajak Basnet  
**LinkedIn Profile**: https://www.linkedin.com/in/sajak-basnet-7b2792353/

---
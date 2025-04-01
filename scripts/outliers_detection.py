import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/boston_housing.csv.csv')

numerical_cols = data.select_dtypes(include=[np.number]).columns

outliers_info = {}

for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    outliers_info[col] = {
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers Count': len(outliers),
        'Outliers': outliers[col].tolist()
    }

for col, info in outliers_info.items():
    print(f"Column: {col}")
    print(f"  Lower Bound: {info['Lower Bound']:.2f}")
    print(f"  Upper Bound: {info['Upper Bound']:.2f}")
    print(f"  Outliers Count: {info['Outliers Count']}")
    print(f"  Outliers: {info['Outliers'][:5]}...")  # Show first 5 outliers
    print()
    
print("Outliers detection completed.")
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def preprocess_data(input_file , output_dir):
    #loading the fuckingg dataset
    df = pd.read_csv(input_file)

    #handling missing values
    df.fillna(df.median(), inplace=True) #inplace true modifies the original dataframe

    #seperating features and target
    X= df.drop(columns=['MEDV'])
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #saving the preprocessed data
        
    np.save(f'{output_dir}/X_train_scaled.npy', X_train_scaled)
    np.save(f'{output_dir}/X_test_scaled.npy', X_test_scaled)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)


if __name__ == "__main__":
    preprocess_data('../data/boston_housing.csv', '../data')

"""
Title: Neural Networks in Python
File Name: ClassProjectGroup6.py

Authors: Baden Stitt, Lucia Rojo, Michael Guillory, Noah Clark
Group: 6
Course: CMPS 3500: Programming Languages
Professor: Walter Morales
Last edited: 12. 06. 24

Description:
This Python script improves a machine learning model for predicting credit scores using a neural network approach as part of CSUB's CMPS 3500 Final Project. 
The system provides an interactive CLI for data loading, processing, model building, and testing.

Our modular pipeline approach is detailed below:
1.) Data Loading: We load the dataset from a selected CSV file.
2.) Data Cleaning: We prepare and clean the data for the model.
3.) Model Building: We build the neural network structure and prepare our hidden layers, epochs, and batch size.
4.) Model Testing: We train the model and evaluate the model's accuracy.
"""

import math
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings
import csv
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras

def calculate_performance_multiclass(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

class CreditScorePredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self.model = None         
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore')  # for features
        self.target_encoder = OneHotEncoder(handle_unknown='ignore')  # for target
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.le = LabelEncoder()
        self.scaler = StandardScaler()  # Added a scaler for numerical features
        self.start_time = None
        self.data_processed = False

    def load_data(self):
        print("\nLoading Data: ************************************")
        self.start_time = time.time()
        
        # Load the data
        current_dir = os.getcwd()

        parent_dir = os.path.dirname(current_dir)
        files = []


        data_dir = os.path.join(parent_dir, 'data')
        for filename in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith('.csv'):
                files.append(filename)

        for i, file in enumerate(files):
            print(f"{i+1}. {file}")
        
        choice = int(input("Enter the number of the file you want to load: "))
        file_path = os.path.join(data_dir, files[choice-1])

        
        self.df = pd.read_csv(file_path)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Script")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading training data set")
        
        total_columns = len(self.df.columns)
        if len(self.df.columns) != 29:
            self.df = None
            raise ValueError("Data columns do not match expected columns: " + str(total_columns))
        total_rows = len(self.df)

        if total_rows < 90:
            self.df = None
            raise ValueError("No/Insufficient data in the file: " + str(total_rows))
        
        print(f"Total Columns Read: {total_columns}")
        print(f"Total Rows Read: {total_rows}")
        
        load_time = time.time() - self.start_time
        print(f"Time to load is: {load_time:.2f} seconds")

    def process_data(self):
        if self.df is None:
            print("Please load data first!")
            return
        
        print("\nProcessing Data: **************************")
        self.start_time = time.time()
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Performing Data Clean Up")
        columns_to_drop = [
            'Unnamed: 0', 'Month', 'Name', 'SSN',
            'Num_Bank_Accounts', 'Num_of_Loan', 'Type_of_Loan', 
            'Delay_from_due_date', 'Num_Credit_Inquiries', 
            'Total_EMI_per_month', 'Amount_invested_monthly',
            'Credit_History_Age'
        ]

        self.df.drop(columns=columns_to_drop, inplace=True)
        
        try:
            self.df['Annual_Income'] = self.df['Annual_Income'].str.replace('_', '').astype(float)
            self.df.loc[self.df['Annual_Income'] > 180000, 'Annual_Income'] = pd.NA
            self.df['Annual_Income'] = self.df.groupby('Customer_ID')['Annual_Income'].fillna(method='ffill').fillna(method='bfill')

            self.df['Monthly_Inhand_Salary'] = self.df.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='ffill').fillna(method='bfill')
            
            self.df.loc[self.df['Interest_Rate'] > 34, 'Interest_Rate'] = pd.NA
            self.df['Interest_Rate'] = self.df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.fillna(x.median()))

            self.df['Outstanding_Debt'] = self.df['Outstanding_Debt'].str.replace('_', '')
            self.df['Outstanding_Debt'][self.df['Outstanding_Debt'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
            self.df['Outstanding_Debt'] = self.df.groupby('Customer_ID')['Outstanding_Debt'].fillna(method='ffill').fillna(method='bfill').astype(float)
            

            self.df['Changed_Credit_Limit'][self.df['Changed_Credit_Limit'].str.fullmatch('[+-]?([0-9]*[.])?[0-9]+')].unique()
            self.df['Changed_Credit_Limit'][self.df['Changed_Credit_Limit'] == '_'] = np.nan
            self.df['Changed_Credit_Limit'] = self.df.groupby('Customer_ID')['Changed_Credit_Limit'].fillna(method='ffill').fillna(method='bfill').astype(float)

            temp_series = self.df['Num_of_Delayed_Payment'][self.df['Num_of_Delayed_Payment'].notnull()]
            temp_series[~temp_series.str.isnumeric()].unique()
            self.df['Num_of_Delayed_Payment'] = self.df['Num_of_Delayed_Payment'].str.replace('_', '').astype(float)
            self.df['Num_of_Delayed_Payment'] = self.df.groupby('Customer_ID')['Num_of_Delayed_Payment'].fillna(method='ffill').fillna(method='bfill').astype(float)


            self.df['Monthly_Balance'] = pd.to_numeric(self.df['Monthly_Balance'], errors='coerce')
            self.df['Monthly_Balance'] = self.df.groupby('Customer_ID')['Monthly_Balance'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

            
            self.df['Occupation'][self.df['Occupation'] == '_______'] = np.nan
            self.df['Occupation'] = self.df.groupby('Customer_ID')['Occupation'].fillna(method='ffill').fillna(method='bfill').astype("string")

            
            self.df['Credit_Mix'][self.df['Credit_Mix'] == '_'] = np.nan
            self.df['Credit_Mix'] = self.df.groupby('Customer_ID')['Credit_Mix'].fillna(method='ffill').fillna(method='bfill').astype("string")

            self.df['Payment_Behaviour'][self.df['Payment_Behaviour'] == '!@9#%8'] = np.nan
            self.df['Payment_Behaviour'] = self.df.groupby('Customer_ID')['Payment_Behaviour'].fillna(method='ffill').fillna(method='bfill').astype("string")

            #affects how much disposable income someone actually has access to each month        
            self.df['Income_to_Salary'] = self.df['Annual_Income'] / (self.df['Monthly_Inhand_Salary'] * 12)

            self.df['Credit_Score'] = self.df['Credit_Score'].astype("string")

            self.df = self.df.drop(columns='Customer_ID')
            self.df['ID'] = self.df['ID'].astype('string')
        except Exception as e:
            self.df = None
            self.data_processed = False
            raise ValueError("Error in processing data: " + str(e))
        
        total_rows_after_cleaning = len(self.df)
        process_time = time.time() - self.start_time

        print(f"Total Rows after cleaning is: {total_rows_after_cleaning}")
        print(f"Time to process is: {process_time:.2f} seconds")

        self.data_processed = True

    def build_model(self):
        if self.df is None:
            print("Please load and process data first!")
            return
        if not self.data_processed:
            print("Please process data first!")
            return
        
        print("\nBuilding Model: ***********************")
        
        categorical_features = ['Occupation', 'Credit_Mix']
        numerical_features = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate','Income_to_Salary', 
                            'Outstanding_Debt', 'Monthly_Balance'
                            ]
        target = ['Credit_Score']

        #added scaling here
        numerical_features_scaled = self.scaler.fit_transform(self.df[numerical_features])
        categorical_features_scaled = self.cat_encoder.fit_transform(self.df[categorical_features]).toarray()
    
        self.X = np.hstack([numerical_features_scaled, categorical_features_scaled])
    
        self.y = self.target_encoder.fit_transform(self.df[target]).toarray()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)
        
        self.model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.X_train.shape[1], activation='relu'),
            keras.layers.Dense(96, activation="relu"),
            keras.layers.Dense(216, activation="relu"),
            keras.layers.Dense(216, activation="relu"),
            keras.layers.Dense(96, activation="relu"),
            keras.layers.Dense(3, activation="softmax")
        ])

        self.model.compile(optimizer='adam', 
                        loss=tf.keras.losses.CategoricalCrossentropy(), 
                        metrics=['accuracy'])

        print(self.model.summary())

        
        print("\nModel Details: ***********************")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model Architecture Created")
        print("Hyper-parameters used are:")
        print("- Input Layer: 24 nodes")
        print("- Hidden Layers: 96, 216, 216, 96 nodes")
        print("- Output Layer: 3 nodes (softmax)")
        print("- Loss Function: Categorical Cross-Entropy")
        print("- Test Size: 20%")

    def test_model(self):
        if self.model is None:
            print("Please build the model first!")
            return
        
        print("\nTesting Model: **************")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating prediction using selected Neural Network")
        
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=50, verbose=1)
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model Performance Metrics:")
        print(f"Model RMSE: {test_loss}")
        print(f"Test Accuracy: {test_acc}")
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Size of training set: {len(self.X_train)}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Size of testing set: {len(self.X_test)}")
        
        # Generate predictions
        predictions = self.model.predict(self.X_test)
        
        # Convert predictions back to original labels
        y_predicted = self.target_encoder.inverse_transform(predictions)
        y_tested = self.target_encoder.inverse_transform(self.y_test)
        
        data = [[y_tested[i], y_predicted[i]] for i in range(15)]
        headers = ["True Value", "Predicted Value"]
        print(tabulate(data, headers=headers, tablefmt="grid"))

        # Calculate performance
        metrics = calculate_performance_multiclass(y_tested, y_predicted)
        print(metrics)

        # Save predictions to CSV
        # Use train_test_split indices to match the predictions with original IDs
        ids_for_test = self.df['ID'].iloc[self.y_test.argmax(axis=1)]
        prediction_data = list(zip(ids_for_test, y_predicted))
        
        with open('predictions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Prediction'])
            writer.writerows(prediction_data)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Predictions generated and saved to predictions.csv")
def main():
    predictor = CreditScorePredictor()
    
    while True:
        print("\n*  Credit Score Prediction System")
        print("*  (1) Load data")
        print("*  (2) Process data")
        print("*  (3) Model details")
        print("*  (4) Test model")
        print("*  (5) Quit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            try:
                predictor.load_data()
            except Exception as e:
                predictor.data_processed = False
                print(f"An error occurred: {e}")
        elif choice == '2':
            try:
                predictor.process_data()
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Unloading Data...")
                predictor.df = None
                predictor.data_processed = False
        elif choice == '3':
            try:
                predictor.build_model()
            except Exception as e:
                print(f"An error occurred: {e}")
        elif choice == '4':
            try:
                predictor.test_model()
            except Exception as e:
                print(f"An error occurred: {e}")
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

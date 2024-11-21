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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.le = LabelEncoder()
        self.start_time = None

    def load_data(self):
        print("\nLoading Data: ************************************")
        self.start_time = time.time()
        
        # Load the data
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, "data\credit_score_data.csv")
        
        self.df = pd.read_csv(file_path)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Script")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading training data set")
        
        total_columns = len(self.df.columns)
        total_rows = len(self.df)
        
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
        
        # Data cleaning steps from original script
        columns_to_drop_unrelated = ['Unnamed: 0', 'Month', 'Name', 'SSN']
        columns_to_drop_not_used = ['Num_Bank_Accounts', 'Num_of_Loan', 'Type_of_Loan', 'Delay_from_due_date', 
                                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 
                                    'Payment_of_Min_Amount', 'Num_Credit_Inquiries', 'Total_EMI_per_month', 
                                    'Amount_invested_monthly', 'Credit_History_Age', 'Monthly_Balance', 
                                    'Payment_Behaviour']
        # input but not 100% sure if done correctly
        # 'Outstanding_Debt'
        
        self.df.drop(columns=columns_to_drop_unrelated + columns_to_drop_not_used, inplace=True)
        
        # Clean specific columns
        self.df['Age'] = self.df['Age'].str.replace('_', '').astype(int)
        self.df['Age'][(self.df['Age'] > 100) | (self.df['Age'] <= 0)] = np.nan
        self.df['Age'] = self.df.groupby('Customer_ID')['Age'].fillna(method='ffill').fillna(method='bfill').astype(int)

        self.df['Occupation'][self.df['Occupation'] == '_______'] = np.nan
        self.df['Occupation'] = self.df.groupby('Customer_ID')['Occupation'].fillna(method='ffill').fillna(method='bfill').astype("string")

        self.df['Annual_Income'] = self.df['Annual_Income'].str.replace('_', '').astype(float)
        self.df.loc[self.df['Annual_Income'] > 180000, 'Annual_Income'] = pd.NA
        self.df['Annual_Income'] = self.df.groupby('Customer_ID')['Annual_Income'].fillna(method='ffill').fillna(method='bfill')

        self.df['Monthly_Inhand_Salary'] = self.df.groupby('Customer_ID')['Monthly_Inhand_Salary'].fillna(method='ffill').fillna(method='bfill')

        self.df.loc[self.df['Num_Credit_Card'] > 11, 'Num_Credit_Card'] = pd.NA
        self.df['Num_Credit_Card'] = self.df.groupby('Customer_ID')['Num_Credit_Card'].fillna(method='ffill').fillna(method='bfill')

        self.df.loc[self.df['Interest_Rate'] > 34, 'Interest_Rate'] = pd.NA
        self.df['Interest_Rate'] = self.df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())

        self.df['Credit_Mix'][self.df['Credit_Mix'] == '_'] = np.nan
        self.df['Credit_Mix'] = self.df.groupby('Customer_ID')['Credit_Mix'].fillna(method='ffill').fillna(method='bfill').astype("string")

#        self.df['Outstanding_Debt'] = self.df['Outstanding_Debt'].str.replace('_', '').astype(float)
#        self.df['Outstanding_Debt'][self.df['Outstanding_Debt'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
#        self.df['Outstanding_Debt'] = self.df.groupby('Customer_ID')['Outstanding_Debt'].fillna(method='ffill').fillna(method='bfill').astype(float)
        
        self.df['Credit_Score'] = self.df['Credit_Score'].astype("string")

        
        # Drop Customer_ID and encode categorical features
        self.df = self.df.drop(columns='Customer_ID')
        self.df['ID'] = self.df['ID'].astype('string')
        
        total_rows_after_cleaning = len(self.df)
        process_time = time.time() - self.start_time
        
        print(f"Total Rows after cleaning is: {total_rows_after_cleaning}")
        print(f"Time to process is: {process_time:.2f} seconds")

    def build_model(self):
        if self.df is None:
            print("Please load and process data first!")
            return
        
        print("\nBuilding Model: ***********************")
        
        # Prepare features and target
        categorical_features = ['Occupation', 'Credit_Mix']
        target = ['Credit_Score']
        
        # Encode categorical features
        encoded_features = self.encoder.fit_transform(self.df[categorical_features])
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=self.encoder.get_feature_names_out(categorical_features))
        
        # Prepare features and target
        self.X = encoded_features.toarray()
        self.y = self.encoder.fit_transform(self.df[target]).toarray()
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)
        
        # Build Neural Network
        self.model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.X_train.shape[1], activation='relu'),
            keras.layers.Dense(48, activation="relu"),
            keras.layers.Dense(96, activation="relu"),
            keras.layers.Dense(96, activation="relu"),
            keras.layers.Dense(48, activation="relu"),
            keras.layers.Dense(3, activation="softmax")
        ])
        
        self.model.compile(optimizer='adam', 
                        loss=tf.keras.losses.CategoricalCrossentropy(), 
                        metrics=['accuracy'])
        
        print("\nModel Details: ***********************")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model Architecture Created")
        print("Hyper-parameters used are:")
        print("- Input Layer: 24 nodes")
        print("- Hidden Layers: 48, 96, 96, 48 nodes")
        print("- Output Layer: 3 nodes (softmax)")
        print("- Loss Function: Categorical Cross-Entropy")
        print("- Test Size: 20%")

    def test_model(self):
        if self.model is None:
            print("Please build the model first!")
            return
        
        print("\nTesting Model: **************")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating prediction using selected Neural Network")
        
        # Fit the model
        self.model.fit(self.X_train, self.y_train, epochs=12, batch_size=20, verbose=0)
        
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
        y_predicted = self.encoder.inverse_transform(predictions)
        y_tested = self.encoder.inverse_transform(self.y_test)
        
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
            predictor.load_data()
        elif choice == '2':
            predictor.process_data()
        elif choice == '3':
            predictor.build_model()
        elif choice == '4':
            predictor.test_model()
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

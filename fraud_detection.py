import streamlit as st
import pickle
import numpy as np 
import pandas as pd  
import matplotlib as plt
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.preprocessing import StandardScaler,LabelEncoder 
from sklearn.metrics import accuracy_score ,classification_report , confusion_matrix 
import xgboost as xgb

def load_model():
    with open("fraud_detection_XGBoost.pkl", 'rb') as file:
        model,scaler,le = pickle.load(file)
        return model,scaler,le

def preprocessing_data(data, scaler, le):
    cat_columns =['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method']
    for col in cat_columns:
        data[col]= le.fit_transform([data[col]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

#Transaction_Amount	
# Transaction_Type	
# Account_Balance	
# Device_Type	
# Location	
# Merchant_Category	
# IP_Address_Flag	
# Previous_Fraudulent_Activity	
# Daily_Transaction_Count	
# Avg_Transaction_Amount_7d	
# Failed_Transaction_Count_7d	
# Card_Type	
# Card_Age	
# Transaction_Distance	
# Authentication_Method	
# Risk_Score	
# Is_Weekend

def main():
    st.title('Fraud Detection Prediction')
    st.write('Enter your data to predict the fraud')

    Transaction_Amount = st.number_input("Transaction_Amount", min_value= 0, value= 0)
    Transaction_Type = st.selectbox("Transaction_Type", ['POS', 'Bank Transfer', 'Online', 'ATM Withdrawal'])
    Device_Type = st.selectbox("Device_Type", ['Mobile', 'Laptop', 'Tablet'])
    Location = st.selectbox("Location", ['Sydney', 'New York', 'Mumbai', 'Tokyo', 'London'])
    Merchant_Category = st.selectbox("Merchant_Category", ['Travel', 'Clothing', 'Restaurants', 'Electronics', 'Groceries'])
    Card_Type = st.selectbox("Card_Type", ['Amex', 'Mastercard', 'Visa', 'Discover'])
    Authentication_Method = st.selectbox("Authentication_Method", ['Biometric', 'Password', 'OTP', 'PIN'])
    Account_Balance = st.number_input("Account_Balance", min_value= 0, value= 0)
    IP_Address_Flag = st.number_input("IP_Address_Flag", min_value= 0, value= 0)
    Previous_Fraudulent_Activity = st.number_input("Previous_Fraudulent_Activity", min_value= 0, value= 0)
    Daily_Transaction_Count = st.number_input("Daily_Transaction_Count", min_value= 0, value= 0)
    Avg_Transaction_Amount_7d = st.number_input("Avg_Transaction_Amount_7d", min_value= 0, value= 0)
    Failed_Transaction_Count_7d =st.number_input("Failed_Transaction_Count_7d", min_value= 0, value= 0)
    Card_Age = st.number_input("Card_Age", min_value= 0, value= 0)
    Transaction_Distance = st.number_input("Transaction_Distance", min_value= 0, value= 0)
    Risk_Score = st.number_input("Risk_Score", min_value= 0, value= 0)
    Is_Weekend = st.number_input("Is_Weekend", min_value= 0, value= 0)

    if st.button("Predict Fraud"):
        #here we will map all the data that is taken in the above variables to our actual headers of data set. here by chance 
        # i have taken bith same.
        user_data = {
            #"Authentication_Method":Authentication_Method,
            "Transaction_Amount":Transaction_Amount,
            "Transaction_Type":Transaction_Type,
            "Account_Balance":Account_Balance,
            "Device_Type":Device_Type,
            "Location":Location,
            "Merchant_Category":Merchant_Category,
            "IP_Address_Flag":IP_Address_Flag,
            "Previous_Fraudulent_Activity":Previous_Fraudulent_Activity,
            "Daily_Transaction_Count":Daily_Transaction_Count,
            "Avg_Transaction_Amount_7d":Avg_Transaction_Amount_7d,
            "Failed_Transaction_Count_7d":Failed_Transaction_Count_7d,
            "Card_Type":Card_Type,
            "Card_Age":Card_Age,
            "Transaction_Distance":Transaction_Distance,
            "Authentication_Method":Authentication_Method,
            "Risk_Score":Risk_Score,
            "Is_Weekend":Is_Weekend
        }

        prediction = predict_data(user_data)
        st.success(f"The predicted fraud is {prediction}")


if __name__ == "__main__":
    main()
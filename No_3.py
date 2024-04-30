import streamlit as st
import joblib
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder  
model = joblib.load('xgb_model.pkl')

def main():
    st.title('XGB Model')

    credit_score = st.number_input('Credit Score', 0, 850)
    age = st.number_input('Age', 17, 100)
    balance = st.number_input('Balance', 0)
    estimated_salary = st.number_input('Estimated Salary', 0)
    gender = st.selectbox('Gender', ['Male', 'Female'])  
    tenure = st.number_input('Tenure', 0, 10)
    num_of_products = st.number_input('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
    is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])  

    if st.button('Make Prediction'):
        label_encoder = LabelEncoder()
        gender_encoded = label_encoder.fit_transform([gender])[0]
        has_cr_card_encoded = label_encoder.fit_transform([has_cr_card])[0]
        is_active_member_encoded = label_encoder.fit_transform([is_active_member])[0]

        features = [
            credit_score, gender_encoded, age, tenure, balance, 
            num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary
        ]
        result = make_prediction(features)
            
        if result == 1:
            prediction_text = "Customer is predicted to churn."
        else:
            prediction_text = "Customer is predicted to stay."
            
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1) 
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  

model = joblib.load('xgb_classifier.pkl')
gender_encode = joblib.load('gender_encode.pkl')

def main():
    st.title('Churn XGBoost')

    age = st.number_input('Age', 17, 100)
    gender = st.radio('Gender', ["Male","Female"])
    credit_score = st.number_input('Credit Score', 0, 850)
    estimated_salary = st.number_input('Estimated Salary', 0)
    tenure = st.number_input('Tenure', 0, 10)
    balance = st.number_input("Balance", 0,10)
    num_of_products = st.number_input('Number of Products', 1, 4)
    has_cr_card = st.radio('Has Credit Card?', ['Yes', 'No']) 
    is_active_member = st.radio('Is Active Member?', ['Yes', 'No'])  

    if st.button('Make Prediction'):
        data = {'Age': int(age), 'Gender': gender, 'Credit Score':int(credit_score),
                'Estimated Salary': int(estimated_salary),  'Tenure':int(tenure),
                'Balance':int(balance), 'Number of Products':num_of_products, 
                'Credit Card': has_cr_card, 'Active Member':(is_active_member)}

        df=pd.DataFrame([list(data.values())], columns=['Age', 'Gender', 'Credit Score', 'Estimated Salary',
                                                        'Tenure', 'Balance', 'Number of Products', 'Credit Card', 'Active Member',
                                                        ])
        df = df.replace(gender_encode)
        label_encoder = LabelEncoder()
        df['Credit Card'] = label_encoder.fit_transform(df['Credit Card'])
        df['Active Member'] = label_encoder.fit_transform(df['Active Member'])

        features = df
        result = make_prediction(features)

        if result == 1:
            prediction_text = "Customer is predicted to churn."
        else:
            prediction_text = "Customer is predicted to stay."

        st.success(prediction_text)

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

model = load_model('artifacts/model/model.keras')

preprocessing_files = os.listdir("artifacts/preprocessing")

def load_files(files) -> list:
    """
    Returns preprocessing files in [gender encoder, geography encoder, scaler]
    """
    preprocessors = []
    for i in files:
        with open(os.path.join('artifacts/preprocessing', i), 'rb') as file:
            preprocessors.append(pickle.load(file))
    
    print(preprocessors)
    return preprocessors

preprocessors = load_files(preprocessing_files)

st.title("Customer Churn Prediction")
geo = st.selectbox('Geography', preprocessors[1].categories_[0])
gender = st.selectbox('Gender', preprocessors[0].classes_)
age = st.slider('Age', 18, 99)
balance = st.number_input('Balance')
cs = st.number_input('Credit Score')
es = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
nop = st.slider("Number of Products", 1, 4)
hcc = st.selectbox('Has Credit Card', [0, 1])
iam = st.selectbox("Is Active Member", [0, 1])

#Input data
input_data = {
    'CreditScore': cs,
    "Geography": geo,
    "Gender": gender,
    'Age': age,
    "Tenure": tenure,
    'Balance': balance,
    'NumOfProducts': nop,
    'HasCrCard': hcc,
    'IsActiveMember': iam,
    'EstimatedSalary': es
}

sample_data = pd.DataFrame([input_data])

geo_sample = pd.DataFrame(preprocessors[1].transform(sample_data[["Geography"]]), columns = preprocessors[1].get_feature_names_out())

sample_data = pd.concat([sample_data, geo_sample], axis = 1).drop("Geography", axis = 1)

sample_data['Gender'] = preprocessors[0].transform(sample_data['Gender'])

sample_data = preprocessors[2].transform(sample_data)

churn_probability = model.predict(sample_data)[0][0]

st.write(f"Churn Probability: {churn_probability}")

if churn_probability >= 0.5:
    st.write("The customer will likely churn")

else:
    st.write("The customer will likely not churn")

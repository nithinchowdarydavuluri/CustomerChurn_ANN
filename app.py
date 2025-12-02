import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title("Customer Churn Predition")

#User input 
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credict Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data  = {
    'CreditScore':credit_score,
    'Geography':geography,
    'Gender':gender,	
    'Age':age,	
    'Tenure':tenure,	
    'Balance':balance,	
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,	
    'IsActiveMember':is_active_member,	
    'EstimatedSalary':estimated_salary	
}

input_data = pd.DataFrame([input_data])
geo_encoded = onehot_encoder_geo.transform([input_data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))



input_data = pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)
print(input_data)
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])


input_scaled = scaler.transform(input_data)
predition = model.predict(input_scaled)

predition_proba = predition[0][0]
print(predition_proba)

if predition_proba > 0.5:
    print("The customer is likely to churn")
else:
    print("The customer is not likely to churn")
# integrating ANN model with Streamlit App
import streamlit as st
import tensorflow as tf
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

# load the train model
model=tf.keras.models.load_model('250221-130437.keras')

# load the encoders and scaler
with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)
with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file) 
# scaler=StandardScaler()

# Streamlit app
st.title('Customer Chunk Prediction')
# user input 
name=st.text_input("name :")
age=st.slider('age :',18,100)
gender=st.selectbox('gender',label_encoder.classes_)
Geography=st.selectbox('geography',one_hot_encoder.categories_[0])

# prepare the input data
input_data=pd.DataFrame({
    'name':[name],
    'age':[age],
    'gender':[label_encoder.transform([gender])[0]],
    'Geography':[Geography],
    'Exited':0
})

# drop unwanted columns
input_data=input_data.drop('name',axis=1)

# onehotencoder for 'Geography'
one_hot_encoder_geo=one_hot_encoder.transform([[Geography]]).toarray()
one_hot_encoder_geo=pd.DataFrame(one_hot_encoder_geo,columns=one_hot_encoder.get_feature_names_out(["Geography"]))

# combine one_hot_encoder_geo and input_data after droping Geography column
input_data=input_data.drop('Geography',axis=1)
input_data=pd.concat([input_data.reset_index(drop=True),one_hot_encoder_geo],axis=1)


# print(input_data)

# # scale the input data
# input_data_scaled=scaler.transform(input_data)
# input_data=input_data.drop('unnamed: 0')
# input_data.replace("", np.nan, inplace=True)
# input_data=input_data.astype(np.float32)
prediction=model.predict(input_data)
prediction_prob=prediction[0][0]

if prediction_prob>0.99:
    st.write('customer is likes to chunk...')
else:
    st.write("customer is not likes to chunk...")

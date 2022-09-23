#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from PIL import Image
import streamlit as st


st.write("""
# Cervical Cancer Detection App
Detects if any person has Cervical Cancer or not !
""")
image = Image.open('cervical-cancer.jpg')
st.image(image, caption = 'Cervical Cancer using ML', use_column_width = True)

df_cv = pd.read_csv('risk_factors_cervical_cancer.csv')


df_cv.describe()

df_cv.head()

df_cv.info()

df_cv.isnull().sum()

df_cv.isnull()

st.subheader("Data Information: ")
st.dataframe(df_cv)
st.write(df_cv.describe())
#st.bar_chart(df_cv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Define input and output features

X = df_cv[['Age', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV',
          'Dx', 'Hinselmann', 'Schiller', 'Citology']]

y = df_cv['Biopsy']

#Splitting the dataset into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Get the data and information from users
def get_user_data():
    age = st.sidebar.slider('age', 13,84,40)
    std_diagnosis = st.sidebar.slider('std_diagnosis', 0,3,1)
    CIN = st.sidebar.slider('CIN', 0.0,1.0)
    HPV = st.sidebar.slider('HPV', 0.0,1.0)
    Hinselmann = st.sidebar.slider('Hinselmann', 0.0,1.0)
    dx = st.sidebar.slider('dx', 0.0,1.0)
    cancer = st.sidebar.slider('cancer', 0.0,1.0)
    Schiller = st.sidebar.slider('Schiller',0.0,1.0)
    Citology = st.sidebar.slider('Citology',0.0,1.0)

    # Store the user data into a dictionary
    user_data = {'age' : age,
                 'std_diagnosis' : std_diagnosis,
                 'CIN' : CIN,
                 'HPV' : HPV,
                 'Dx' : dx,
                 'Hinselmann' : Hinselmann,
                 'cancer' : cancer,
                 'Schiller' : Schiller,
                 'Citology' : Citology
                }
    # Transform the data into dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_data()

# Creating Subheader for the input from users

st.subheader("User Inputs : ")
st.write(user_input)

# Creating a model and testing the accuracy after feeding the training data into model

model = RandomForestClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# Showing the model performance
st.subheader("The accuracy of the model is : ")
st.write( str(accuracy_score(y_test, prediction)* 100)+'%' )

prediction_user = model.predict(user_input)

# Showing the biopsy result of the users
st.subheader('The person is diagnosed with positive/negative biopsy result : ')
st.write(prediction_user)
    
    





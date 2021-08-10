import streamlit as st
# Update title and icon
st.set_page_config(
    page_title="Predict Health Costs With Regression",
    page_icon='https://link477.com/images/link477.png'
)

st.title("Predict Health Costs With Regression")

from getData import getTestAndTrainingData, getFilePath, getData
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys,os
import pandas as pd


#st.write(getFilePath())

train_dataset, test_dataset = getTestAndTrainingData(getData())

# Pop the expenses column off
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Get from interpreter
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Predict Health costs
age = st.sidebar.slider(
    'Enter your age',
    18.0,
    100.0,
    39.0,1.0
)

sex = st.sidebar.radio(
    'Gender:',
    ('Male','Female')
)

bmi = st.sidebar.slider(
    'BMI:',
    16.0,
    55.0,
    30.0,
    0.1
)

children = st.sidebar.number_input(
    'Number of Children:',
    0
)

smoker = st.sidebar.checkbox(
    'Smoker?'
)

region = st.sidebar.selectbox(
    'Region:',
    ['South West', 'South East', 'North West', 'North East']
)

# test interpreted model on input data

inputtensor = interpreter.tensor(interpreter.get_input_details()[0]["index"])
outputtensor = interpreter.tensor(interpreter.get_output_details()[0]["index"])
#st.write(inputtensor())

def InterpreterPredictFromDataFrame(age, bmi, children, smoker, female, male, southwest, southeast, northwest, northeast):
    '''
    This is to input a dataframe and get the results from that
    Unfortunately the easy way of setting the tensor through interpreter.set_tensor(input_details[0]['index'], input_data) decided not to work
    '''
    inputtensor()[0][0] = age
    inputtensor()[0][1] = bmi
    inputtensor()[0][2] = children
    inputtensor()[0][3] = smoker
    inputtensor()[0][4] = female
    inputtensor()[0][5] = male
    inputtensor()[0][6] = southwest
    inputtensor()[0][7] = southeast
    inputtensor()[0][8] = northwest
    inputtensor()[0][9] = northeast
    interpreter.invoke()
    return outputtensor()[0][0]



# Predict the graph

test_predictions2 = test_dataset.apply(lambda x: InterpreterPredictFromDataFrame(x.age, x.bmi, x.children, x.smoker, x.female, x.male, x.southwest, x.southeast, x.northwest, x.northeast), axis=1)

fig, ax = plt.subplots()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions2)
plt.xlabel('True Values (expenses)')
plt.ylabel('Predictions (expenses)')
plt.title('Predicted Expenses vs Actual Expenses')
lims = [0, 5000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
st.pyplot(fig)

# Predict their one
predict2 = InterpreterPredictFromDataFrame(age, bmi, children, smoker, sex == 'Female', sex == 'Male', region == 'South West', region == 'South East', region == 'North West', region == 'North East')
#st.write(predict2)
st.write("Your predicted health costs are ${:,.2f}.".format(predict2))

#interpreter.set_tensor(input_details[0]['index'], input_data)
##interpreter.set_tensor(0, float(age))
#interpreter.invoke()

#https://www.tensorflow.org/lite/guide/inference
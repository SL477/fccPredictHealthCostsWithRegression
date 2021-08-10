import streamlit as st
# Update title and icon
st.set_page_config(
    page_title="Predict Health Costs With Regression",
    page_icon='https://link477.com/images/link477.png'
)

st.title("Predict Health Costs With Regression")

from getData import getTestAndTrainingData, getFilePath, getData
import tensorflow as tf
#from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys,os
import pandas as pd


#st.write(getFilePath())

train_dataset, test_dataset = getTestAndTrainingData(getData())

model = load_model(os.path.join(getFilePath(), 'model.h5'))

# Pop the expenses column off
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

evaluationStr = "Testing set Mean Abs Error: {:,.2f} expenses".format(mae)

st.write(evaluationStr)

# Plot the predictions
test_predictions = model.predict(test_dataset).flatten()

# Get from interpreter
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

fig, ax = plt.subplots()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values (expenses)')
plt.ylabel('Predictions (expenses)')
plt.title('Predicted Expenses vs Actual Expenses')
lims = [0, 5000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
st.pyplot(fig)

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

d = {
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'female': [sex == 'Female'],
    'male': [sex == 'Male'],
    'southwest': [region == 'South West'],
    'southeast': [region == 'South East'],
    'northwest': [region == 'North West'],
    'northeast': [region == 'North East']
}

df = pd.DataFrame(data=d)
df['age'] = df['age'].astype(np.int32)
df['bmi'] = df['bmi'].astype(np.float32)
df['children'] = df['children'].astype(np.int32)
df['smoker'] = df['smoker'].astype(np.int32)
df['female'] = df['female'].astype(np.int32)
df['male'] = df['male'].astype(np.int32)
df['southwest'] = df['southwest'].astype(np.int32)
df['southeast'] = df['southeast'].astype(np.int32)
df['northwest'] = df['northwest'].astype(np.int32)
df['northeast'] = df['northeast'].astype(np.int32)

st.write("## Predicted Health Costs")
#st.write(df.head())
st.write("Your predicted health costs are ${:,.2f}.".format(model.predict(df)[0][0]))

# test interpreted model on input data

st.write('tensor')
inputtensor = interpreter.tensor(interpreter.get_input_details()[0]["index"])
outputtensor = interpreter.tensor(interpreter.get_output_details()[0]["index"])
#st.write(inputtensor())

inputtensor()[0][0] = float(age)
inputtensor()[0][1] = float(bmi)
inputtensor()[0][2] = float(children)
inputtensor()[0][3] = float(smoker)
inputtensor()[0][4] = float(sex == 'Female')
inputtensor()[0][5] = float(sex == 'Male')
inputtensor()[0][6] = float(region == 'South West')
inputtensor()[0][7] = float(region == 'South East')
inputtensor()[0][8] = float(region == 'North West')
inputtensor()[0][9] = float(region == 'North East')
st.write(inputtensor())
interpreter.invoke()
st.write(outputtensor())
st.write("Your predicted health costs are ${:,.2f}.".format(outputtensor()[0][0]))

#interpreter.set_tensor(input_details[0]['index'], input_data)
##interpreter.set_tensor(0, float(age))
#interpreter.invoke()

#https://www.tensorflow.org/lite/guide/inference
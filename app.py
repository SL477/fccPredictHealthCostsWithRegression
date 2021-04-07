import streamlit as st


st.title("Predict Health Costs With Regression")

from getData import getTestAndTrainingData, getFilePath, getData
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import sys,os

#st.write(getFilePath())

train_dataset, test_dataset = getTestAndTrainingData(getData())

model = load_model(os.path.join(getFilePath(), 'model.h5'))

# Pop the expenses column off
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

evaluationStr = "Testing set Mean Abs Error: {:5.2f} expenses".format(mae)

st.write(evaluationStr)

# Plot the predictions
test_predictions = model.predict(test_dataset).flatten()

fig, ax = plt.subplots()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 5000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
st.pyplot(fig)
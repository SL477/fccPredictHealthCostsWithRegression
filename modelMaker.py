'''
This is to make the model
'''
from getData import getData, getTestAndTrainingData
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
df = pd.read_csv('insurance.csv')
train_dataset, test_dataset = getTestAndTrainingData(df)

# Pop the expenses column off
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')
print('train_dataset\n',train_dataset.head())
print('train_dataset info\n',train_dataset.info())
print('train_labels\n',train_labels.head())

# Normalise the data
from tensorflow.keras.layers.experimental import preprocessing
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_dataset))

# Create Model
def getModel():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  model.compile(
      optimizer='rmsprop',
      loss='mse',
      metrics=['mae','mse']
  )
  return model

model = getModel()
model.fit(train_dataset, train_labels, epochs=60, batch_size=2, shuffle=True)

model.save('model.h5')
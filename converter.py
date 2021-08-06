# This is to convert a tensorflow model to tensorflowlite
import tensorflow as tf
from tensorflow.keras.models import load_model
from getData import getTestAndTrainingData, getFilePath, getData
import sys,os

# load
model = load_model('model.h5')

# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
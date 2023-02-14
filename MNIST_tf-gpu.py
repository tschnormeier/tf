# www.kaggle.com/hassanamin/tensorflow-mnist-gpu-tutorial
# Titled: "Tensorflow MNIST GPU Tutorial"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read.csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import tensorflow as tf

# To check GPU availability in Tensorflow

gpus = tf.config.experimental.list_physical_devices('GPU')
print("type(gpus) is: ", type(gpus))
for gpu in gpus:
    print("Name: ", gpu.name, "  Type: ", gpu.device_type)

# Listing Devices including GPU's with Tensorflow

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# To Check GPU in Tensorflow
gpu_flag = tf.test.is_gpu_available()
print("tf.test.is_gpu_available() returned: ", gpu_flag)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-processing of Training and Test Datasets
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create Sequential Model Using Tensorflow Keras
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

# Creating Loss Function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the Model Designed Earlier
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics = ['accuracy'])

# Training and Validation
# The Model.fit method adjusts the model parameters to minimize the loss:
model.fit(x_train, y_train, epochs = 5)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)



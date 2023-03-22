import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential

#Regular Fully Connect Neural Network using Demographicc data

def fix_random_seed(seed):
    """ Setting the random seed of various libraries """
    try:
        np.random.seed(seed)
    except NameError:
        print("Warning: Numpy is not imported. Setting the seed for Numpy failed.")
    try:
        tf.random.set_seed(seed)
    except NameError:
        print("Warning: TensorFlow is not imported. Setting the seed for TensorFlow failed.")
    try:
        random.seed(seed)
    except NameError:
        print("Warning: random module is not imported. Setting the seed for random failed.")


# Fixing the random seed
fix_random_seed(4321)
print("TensorFlow version: {}".format(tf.__version__))


K.clear_session() # Making sure we are clearing out the TensorFlow graph

# Defining Model A with the Sequential API
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu')
    Dense(16, activation='relu'),
    Dense(20, activation='softmax')
])







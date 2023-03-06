"""
Gabriel Q, 2023
    python3 InceptionNetv1.py --seed random_seed compile

"""

import argparse
import tensorflow as tf
import tensorflow_hub as hub
import random
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Dense, Concatenate, Flatten, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import numpy as np
import tensorflow.keras.backend as K
import pickle
from tensorflow.keras.models import load_model, Model


parser = argparse.ArgumentParser()
parser.add_argument('--seed' , type=int, required=True)
parser.add_argument('compile')
args = parser.parse_args()



def fix_random_seed(seed):
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

def cpu_gpu_configuration():
    # Check gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            print("Couldn't set memory_growth")
            pass
# First Component on InceptionNet1
def stem(inp):
    conv1 = Conv2D(64, (7,7), strides=(1,1), activation='relu', padding='same')(inp)
    maxpool2 = MaxPool2D((3,3), strides=(2,2), padding='same')(conv1)
    lrn3 = Lambda(lambda x: tf.nn.local_response_normalization(x))(maxpool2)

    conv4 = Conv2D(64, (1,1), strides=(1,1), padding='same')(lrn3)
    conv5 = Conv2D(192, (3,3), strides=(1,1), activation='relu', padding='same')(conv4)
    lrn6 = Lambda(lambda x: tf.nn.local_response_normalization(x))(conv5)

    maxpool7 = MaxPool2D((3,3), strides=(1,1), padding='same')(lrn6)

    return maxpool7

def inception(inp, n_filters):

    # 1x1 layer
    # init argument defaults to glorot_uniform
    out1 = Conv2D(n_filters[0][0], (1,1), strides=(1,1), activation='relu', padding='same')(inp)

    # 1x1 followed by 3x3
    out2_1 = Conv2D(n_filters[1][0], (1,1), strides=(1,1), activation='relu', padding='same')(inp)
    out2_2 = Conv2D(n_filters[1][1], (3,3), strides=(1,1), activation='relu', padding='same')(out2_1)

    # 1x1 followed by 5x5
    out3_1 = Conv2D(n_filters[2][0], (1,1), strides=(1,1), activation='relu', padding='same')(inp)
    out3_2 = Conv2D(n_filters[2][1], (5,5), strides=(1,1), activation='relu', padding='same')(out3_1)

    # 3x3 (pool) followed by 1x1
    out4_1 = MaxPool2D((3,3), strides=(1,1), padding='same')(inp)
    out4_2 = Conv2D(n_filters[3][0], (1,1), strides=(1,1), activation='relu', padding='same')(out4_1)

    out = Concatenate(axis=-1)([out1, out2_2, out3_2, out4_2])
    return out

def aux_out(inp,name=None):
    avgpool1 = AvgPool2D((5,5), strides=(3,3), padding='valid')(inp)
    conv1 = Conv2D(128, (1,1), activation='relu', padding='same')(avgpool1)
    flat = Flatten()(conv1)
    dense1 = Dense(1024, activation='relu')(flat)
    aux_out = Dense(200, activation='softmax', name=name)(dense1)
    return aux_out

def inception_v1():
    """Defining and compiling Inception Net v1"""
    K.clear_session()
    inp = Input(shape=(56,56,3)) #check size
    stem_out = stem(inp)
    inc_3a = inception(stem_out, [(64,), (96, 128), (16, 32), (32,)])
    inc_3b = inception(inc_3a, [(128,), (128, 192), (32, 96), (64,)])

    maxpool = MaxPool2D((3, 3), strides=(2, 2), padding='same')(inc_3b)

    inc_4a = inception(maxpool, [(192,), (96, 208), (16, 48), (64,)])
    inc_4b = inception(inc_4a, [(160,), (112, 224), (24, 64), (64,)])

    aux_out1 = aux_out(inc_4a, name='aux1')

    inc_4c = inception(inc_4b, [(128,), (128, 256), (24, 64), (64,)])
    inc_4d = inception(inc_4c, [(112,), (144, 288), (32, 64), (64,)])
    inc_4e = inception(inc_4d, [(256,), (160, 320), (32, 128), (128,)])

    maxpool = MaxPool2D((3, 3), strides=(2, 2), padding='same')(inc_4e)

    aux_out2 = aux_out(inc_4d, name='aux2')

    inc_5a = inception(maxpool, [(256,), (160, 320), (32, 128), (128,)])
    inc_5b = inception(inc_5a, [(384,), (192, 384), (48, 128), (128,)])
    avgpool1 = AvgPool2D((7, 7), strides=(1, 1), padding='valid')(inc_5b)

    flat_out = Flatten()(avgpool1)
    out_main = Dense(200, activation='softmax', name='final')(flat_out)

    model = Model(inputs=inp, outputs=[out_main, aux_out1, aux_out2])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model



if __name__=="__main__":
    random_seed = args.seed
    fix_random_seed(random_seed)
    cpu_gpu_configuration()
    model = inception_v1()
    model.summary()
    if args.compile:
        model.compile()







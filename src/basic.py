import tensorflow as tf
#from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPool2D

def top_model(model) :
    top_model = Sequential()
    top_model.add(model)
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(2, activation='tanh'))
    
    return top_model

def AlexNet() :
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPool2D(pool_size = (3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size = (3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size = (3, 3), strides=(2, 2)))
    
    final_model = top_model(model)

    return final_model

def VGG(num_layer) :
    if num_layer == 16 :
        model = tf.keras.applications.VGG16(include_top=False, weights=None, pooling='max')
    elif num_layer == 19 :
        model = tf.keras.applications.VGG19(include_top=False, weights=None, pooling='max')


    final_model = top_model(model)
    
    return final_model

def ResNet(num_layer) :
    if num_layer == 50 :
        model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling='max')
    if num_layer == 101 :
        model = tf.keras.applications.ResNet101(include_top=False, weights=None, pooling='max')
    if num_layer == 152 :
        model = tf.keras.applications.ResNet152(include_top=False, weights=None, pooling='max')

    final_model = top_model(model)
    
    return final_model

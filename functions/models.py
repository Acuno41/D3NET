# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization


def cnn_network(height, width, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)

	model = Sequential()
	model.add(BatchNormalization(input_shape=inputShape))
	model.add(Convolution2D(16, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 3, 3, activation='relu')) 
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, activation='relu')) 
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(.25))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(1, activation='linear'))
	return model

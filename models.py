
"""Defines AbstractModel class that wraps a Keras model and trains on a supplied generator 
that yields (batch_data, batch_target) pairs. AbstractModels are saveable and loadable. Also 
implements some example recurrent and convolutional models."""

import numpy as np

from keras.layers import Input, Dense, Dropout, Reshape, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, SpatialDropout2D, Cropping2D
from keras.layers import LSTM, GRU, TimeDistributedDense, Embedding

from keras.models import Model, load_model
from keras.optimizers import adadelta
from keras.callbacks import ProgbarLogger, RemoteMonitor, ReduceLROnPlateau, ModelCheckpoint

from constants import *



class AbstractModel(object):
	def __init__(self, window_size, vector_size, lr=0.1):
		self.window_size = window_size
		self.vector_size = vector_size
		self.lr = lr
		self.model = self.build_model()


	def train(self, generator, epochs=20, checkpoint=None):
		callbacks = []
		callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
                  patience=3, min_lr=0.001))
		if checkpoint is not None:
			callbacks.append(ModelCheckpoint(filepath=checkpoint, verbose=1, save_best_only=False))

		generator.model = self
		self.model.fit_generator(generator.train(),
                nb_epoch=epochs,
                samples_per_epoch=generator.samples_per_epoch,
                verbose=2,
                validation_data=generator.val(),
                nb_val_samples=generator.val_samples,
                callbacks=callbacks,
                nb_worker=1,
            )

	def predict(self, test_data):
		return self.model.predict(test_data)

	def build_model(self):
		return None

	def save(self, weights_file):
		self.model.save(weights_file)

	@classmethod
	def load(self, weights_file, window_size, vector_size):

		model = load_model(weights_file)

		dnn = self(window_size, vector_size)
		dnn.model = model

		return dnn




# Takes in a recurrent sequence of vectors of length vector_size, with window_size vectors
# Predicts a single value (might represent predicting note volume or note duration)

class SequenceToValueRNN(AbstractModel):
	def __init__(self, *args, **kwargs):
		super(SequenceToValueRNN, self).__init__(*args, **kwargs)

	def build_model(self):

		input_img = Input(shape=(self.window_size, self.vector_size))

		x = GRU(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True) (input_img)
		x = GRU(128, dropout_W=0.2, dropout_U=0.2, return_sequences=False) (input_img)
		x = Dense(16, activation="sigmoid") (x)
		x = Dense(1, activation="linear") (x)

		model = Model(input_img, x)
		model.summary()

		model.compile(optimizer=adadelta(lr=0.3, clipvalue=10), loss='mse', metrics=['mae'])

		return model




# Takes in a recurrent sequence of vectors of length vector_size, with window_size vectors
# Predicts a classification of length vector_size. Might represent predicting a categorical pitch.

class SequenceToClassificationRNN(AbstractModel):
	def __init__(self, *args, **kwargs):
		super(SequenceToClassificationRNN, self).__init__(*args, **kwargs)

	def build_model(self):

		input_img = Input(shape=(self.window_size, self.vector_size))

		x = GRU(256, dropout_W=0.2, dropout_U=0.2, return_sequences=True) (input_img)
		x = GRU(128, dropout_W=0.2, dropout_U=0.2, return_sequences=False) (x)
		x = Dense(64, activation="relu") (x)
		x = Dense(self.vector_size, activation="softmax") (x)

		model = Model(input_img, x)
		model.summary()

		model.compile(optimizer=adadelta(lr=0.04, clipvalue=10), loss='categorical_crossentropy', 
			metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

		return model







# Takes in a convolution image of size (window_size, vector_size)
# Predicts a classification of length vector_size. Might represent predicting a categorical pitch.


class VGGNetModel(AbstractModel):

	def __init__(self, *args, **kwargs):
		super(VGGNetModel, self).__init__(*args, **kwargs)
	
	def build_model(self):
		input_img = Input(shape=(self.window_size, self.vector_size))
		map_size = 96

		x = Reshape((1, self.window_size, self.vector_size))(input_img)
		x = Convolution2D(map_size/8, 3, 12, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/4, 5, 3, activation='relu', border_mode='same')(x)
		x = SpatialDropout2D(p=0.4)(x)
		x = link1 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/4, 3, 12, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/4, 5, 3, activation='relu', border_mode='same')(x)
		x = SpatialDropout2D(p=0.4)(x)
		x = link2 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/2, 3, 12, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/4, 5, 3, activation='relu', border_mode='same')(x)
		x = Dropout(p=0.3)(x)
		x = link3 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/8, 3, 12, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/8, 5, 3, activation='relu', border_mode='same')(x)
		x = Dropout(p=0.3)(x)
		x = link4 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Flatten()(x)
		x = Dense(32, activation='sigmoid')(x)
		x = Dropout(0.15)(x)
		x = Dense(self.vector_size, activation='softmax')(x)

		model = Model(input_img, x)
		model.compile(optimizer=adadelta(lr=0.02, clipvalue=10), loss='binary_crossentropy', 
			metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
		model.summary()

		return model

if __name__ == "__main__":
	model = VGGNetEncoder(256)






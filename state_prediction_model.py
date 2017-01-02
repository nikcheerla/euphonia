#code for data generation from heatmaps

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import glob, sys, os, random, time, logging, threading

from sklearn.cross_validation import train_test_split
from scipy import stats
import progressbar

from representations import NoteMatrix, StateMatrix, ExpandedStateMatrix, NoteList
from generators import AbstractTrainSplitGenerator
from models import AbstractModel

from utils import sample
from constants import *

from keras.layers import LSTM, GRU, TimeDistributedDense, Input, Dense
from keras.models import Model
from keras.optimizers import adadelta

import IPython



"""2-layer GRU RNN model (implements AbstractModel from models.py) that predicts next state 
from window of previous states"""

class StatePredictionRNN(AbstractModel):
	def __init__(self, *args, **kwargs):
		super(StatePredictionRNN, self).__init__(*args, **kwargs)

	def build_model(self):

		input_img = Input(shape=(self.window_size, self.vector_size))

		x = GRU(128, dropout_W=0.25, dropout_U=0.25, return_sequences=True) (input_img)
		#x = TimeDistributedDense(96) (x)
		x = GRU(96, dropout_W=0.2, dropout_U=0.2, return_sequences=False) (x)
		x = Dense(64, activation="sigmoid") (x)
		x = Dense(self.vector_size, activation="softmax") (x)

		model = Model(input_img, x)
		model.summary()

		model.compile(optimizer=adadelta(lr=0.001, clipvalue=100), loss='mse', 
			metrics=['mae'])

		return model



"""Generator that yields train_test batches, based on AbstractTrainSplitGenerator from 
generators.py. It uses ExpandedStateMatrix to create a representation of a music file, then takes
a random position and returns the window and the target prediction for the next vector in the window."""

class StatePredictionGenerator(AbstractTrainSplitGenerator):

	def __init__(self, window_size, **kwargs):
		super(StatePredictionGenerator, self).__init__(window_size, **kwargs)

	def gen_sample_pair(self, files_list):
		music_file = random.choice(files_list)
		if self.data is None or random.randint(0, 20) == 1:
			self.data = ExpandedStateMatrix.load(music_file)

		data = self.data
		x = random.randint(0, len(data) - self.window_size - 1)

		window = (data[x:(x + self.window_size), :])/127.0 #Scale windows and predictions to 0-1 range
		target = (data[x + self.window_size, :])/127.0

		return window, target


#Create generator and model, then train model on generator

generator = StatePredictionGenerator(window_size=50, samples_per_epoch=1000, batch_size=50)
model = StatePredictionRNN(window_size=50, vector_size=ExpandedStateMatrix.vector_size)

while True:
	model.train(generator, epochs=8, checkpoint="results/state_pred_model.h5")

	#Use trained model to read in song and predict next states
	song = ExpandedStateMatrix.load("music/23-ballade.mid")
	
	for idx in range(50, 200):
		window = song[(idx - 50):idx]/127.0

		#predicts next state given window of previous states
		target_pred = model.predict(np.array([window]))[0]
		target_pred = (127.0*target_pred).astype(int)

		#Append new predicted state to song
		song[idx, :] = target_pred

	#Save song predictions to result files
	plt.imsave("results/state_pred_model.jpg", song[0:200].swapaxes(0, 1), cmap="hot_r")
	ExpandedStateMatrix.save(song[0:200], "results/state_pred_model.mid")

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

from keras.layers import LSTM, GRU, TimeDistributedDense, Input, Dense, Embedding
from keras.models import Model
from keras.optimizers import adadelta

import IPython



"""2-layer Embedding RNN model (implements AbstractModel from models.py) that predicts next pitch 
from sequence of previous pitches. Previous pitches played are represented as integer values, which
are turned to length-32 vectors by the Embedding layer (word2vec style). """

class EmbeddedSequencePredictClass(AbstractModel):
	def __init__(self, *args, **kwargs):
		super(EmbeddedSequencePredictClass, self).__init__(*args, **kwargs)

	def build_model(self):

		input_img = Input(shape=(self.window_size,))

		x = Embedding(self.vector_size, 32, dropout=0.2) (input_img)
		x = GRU(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True) (x)
		x = GRU(128, dropout_W=0.2, dropout_U=0.2, return_sequences=False) (x)
		x = Dense(64, activation="relu") (x)
		x = Dense(self.vector_size, activation="softmax") (x)

		model = Model(input_img, x)
		model.summary()

		model.compile(optimizer=adadelta(lr=0.08, clipvalue=1000), loss='categorical_crossentropy', 
			metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

		return model





"""Generator that yields train_test batches, based on AbstractTrainSplitGenerator from 
generators.py. It uses NoteList to get the list of notes, then takes
a random position and returns the pitch sequence history and the target pitch to be predicted."""

class PitchGuessingGenerator(AbstractTrainSplitGenerator):

	def __init__(self, window_size, **kwargs):
		super(PitchGuessingGenerator, self).__init__(window_size, **kwargs)

	def gen_sample_pair(self, files_list):
		music_file = random.choice(files_list)
		if self.data is None or random.randint(0, 10) == 1:
			self.data = NoteList.load(music_file)

		data = self.data
		x = random.randint(0, len(data) - self.window_size - 1)

		#input is sequential integers representing pitch, to be used in embedding
		window = (data[x:(x + self.window_size), 1]).astype(int)  
		target_idx = (data[x + self.window_size, 1]).astype(int)
		target = np.zeros(upperBound)
		target[target_idx] = 1 #categorical prediction for target

		return window, target




#Create generator and model, then train model on generator

generator = PitchGuessingGenerator(window_size=50, samples_per_epoch=1000, batch_size=5)
model = EmbeddedSequencePredictClass(window_size=50, vector_size=upperBound)

while True:
	model.train(generator, epochs=8, checkpoint="results/pitch_guess_model.h5")

	#Use trained model to read in song and predict next states
	song = NoteList.load("music/beethoven_opus10_1.mid")
	
	for idx in range(50, 200):
		window = song[(idx - 50):idx, 1]
		target_pred = model.predict(np.array([window]))[0]

		print (target_pred)
		target_idx = sample(target_pred, temperature=0.2)
		song[idx, 1] = target_idx

	NoteList.save(song[0:200], file_name="results/pitch_guess_song.mid")

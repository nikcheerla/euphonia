





import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import glob, sys, os, random, time, logging, threading

from sklearn.cross_validation import train_test_split
import progressbar

from representations import NoteMatrix, StateMatrix, ExpandedStateMatrix
from constants import *

import IPython













# THREADSAFE DECORATORS

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g















"""AbstractGenerator implements a train and validation generator based on a list of train
and validation MIDI files. The mechanisms of how to generate data/target pairs are left to child
classes. The verbose keyword allows a progressbar to be printed as training data is generated."""

class AbstractGenerator(object):
	def __init__(self, window_size, train_midi_list, val_midi_list, samples_per_epoch=500, 
			val_samples=30, batch_size=50, verbose=True):

		self.window_size = window_size
		self.train_midi_list = train_midi_list
		self.val_midi_list = val_midi_list
		self.samples_per_epoch = samples_per_epoch
		self.val_samples = val_samples
		self.batch_size = batch_size
		self.verbose = verbose
		self.progress_val = None
		self.data = None

	def progressbar(self):
		self.progress = progressbar.ProgressBar(max_value=self.samples_per_epoch)
		self.progress_val = 0

	def update_progress(self, sample_num):
		if self.verbose:
			if self.progress_val == None:
				print ("\nStarting Training:")
				self.progressbar()

			self.progress_val += sample_num
			
			if self.progress_val >= self.samples_per_epoch:
				self.progress_val -= self.samples_per_epoch

			self.progress.update(self.progress_val)

	@threadsafe_generator
	def train(self):
		while True:
			batch_data, batch_target = [], []
			for i in range(0, self.batch_size):
				window, target = self.gen_sample_pair(self.train_midi_list)

				self.update_progress(1)
				
				batch_data.append(window)
				batch_target.append(target)
			yield np.array(batch_data), np.array(batch_target)

	@threadsafe_generator
	def val(self):
		while True:
			batch_data, batch_target = [], []
			for i in range(0, self.batch_size):
				window, target = self.gen_sample_pair(self.val_midi_list)
				
				batch_data.append(window)
				batch_target.append(target)

			yield np.array(batch_data), np.array(batch_target)

	def gen_sample_pair(self, files_list):

		# All generators must implement gen_sample pair -- returns a
		# pair (window, target) that represents a data/target pair
		raise NotImplementedError()












"""AbstractGenerator implements a train and validation generator based on a directory. 
It divides the directory into training and validation MIDI files in ratios based on the split argument.
The mechanisms of how to generate data/target pairs are left to child classes. """

class AbstractTrainSplitGenerator(AbstractGenerator):
	
	def __init__(self, window_size, directory='music/', split=0.15, **kwargs):
		files_list = glob.glob(directory + "/*.mid")
		train_list, val_list = train_test_split(files_list, test_size=split)
		super(AbstractTrainSplitGenerator, self).__init__(window_size, train_list, val_list, **kwargs)

	



# Euphonia

This is a repository designed to facilitate using deep neural networks to predict the structure of and generate music. 

Dependencies: 
- the latest versions of Keras (```pip install git+https://github.com/fchollet/keras```), and Theano.
- MIDI file parsing code (```pip install git+https://github.com/vishnubob/python-midi```)

Usage:
representations.py houses the MIDI music representation code. This code allows you to represent MIDI files in a variety of different "formats", which are 2D matrices. These matrices can be modified and saved back to MIDI files. For example, StateMatrix converts MIDI files to a 2D Numpy array format somewhat like this Youtube video (https://www.youtube.com/watch?v=xh3FMTff5nY). Examples of these representations are stored in the results/ folder. This data can then be used to train ML models.

generators.py houses general code for generating batches of "data" that is fed to a Keras model.

models.py houses abstract code for Keras models that learn from/train on data from a generator.

See pitch_guess_model.py and state_prediction_model.py for an example of how it all fits together! pitch_guess model tries to predict the pitch of the next note given the sequence of pitches of the previous 50 pitches, and it achieves a bit of success (an example generated piece is in the results folder), although it still sounds pretty bad. state_prediction_model.py tries to ambitiously predict the entire state beat by beat, and fails miserably ;)

The most useful part of this code is probably representations.py, as it puts MIDI 





"""Classes that represent MIDI music files using various 1 to 1 representations. Each class has a static
.load() method and a .save() method that allows you to read from a MIDI file to a data array,
and write a data array to a MIDI file."""




import midi
from utils import to_notes

import numpy as np
import glob, random, os, sys
import matplotlib.pyplot as plt
from itertools import chain
from constants import *

import IPython















"""State matrix with notes, locations, beats. Each timestep respresents one beat in the MIDI file.
Whenever a new note is played, the velocity of the played note is added to the state at that position.
As long as the note is articulated, a value 20 (min_volume) is added to the state, creating a "tail" that
extends for the duration the note is played. Run "python representations.py" and view "results/StateMatrix_example.png"
to see how this representation looks."""

class StateMatrix(object):

    @classmethod
    def load(self, midifile, carry_volume=False):
        pattern = midi.read_midifile(midifile)

        timeleft = [track[0].tick for track in pattern]

        posns = [0 for track in pattern]

        statematrix = []
        time = 0

        state = [[0,0] for x in range(span)]
        statematrix.append(state)
        while True:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                # Crossed a note boundary. Create a new state, defaulting to holding notes
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(span)]
                statematrix.append(state)

            for i in range(len(timeleft)):
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]

                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                            pass
                            # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-lowerBound] = [0, 0]
                            elif carry_volume:
                                state[evt.pitch-lowerBound] = [evt.velocity*volume_scale, evt.velocity*volume_scale]
                            else:
                                state[evt.pitch-lowerBound] = [min_volume, evt.velocity*volume_scale]

                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None

                if timeleft[i] is not None:
                    timeleft[i] -= 1

            if all(t is None for t in timeleft):
                break

            time += 1

        statematrix = np.array(statematrix)

        if carry_volume:
            return statematrix[:, :, 0]
        
        state2 = np.copy(statematrix[:, :, 0])

        for note in range(span):
            for i in range(0, len(statematrix)):
                if statematrix[i, note, 1] > 0:
                    state2[i, note] = max(statematrix[i, note, 1], min_volume + 1)

        return state2

    @classmethod
    def save(self, statematrix, file_name="example"):
        statematrix = np.asarray(statematrix)

        # We assume that values near the min_volume (0.75 to 1.25 multiplier) are articulated
        # Values below are silent, values above are played with the specified velocity

        statematrix[statematrix < int(min_volume*0.75)] = 0
        statematrix[np.logical_and(statematrix > int(min_volume*0.75), 
            statematrix < int(min_volume*1.25))] = min_volume


        state1 = np.copy(statematrix)
        state1[statematrix == min_volume] = 0

        state2 = np.copy(statematrix)
        state2[statematrix > min_volume] = 1

        statematrix = np.rollaxis(np.array([state1, state2]), 0, 3)

        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        
        span = upperBound-lowerBound
        tickscale = 55
        
        lastcmdtime = 0
        prevstate = [[0,0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):  
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] != 0:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] != 0:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=max(state[note, 1], min_volume), pitch=note+lowerBound))
                lastcmdtime = time
                
            prevstate = state
        
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        midi.write_midifile("{}".format(file_name), pattern)












"""Augments the previous state matrix with additional information about the total volume at a time step, 4-bit 
beat information, information about the tones played without regard to octave, and other useful data. 
Run "python representations.py" and view "results/StateMatrix_example.png" to see how this representation looks. """


class ExpandedStateMatrix(StateMatrix):

    vector_size = 114

    @classmethod
    def load(self, midifile):

        statemat1 = super(self, ExpandedStateMatrix).load(midifile)
        pad = np.zeros((len(statemat1), 4))

        statemat2 = np.zeros((len(statemat1), 12))

        for i in range(0, len(statemat1)):
            for note in range(0, span):
                if statemat1[i, note] != 0:
                    statemat2[i, note%12] = statemat1[i, note]

        statemat3 = np.zeros((len(statemat1), 4))
        for i in range(0, len(statemat1)):
            statemat3[i][0] = i%2 * min_volume
            statemat3[i][1] = (i//2)%2 * min_volume
            statemat3[i][2] = (i//4)%2 * min_volume
            statemat3[i][3] = (i//8)%2 * min_volume

        carry_volume = super(self, ExpandedStateMatrix).load(midifile, carry_volume=True)

        statemat4 = np.clip(np.sum(carry_volume/span*15, axis=1).reshape(len(statemat1), 1), 0, max_cumulative_volume)

        statemat5 = np.zeros((len(statemat1), 1))
        for i in range(0, len(carry_volume)):
            x = []
            for note in range(0, span):
                if carry_volume[i, note] != 0:
                    x.append(carry_volume[i, note])
            x = np.array(x)
            if len(x) >= 2:
                statemat5[i, 0] = np.std(x)*volume_scale

        statematrix = np.append(statemat1, statemat2, axis=1)
        statematrix = np.append(statematrix, pad, axis=1)
        statematrix = np.append(statematrix, statemat3, axis=1)
        statematrix = np.append(statematrix, pad, axis=1)
        statematrix = np.append(statematrix, statemat4, axis=1)
        statematrix = np.append(statematrix, pad, axis=1)
        statematrix = np.append(statematrix, statemat5, axis=1)

        return statematrix

    @classmethod
    def save(self, statematrix, file_name="example"):
        statematrix = statematrix[:, 0:span]
        return super(self, ExpandedStateMatrix).save(statematrix, file_name=file_name)
















"""Simple note list that stores the pitch, volume, duration, and tick (time between last note played and this note),
for each note in a song. Run "python representations.py" and view "results/StateMatrix_example.png" to see how 
this representation looks. """

class NoteList(object):

    vector_size = 4

    @classmethod
    def load(self, midifile, carry_volume=False):
        
        song = to_notes(midifile)
        
        return song

    @classmethod
    def save(self, statematrix, file_name="example.mid"):
        
        notes = statematrix
        notes = np.array(notes)

        pitches = []
        ticks = []
        velocities = []

        for note in notes:
            pitches.append(note[1])
            ticks.append(note[0])
            velocities.append(note[3])

            pitches.append(note[1])
            ticks.append(note[0] + note[2])
            velocities.append(0)

        idxs = np.argsort(np.array(ticks))

        ticks = np.array(ticks)[idxs]
        pitches = np.array(pitches)[idxs]
        velocities = np.array(velocities)[idxs]

        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)

        tickscale = 0.5
        lastcmdtime = 0

        for tick, pitch, velocity in zip(ticks, pitches, velocities):
            evt = midi.NoteOnEvent(tick=int((tick-lastcmdtime)*tickscale), velocity=int(velocity), pitch=int(pitch))
            track.append(evt)
            lastcmdtime = tick

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        #IPython.embed()

        midi.write_midifile("{}".format(file_name), pattern)





















"""Per-note matrix that stores each note pitch as a categorical variable in a sparse vector of size pitch_span.
Each note row also contains the volume, duration, and tick (time between last note played and this note).
Run "python representations.py" and view "results/StateMatrix_example.png" to see how this representation looks. """

class NoteMatrix(object):

    queue_len = 10
    duration_repeat = 10
    velocity_repeat = 5

    pos1 = span + queue_len
    pos2 = pos1 + duration_repeat
    pos3 = pos2 + velocity_repeat
    vector_size = pos3

    @classmethod
    def load(self, midifile, carry_volume=False):
        
        song = to_notes(midifile)
        
        notes = []
        for note_idx in range(0, len(song)):
            note = song[note_idx]
            tick, pitch, duration, velocity = note[0], note[1] - lowerBound, note[2], note[3]
            
            note_arr = np.zeros(span + self.queue_len + self.duration_repeat + self.velocity_repeat)

            note_arr[int(pitch)] = min(duration, max_duration)

            note_arr[span] = tick
            for i in range(0, min(self.queue_len, note_idx) + 1):
                note_arr[span + i] = min((tick - song[note_idx - i][0]), max_delta)

            note_arr[self.pos1:self.pos2] = min(duration, max_duration)
            note_arr[self.pos2:self.pos3] = min(max(velocity*volume_scale, min_volume), max_volume)
            notes.append(note_arr)

        notes = np.array(notes).astype(int)
        #print (notes.shape)
        return notes

    @classmethod
    def save(self, statematrix, file_name="example.mid"):
        
        notes = []
        tick = 0

        for note_arr in statematrix:
            pitch = np.argmax(note_arr[0:span]) + lowerBound
            duration = np.mean(note_arr[self.pos1:self.pos2])
            tick += note_arr[span + 1]
            velocity = np.mean(note_arr[self.pos2:self.pos3])

            notes.append([tick, pitch, duration, velocity])

        notes = np.array(notes)

        song = to_notes("music/28-04-prelude.mid")

        pitches = []
        ticks = []
        velocities = []

        for note in notes:
            pitches.append(note[1])
            ticks.append(note[0])
            velocities.append(note[3])

            pitches.append(note[1])
            ticks.append(note[0] + note[2])
            velocities.append(0)

        idxs = np.argsort(np.array(ticks))

        ticks = np.array(ticks)[idxs]
        pitches = np.array(pitches)[idxs]
        velocities = np.array(velocities)[idxs]

        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)

        tickscale = 0.5
        lastcmdtime = 0

        for tick, pitch, velocity in zip(ticks, pitches, velocities):
            evt = midi.NoteOnEvent(tick=int((tick-lastcmdtime)*tickscale), velocity=int(velocity), pitch=int(pitch))
            track.append(evt)
            lastcmdtime = tick

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        midi.write_midifile("{}".format(file_name), pattern)
















"""Utility function to load an array of data from the specified directory in a given representation.
Returns: array of shape (size, slice_len, vector_size) where vector_size varies based on the
representation. """

def load_data(directory, size=60000, slice_len=84, representation=StateMatrix):
    
    cache_file = "cache.{0}.size{1}.slice{2}.npy".format(directory, size, slice_len)
    print (cache_file)
    if os.path.exists(cache_file):
        return np.load(cache_file)

    songs = glob.glob(directory + "/*.mid")

    arr = []
    while len(arr) < size:
        song = representation.load(random.choice(songs))
        x = 0
        while x+slice_len < len(song):
            arr.append(song[x:x+slice_len])
            if len(arr) % 50 == 0:
                print (len(arr))
            x += slice_len

    X_train = np.array(arr)[0:size]
    print (X_train.shape)
    np.save(cache_file, X_train)

    return X_train














if __name__ == "__main__":
    song = StateMatrix.load("music/23-ballade.mid")
    plt.imsave("results/StateMatrix_example.png", song.swapaxes(0, 1), cmap=plt.get_cmap("hot_r"))
    StateMatrix.save(song, file_name="results/StateMatrix_recoded.mid")

    song = ExpandedStateMatrix.load("music/23-ballade.mid")
    plt.imsave("results/ExpandedStateMatrix_example.png", song.swapaxes(0, 1), cmap=plt.get_cmap("hot_r"))
    ExpandedStateMatrix.save(song, file_name="results/ExpandedStateMatrix_recoded.mid")

    song = NoteMatrix.load("music/23-ballade.mid")
    plt.imsave("results/NoteMatrix_example.png", song.swapaxes(0, 1), cmap=plt.get_cmap("hot_r"))
    NoteMatrix.save(song, file_name="results/NoteMatrix_recoded.mid")

    song = NoteList.load("music/23-ballade.mid")
    plt.imsave("results/NoteList_example.png", song.swapaxes(0, 1), cmap=plt.get_cmap("hot_r"))
    NoteList.save(song, file_name="results/NoteList_recoded.mid")












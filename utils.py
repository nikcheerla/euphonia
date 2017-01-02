

import numpy as np
import midi
from midi import NoteEvent, NoteOnEvent, NoteOffEvent
import IPython





def is_note_on(e):
    return isinstance(e, NoteOnEvent) and e.velocity > 0

def is_note_off(e):
    return isinstance(e, NoteOffEvent) or (isinstance(e, NoteOnEvent) and e.velocity == 0)

def to_notes(midi_file):

        pattern = midi.read_midifile(midi_file)
        
        
        # list for all notes
        notes = []
        # dictionaries for storing the last onset and velocity per pitch
        note_onsets = {}
        note_velocities = {}
        for track in pattern:
            # get a list with note events
            note_events = [e for e in track]
            # process all events
            cum_tick = 0
            for e in note_events:
                cum_tick += e.tick

                # if it's a note on event with a velocity > 0,
                if is_note_on(e):
                    # save the onset time and velocity
                    note_onsets[e.pitch] = cum_tick
                    note_velocities[e.pitch] = e.velocity
                # if it's a note off event or a note on with a velocity of 0,
                elif is_note_off(e):
                    duration = cum_tick - note_onsets[e.pitch]
                    
                    notes.append((note_onsets[e.pitch], e.pitch,
                                  cum_tick - note_onsets[e.pitch],
                                  note_velocities[e.pitch]))

        # sort the notes and convert to numpy array
        notes.sort()
        notes = np.asarray(notes, dtype=int)

        return notes


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
    


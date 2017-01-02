"""Crawls all files from the piano-midi archive and saves them to the music directory."""


import re, urllib, time, sys, subprocess
def getChildLinks(link):
    links = [];
    start_time = time.time();
    try:
        for ee in re.findall('''href=["'](.[^"']+)["']''', urllib.urlopen(link).read(), re.I):
			links.append(ee)
    except:
        #print "Invalid link type"
        pass
    return links


artist_links = ["http://www.piano-midi.de/albeniz.htm",
	"http://www.piano-midi.de/bach.htm",
	"http://www.piano-midi.de/balak.htm",
	"http://www.piano-midi.de/beeth.htm",
	"http://www.piano-midi.de/borodin.htm",
	"http://www.piano-midi.de/brahms.htm",
	"http://www.piano-midi.de/burgm.htm",
	"http://www.piano-midi.de/chopin.htm",
	"http://www.piano-midi.de/clementi.htm",
	"http://www.piano-midi.de/debuss.htm",
	"http://www.piano-midi.de/godowsky.htm",
	"http://www.piano-midi.de/grana.htm",
	"http://www.piano-midi.de/grieg.htm",
	"http://www.piano-midi.de/haydn.htm",
	"http://www.piano-midi.de/liszt.htm",
	"http://www.piano-midi.de/mendelssohn.htm",
	"http://www.piano-midi.de/moszkowski.htm",
	"http://www.piano-midi.de/mozart.htm", 
	"http://www.piano-midi.de/muss.htm", 
	"http://www.piano-midi.de/rach.htm",
	"http://www.piano-midi.de/ravel.htm",
	"http://www.piano-midi.de/schub.htm",
	"http://www.piano-midi.de/schum.htm",
	"http://www.piano-midi.de/sinding.htm",
	"http://www.piano-midi.de/tschai.htm"]


music_links = []
for artist_link in artist_links:
	for midi_link in getChildLinks(artist_link):
		if midi_link[-4:] == ".mid":
			music_links.append(midi_link)

print ("MIDI Files: ", len(music_links))

for i, link in enumerate(music_links):
	print (i)
	link2 = "http://www.piano-midi.de/" + link
	subprocess.call(["wget", link2])



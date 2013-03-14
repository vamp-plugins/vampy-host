import sys
import os

sys.path.append(os.getcwd())

import numpy as np
print np.__version__

import matplotlib.pyplot as plt
import scikits.audiolab as al

import vampyhost as vh

#from melscale import melscale
#from melscale import initialize
# from melscale import *
#import pyRealTime

#deal with an audio file
wavfile = 'test-mono.wav'
# wavfile = '4sample-stereo-ny.wav'

af = al.Sndfile(wavfile)

nchannels = af.channels

print "Samplerate: ", af.samplerate
print "Number of channels: ", nchannels
print "Number of samples (frames): ", af.nframes

rt = vh.realtime(4, 70)

#test RealTime Object
for i in [0, 1, 2]:
    if i == 0:
        rtl = []
    rtl.append(vh.realtime())
    print ">>>>>RealTime's method: ", rtl[i].values()


class feature_example():
    def __init__(self):
        self.hasTimestamp
        self.timestamp
        self.values
        self.label

pluginlist = vh.enumeratePlugins()
for i, n in enumerate(pluginlist):
    print i, ":", n

pluginKey = pluginlist[0]  # try the first plugin listed

retval = vh.getLibraryPath(pluginKey)
print pluginKey
print retval

print vh.getPluginCategory(pluginKey)
print vh.getOutputList(pluginKey)
handle = vh.loadPlugin(pluginKey, af.samplerate)


print "\n\nPlugin handle: ", handle
print "Output list of: ", pluginKey, "\n", vh.getOutputList(handle)

# initialise: pluginhandle, channels, stepSize, blockSize
if vh.initialise(handle, nchannels, 1024, 1024):
    print "Initialise succeeded"
else:
    print "Initialise failed!"
    exit(1)

# should return a realtime object
rt = vh.frame2RealTime(100000, 22050)
print rt

assert type(rt) == type(vh.realtime())

audio = af.read_frames(af.nframes)
audio = np.transpose(audio)

print "Gonna send", len(audio)

out = vh.process(handle, audio, rt)
print "OKEYDOKEY: Processed"

output = vh.getOutput(handle, 1)

print type(output)
print output
#print output[1].label

print "_______________OUTPUT TYPE_________:", type(out)
in_audio = np.frombuffer(audio, np.int16, -1, 0)
out_audio = np.frombuffer(out, np.float32, -1, 0)
plt.subplot(211)
plt.plot(in_audio)
plt.subplot(212)
plt.plot(out_audio)

plt.show()
#do some processing here

#buffer is a multichannel frame or a numpy array containing samples
#buffer = vh.frame(audiodata,stepSize,blockSize)

#output = vh.process(handle,buffer)

#output is a list of list of features

vh.unloadPlugin(handle)
vh.unloadPlugin(handle)  # test if it chrashes...

print vh.getOutputList(handle)

#cases:
#buffer = blockSize : evaluate
#buffer > blockSize : enframe and zeropad
#return:
#oneSamplePerStep, FixedSamplerate : can return numpy array
#variableSamplerate : list of featres only

#print dir(vampyhost)

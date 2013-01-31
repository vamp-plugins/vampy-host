
import sys
import os

sys.path.append(os.getcwd())

import scikits.audiolab as al;

#from melscale import melscale
#from melscale import initialize
from pylab import *
# from melscale import *
from numpy import *
from pylab import *
from time import *

from vampyhost import *
import vampyhost
import vampyhost as vh
#import pyRealTime
#from pyRealTime import *

#deal with an audio file
wavfile='test.wav'

wavdata, samplerate, format = al.wavread(wavfile);

print "samplerate: ",samplerate
print "number of samples (frames): ",wavdata.size

audio = wavdata.transpose()

channels = audio.size
print "channels: ",channels

#!!! continue with this lark

rt=realtime(4,70)

#test RealTime Object
for i in [0,1,2] :
	if (i==0) : rtl=[]
	rtl.append(realtime())
	print ">>>>>RealTime's method: ", rtl[i].values()


class feature_example():
	def __init__(self):
		self.hasTimestamp
		self.timestamp
		self.values
		self.label

pluginlist = vh.enumeratePlugins()
for i,n in enumerate(pluginlist) : print i,":",n
pluginKey=pluginlist[0]; # try the first plugin listed

retval = vh.getLibraryPath(pluginKey)
print pluginKey
print retval

print vh.getPluginCategory(pluginKey)
print vh.getOutputList(pluginKey)
handle = vh.loadPlugin(pluginKey,samplerate);
print "\n\nPlugin handle: ",handle

print "Output list of: ",pluginKey,"\n",vh.getOutputList(handle)

#initialise: pluginhandle, channels, stepSize, blockSize
vh.initialise(handle,1,1024,1024)

rt=frame2RealTime(100000,22050)
print type(rt)

out=vh.process(handle,audio,rt)
output = vh.getOutput(handle,1);

print type(output)
print output
#print output[1].label

print "_______________OUTPUT TYPE_________:",type(out)
in_audio = frombuffer(audio,int16,-1,0)
out_audio = frombuffer(out,float32,-1,0)
subplot(211)
plot(in_audio)
subplot(212)
plot(out_audio)

show()
#do some processing here

#buffer is a multichannel frame or a numpy array containing samples
#buffer = vh.frame(audiodata,stepSize,blockSize)

#output = vh.process(handle,buffer)

#output is a list of list of features

vh.unloadPlugin(handle);
vh.unloadPlugin(handle); # test if it chrashes...

print vh.getOutputList(handle)

#cases:
#buffer = blockSize : evaluate
#buffer > blockSize : enframe and zeropad
#return:
#oneSamplePerStep, FixedSamplerate : can return numpy array
#variableSamplerate : list of featres only

#print dir(vampyhost)

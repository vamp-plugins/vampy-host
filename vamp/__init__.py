'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

###!!! todo: move all the real code out of __init__.py

import vampyhost
import numpy

def listPlugins():
    return vampyhost.listPlugins()

def framesFromArray(arr, stepSize, frameSize):
    """Generate a list of frames of size frameSize, extracted from the input array arr at stepSize intervals"""
    # presumably such a function exists in many places, but I need practice
    assert(stepSize > 0)
    if arr.ndim == 1: # turn 1d into 2d array with 1 channel
        arr = numpy.reshape(arr, (1, arr.shape[0]))
    assert(arr.ndim == 2)
    n = arr.shape[1]
    i = 0
    while (i < n):
        frame = arr[:, i : i + frameSize]
        w = frame.shape[1]
        if (w < frameSize):
            pad = numpy.zeros((frame.shape[0], frameSize - w))
            frame = numpy.concatenate((frame, pad), 1)
        yield frame
        i = i + stepSize


def loadAndConfigureFor(data, sampleRate, key, parameters):
    plug = vampyhost.loadPlugin(key, sampleRate,
                                vampyhost.AdaptInputDomain +
                                vampyhost.AdaptChannelCount)

    plug.setParameterValues(parameters)

    stepSize = plug.getPreferredStepSize()
    blockSize = plug.getPreferredBlockSize()

    if blockSize == 0:
        blockSize = 1024
    if stepSize == 0:
        stepSize = blockSize ##!!! or blockSize/2, but check this with input domain adapter

    channels = 1
    if data.ndim > 1:
        channels = data.shape[0]

    plug.initialise(channels, stepSize, blockSize)
    return (plug, stepSize, blockSize)


def process(data, sampleRate, key, parameters = {}, outputs = []):
#!!! docstring

    plug, stepSize, blockSize = loadAndConfigureFor(data, sampleRate, key, parameters)

    plugOuts = plug.getOutputs()
    if plugOuts == []:
        return

    outIndices = dict(zip([o["identifier"] for o in plugOuts],
                          range(0, len(plugOuts))))  # id -> n

    for o in outputs:
        assert o in outIndices

    if outputs == []:
        outputs = [plugOuts[0]["identifier"]]

    ff = framesFromArray(data, stepSize, blockSize)
    fi = 0

    #!!! should we fill in the correct timestamps here?

    for f in ff:
        results = plug.processBlock(f, vampyhost.frame2RealTime(fi, sampleRate))
        # results is a dict mapping output number -> list of feature dicts
        for o in outputs:
            if outIndices[o] in results:
                for r in results[outIndices[o]]:
                    yield { o: r }
        fi = fi + stepSize

    results = plug.getRemainingFeatures()
    for o in outputs:
        if outIndices[o] in results:
            for r in results[outIndices[o]]:
                yield { o: r }

    plug.unload()


def selectFeaturesForOutput(output, features):
    for ff in features:
        if output in ff:
            for f in ff[output]:
                yield f

##!!!
##
## We could also devise a generator for the timestamps that need
## filling: provide the output type & rate and get back a timestamp
## generator
##
##!!!

# def timestampFeatures(sampleRate, stepSize, outputDescriptor, features):

#     n = 0
    
#     if outputDict.sampleType == vampyhost.OneSamplePerStep:
#         for True:
#             yield vampyhost.frame2RealTime(n * stepSize, sampleRate)
#             n = n + 1

#     elif outputDict.sampleType == vampyhost.FixedSampleRate:
#         for True:
            


def collect(data, sampleRate, key, parameters = {}, output = ""):
    
    plug, stepSize, blockSize = loadAndConfigureFor(data, sampleRate, key, parameters)

    plugOuts = plug.getOutputs()
    if plugOuts == []:
        return

    outNo = -1
    for n, o in zip(range(0, len(plugOuts)), plugOuts):
        if output == "" or o["identifier"] == output:
            outNo = n
            break

    assert outNo >= 0 #!!! todo proper error reporting

    ff = framesFromArray(data, stepSize, blockSize)
    fi = 0

    #!!! todo!

    plug.unload()
    
    return {}



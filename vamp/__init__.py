'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

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

def process(data, samplerate, key, parameters = {}, outputs = []):
#!!! docstring
    plug = vampyhost.loadPlugin(key, samplerate,
                                vampyhost.AdaptInputDomain +
                                vampyhost.AdaptChannelCount)

    plugOuts = plug.getOutputs()

    outIndices = dict(zip([o["identifier"] for o in plugOuts],
                          range(0, len(plugOuts))))  # id -> n

    for o in outputs:
        assert o in outIndices

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
    ff = framesFromArray(data, stepSize, blockSize)
    fi = 0

    for f in ff:
        results = plug.processBlock(f, vampyhost.frame2RealTime(fi, samplerate))
        # results is a dict mapping output number -> list of feature dicts
        if outputs == []:
            if 0 in results:
                for r in results[0]:
                    yield r
        else:
            for o in outputs:
                if outIndices[o] in results:
                    for r in results[outIndices[o]]:
                        yield { o: r }
        fi = fi + stepSize

    ##!!! now getRemainingFeatures

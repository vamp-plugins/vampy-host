'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

def listPlugins():
    return vampyhost.listPlugins()

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


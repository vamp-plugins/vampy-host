'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import load
import frames

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
    
#     if outputDict.sampleType == vampyhost.ONE_SAMPLE_PER_STEP:
#         for True:
#             yield vampyhost.frame2RealTime(n * stepSize, sampleRate)
#             n = n + 1

#     elif outputDict.sampleType == vampyhost.FIXED_SAMPLE_RATE:
#         for True:
            


def collect(data, sampleRate, key, parameters = {}, output = ""):
    
    plug, stepSize, blockSize = load.loadAndConfigureFor(data, sampleRate, key, parameters)

    plugOuts = plug.getOutputs()
    if plugOuts == []:
        return

    outNo = -1
    for n, o in zip(range(0, len(plugOuts)), plugOuts):
        if output == "" or o["identifier"] == output:
            outNo = n
            break

    assert outNo >= 0 #!!! todo proper error reporting

    ff = frames.framesFromArray(data, stepSize, blockSize)
    fi = 0

    #!!! todo!

    plug.unload()
    
    return {}



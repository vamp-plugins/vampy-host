'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import load
import frames

def select_features_for_output(output, features):
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

# def timestampFeatures(sample_rate, step_size, outputDescriptor, features):

#     n = 0
    
#     if outputDict.sampleType == vampyhost.ONE_SAMPLE_PER_STEP:
#         for True:
#             yield vampyhost.frame_to_realtime(n * step_size, sample_rate)
#             n = n + 1

#     elif outputDict.sampleType == vampyhost.FIXED_SAMPLE_RATE:
#         for True:
            


def collect(data, sample_rate, key, parameters = {}, output = ""):
    
    plug, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    plug_outs = plug.get_outputs()
    if plug_outs == []:
        return

    outNo = -1
    for n, o in zip(range(0, len(plug_outs)), plug_outs):
        if output == "" or o["identifier"] == output:
            outNo = n
            break

    assert outNo >= 0 #!!! todo proper error reporting

    ff = frames.frames_from_array(data, step_size, block_size)
    fi = 0

    #!!! todo!

    plug.unload()
    
    return {}



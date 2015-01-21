'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import frames
import load

def load_and_query(data, sampleRate, key, parameters):
    plug, stepSize, blockSize = load.load_and_configure(data, sampleRate, key, parameters)
    plugOuts = plug.get_outputs()
    outIndices = dict(zip([o["identifier"] for o in plugOuts],
                          range(0, len(plugOuts))))  # id -> n
    return plug, stepSize, blockSize, outIndices
    

def process_multiple_outputs(data, sampleRate, key, outputs, parameters = {}):
#!!! docstring

    plug, stepSize, blockSize, outIndices = load_and_query(data, sampleRate, key, parameters)

    for o in outputs:
        assert o in outIndices

    ff = frames.frames_from_array(data, stepSize, blockSize)
    fi = 0

    for f in ff:
        results = plug.process_block(f, vampyhost.frame_to_realtime(fi, sampleRate))
        # results is a dict mapping output number -> list of feature dicts
        for o in outputs:
            outix = outIndices[o]
            if outix in results:
                for r in results[outix]:
                    yield { o: r }
        fi = fi + stepSize

    results = plug.get_remaining_features()
    for o in outputs:
        outix = outIndices[o]
        if outix in results:
            for r in results[outix]:
                yield { o: r }

    plug.unload()

def process(data, sampleRate, key, output = "", parameters = {}):
#!!! docstring

    plug, stepSize, blockSize, outIndices = load_and_query(data, sampleRate, key, parameters)

    if output == "":
        outix = 0
    else:
        assert output in outIndices
        outix = outIndices[output]
    
    ff = frames.frames_from_array(data, stepSize, blockSize)
    fi = 0

    for f in ff:
        results = plug.process_block(f, vampyhost.frame_to_realtime(fi, sampleRate))
        # results is a dict mapping output number -> list of feature dicts
        if outix in results:
            for r in results[outix]:
                yield r
        fi = fi + stepSize

    results = plug.get_remaining_features()
    if outix in results:
        for r in results[outix]:
            yield r

    plug.unload()


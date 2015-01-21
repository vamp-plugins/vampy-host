'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import frames
import load

def process(data, sampleRate, key, parameters = {}, outputs = []):
#!!! docstring

    plug, stepSize, blockSize = load.loadAndConfigureFor(data, sampleRate, key, parameters)

    plugOuts = plug.getOutputs()
    if plugOuts == []:
        return

    outIndices = dict(zip([o["identifier"] for o in plugOuts],
                          range(0, len(plugOuts))))  # id -> n

    for o in outputs:
        assert o in outIndices

    if outputs == []:
        outputs = [plugOuts[0]["identifier"]]

    ff = frames.framesFromArray(data, stepSize, blockSize)
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


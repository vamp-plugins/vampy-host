'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import frames
import load

def load_and_query(data, sample_rate, key, parameters):
    plug, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)
    plug_outs = plug.get_outputs()
    out_indices = dict([(o["identifier"], o["output_index"]) for o in plug_outs])
    return plug, step_size, block_size, out_indices
    

def process_multiple_outputs(data, sample_rate, key, outputs, parameters = {}):
#!!! docstring

    plug, step_size, block_size, out_indices = load_and_query(data, sample_rate, key, parameters)

    for o in outputs:
        assert o in out_indices

    ff = frames.frames_from_array(data, step_size, block_size)
    fi = 0

    for f in ff:
        results = plug.process_block(f, vampyhost.frame_to_realtime(fi, sample_rate))
        # results is a dict mapping output number -> list of feature dicts
        for o in outputs:
            outix = out_indices[o]
            if outix in results:
                for r in results[outix]:
                    yield { o: r }
        fi = fi + step_size

    results = plug.get_remaining_features()
    for o in outputs:
        outix = out_indices[o]
        if outix in results:
            for r in results[outix]:
                yield { o: r }

    plug.unload()

def process(data, sample_rate, key, output = "", parameters = {}):
#!!! docstring

    plug, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    if output == "":
        out = plug.get_output(0)
    else:
        out = plug.get_output(output)

    outix = out["output_index"]
    
    ff = frames.frames_from_array(data, step_size, block_size)
    fi = 0

    for f in ff:
        results = plug.process_block(f, vampyhost.frame_to_realtime(fi, sample_rate))
        # results is a dict mapping output number -> list of feature dicts
        if outix in results:
            for r in results[outix]:
                yield r
        fi = fi + step_size

    results = plug.get_remaining_features()
    if outix in results:
        for r in results[outix]:
            yield r

    plug.unload()


'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import frames
import load


def process_frames_with_plugin(ff, sample_rate, step_size, plugin, outputs):

    out_indices = dict([(id, plugin.get_output(id)["output_index"]) for id in outputs])
    plugin.reset()
    fi = 0

    for f in ff:
        timestamp = vampyhost.frame_to_realtime(fi, sample_rate)
        results = plugin.process_block(f, timestamp)
        # results is a dict mapping output number -> list of feature dicts
        for o in outputs:
            ix = out_indices[o]
            if ix in results:
                for r in results[ix]:
                    yield { o: r }
        fi = fi + step_size

    results = plugin.get_remaining_features()
    for o in outputs:
        ix = out_indices[o]
        if ix in results:
            for r in results[ix]:
                yield { o: r }


def process(data, sample_rate, key, output = "", parameters = {}):
#!!! docstring

    plugin, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    if output == "":
        output = plugin.get_output(0)["identifier"]

    ff = frames.frames_from_array(data, step_size, block_size)

    for r in process_frames_with_plugin(ff, sample_rate, step_size, plugin, [output]):
        yield r[output]
    
    plugin.unload()


def process_multiple_outputs(data, sample_rate, key, outputs, parameters = {}):
#!!! docstring

    plugin, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    ff = frames.frames_from_array(data, step_size, block_size)

    for r in process_frames_with_plugin(ff, sample_rate, step_size, plugin, outputs):
        yield r

    plugin.unload()


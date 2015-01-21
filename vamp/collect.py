'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import load
import process
import frames


def timestamp_features(sample_rate, step_size, output_desc, features):
    n = -1
    if output_desc["sample_type"] == vampyhost.ONE_SAMPLE_PER_STEP:
        for f in features:
            n = n + 1
            t = vampyhost.frame_to_realtime(n * step_size, sample_rate)
            f["timestamp"] = t
            yield f
    elif output_desc["sample_type"] == vampyhost.FIXED_SAMPLE_RATE:
        output_rate = output_desc["sample_rate"]
        for f in features:
            if "has_timestamp" in f:
                n = int(f["timestamp"].to_float() * output_rate + 0.5)
            else:
                n = n + 1
            f["timestamp"] = vampyhost.RealTime('seconds', float(n) / output_rate)
            yield f
    else:
        for f in features:
            yield f


def process_and_fill_timestamps(data, sample_rate, key, output, parameters = {}):

    plugin, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    if output == "":
        output_desc = plugin.get_output(0)
        output = output_desc["identifier"]
    else:
        output_desc = plugin.get_output(output)

    ff = frames.frames_from_array(data, step_size, block_size)

    results = process.process_frames_with_plugin(ff, sample_rate, step_size, plugin, [output])

    selected = [ r[output] for r in results ]

    stamped = timestamp_features(sample_rate, step_size, output_desc, selected)

    for s in stamped:
        yield s

    plugin.unload()
        

def collect(data, sample_rate, key, output, parameters = {}):
    return process_and_fill_timestamps(data, sample_rate, key, output, parameters)

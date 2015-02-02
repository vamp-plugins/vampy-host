'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import load
import process
import frames

import numpy as np

def get_feature_step_time(sample_rate, step_size, output_desc):
    if output_desc["sample_type"] == vampyhost.ONE_SAMPLE_PER_STEP:
        return vampyhost.frame_to_realtime(step_size, sample_rate)
    elif output_desc["sample_type"] == vampyhost.FIXED_SAMPLE_RATE:
        return vampyhost.RealTime('seconds', 1.0 / output_desc["sample_rate"])
    else:
        return 1

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

def fill_timestamps(results, sample_rate, step_size, output_desc):

    output = output_desc["identifier"]
    
    selected = [ r[output] for r in results ]

    stamped = timestamp_features(sample_rate, step_size, output_desc, selected)

    for s in stamped:
        yield s

def deduce_shape(output_desc):
    if output_desc["has_duration"]:
        return "individual"
    if output_desc["sample_type"] == vampyhost.VARIABLE_SAMPLE_RATE:
        return "individual"
    if not output_desc["has_fixed_bin_count"]:
        return "individual"
    if output_desc["bin_count"] == 0:
        return "individual"
    if output_desc["bin_count"] == 1:
        return "vector"
    return "matrix"


def reshape(results, sample_rate, step_size, output_desc):

    output = output_desc["identifier"]
    shape = deduce_shape(output_desc)
    out_step = get_feature_step_time(sample_rate, step_size, output_desc)

    if shape == "vector":
        rv = ( out_step,
               np.array([r[output]["values"][0] for r in results]) )
    elif shape == "matrix":
        rv = ( out_step,
               np.array(
                   [[r[output]["values"][i] for r in results]
                    for i in range(0, output_desc["bin_count"])]) )
    else:
        rv = list(fill_timestamps(results, sample_rate, step_size, output_desc))

    return rv

        
def collect(data, sample_rate, key, output, parameters = {}):

    plugin, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    if output == "":
        output_desc = plugin.get_output(0)
        output = output_desc["identifier"]
    else:
        output_desc = plugin.get_output(output)

    ff = frames.frames_from_array(data, step_size, block_size)

    results = process.process_frames_with_plugin(ff, sample_rate, step_size, plugin, [output])

    rv = reshape(results, sample_rate, step_size, output_desc)

    plugin.unload()
    return rv

        
def collect_frames(ff, channels, sample_rate, step_size, key, output, parameters = {}):

    plug = vampyhost.load_plugin(key, sample_rate,
                                 vampyhost.ADAPT_INPUT_DOMAIN +
                                 vampyhost.ADAPT_BUFFER_SIZE +
                                 vampyhost.ADAPT_CHANNEL_COUNT)

    plug.set_parameter_values(parameters)

    if not plug.initialise(channels, step_size, block_size):
        raise "Failed to initialise plugin"

    if output == "":
        output_desc = plugin.get_output(0)
        output = output_desc["identifier"]
    else:
        output_desc = plugin.get_output(output)

    results = process.process_frames_with_plugin(ff, sample_rate, step_size, plugin, [output])

    rv = reshape(results, sample_rate, step_size, output_desc)

    plugin.unload()
    return rv


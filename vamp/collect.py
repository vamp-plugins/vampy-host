'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost
import load
import process

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


def collect(data, sample_rate, key, output, parameters = {}):

    plug, step_size, block_size = load.load_and_configure(data, sample_rate, key, parameters)

    if output == "":
        out = plug.get_output(0)
    else:
        out = plug.get_output(output)

    plug.unload()
        
    results = process.process(data, sample_rate, key, output, parameters)
        
    return timestamp_features(sample_rate, step_size, out, results)




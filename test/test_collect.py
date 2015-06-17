
import vamp
import numpy as np
import vamp.frames as fr

plugin_key = "vamp-test-plugin:vamp-test-plugin"
plugin_key_freq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100.0

# Throughout this file we have the assumption that the plugin gets run with a
# blocksize of 1024, and with a step of 1024 for the time-domain version or 512
# for the frequency-domain one. That is certainly expected to be the norm for a
# plugin like this that declares no preference, and the Python Vamp module is
# expected to follow the norm

blocksize = 1024
eps = 1e-6

def input_data(n):
    # start at 1, not 0 so that all elts are non-zero
    return np.arange(n) + 1    

def test_collect_runs_at_all():
    buf = input_data(blocksize * 10)
    step, results = vamp.collect(buf, rate, plugin_key, "input-timestamp")
    assert results != []

##!!! add test for default output
    
def test_collect_one_sample_per_step():
    buf = input_data(blocksize * 10)
    step, results = vamp.collect(buf, rate, plugin_key, "input-timestamp")
    assert abs(float(step) - (1024.0 / rate)) < eps
    assert len(results) == 10
    for i in range(len(results)):
        # The timestamp should be the frame number of the first frame in the
        # input buffer
        expected = i * blocksize
        actual = results[i]
        assert actual == expected

def test_collect_fixed_sample_rate():
    buf = input_data(blocksize * 10)
    step, results = vamp.collect(buf, rate, plugin_key, "curve-fsr")
    assert abs(float(step) - 0.4) < eps
    assert len(results) == 10
    for i in range(len(results)):
        assert abs(results[i] - i * 0.1) < eps

def test_collect_fixed_sample_rate_2():
    buf = input_data(blocksize * 10)
    step, results = vamp.collect(buf, rate, plugin_key, "curve-fsr-timed")
    assert abs(float(step) - 0.4) < eps
    assert len(results) == 10
    for i in range(len(results)):
        assert abs(results[i] - i * 0.1) < eps
        
def test_collect_variable_sample_rate():
    buf = input_data(blocksize * 10)
    results = vamp.collect(buf, rate, plugin_key, "curve-vsr")
    assert len(results) == 10
    i = 0
    for r in results:
        print("timestamp = " + str(r["timestamp"]))
        assert r["timestamp"] == vamp.vampyhost.RealTime('seconds', i * 0.75)
        assert abs(r["values"][0] - i * 0.1) < eps
        i = i + 1

def test_collect_grid_one_sample_per_step():
    buf = input_data(blocksize * 10)
    step, results = vamp.collect(buf, rate, plugin_key, "grid-oss")
    assert abs(float(step) - (1024.0 / rate)) < eps
    assert len(results) == 10
    for i in range(len(results)):
        expected = np.array([ (j + i + 2.0) / 30.0 for j in range(0, 10) ])
        assert (abs(results[i] - expected) < eps).all()

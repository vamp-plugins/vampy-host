
import vamp
import numpy as np

testPluginKey = "vamp-test-plugin:vamp-test-plugin"
testPluginKeyFreq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100

# Throughout this file we have the assumption that the plugin gets run with a
# blocksize of 1024, and with a step of 1024 for the time-domain version or 512
# for the frequency-domain one. That is certainly expected to be the norm for a
# plugin like this that declares no preference, and the Python Vamp module is
# expected to follow the norm.

blocksize = 1024

def input_data(n):
    # start at 1, not 0 so that all elts are non-zero
    return np.arange(n) + 1    

def test_process_n():
    buf = input_data(blocksize)
    results = list(vamp.process(buf, rate, testPluginKey, "input-summary"))
    assert len(results) == 1

def test_process_freq_n():
    buf = input_data(blocksize)
    results = list(vamp.process(buf, rate, testPluginKeyFreq, "input-summary", {}))
    assert len(results) == 2 # one complete block starting at zero, one half-full

def test_process_default_output():
    # If no output is specified, we should get the first one (instants)
    buf = input_data(blocksize)
    results = list(vamp.process(buf, rate, testPluginKey, "", {}))
    assert len(results) == 10
    for i in range(len(results)):
        expectedTime = vamp.vampyhost.RealTime('seconds', i * 1.5)
        actualTime = results[i]["timestamp"]
        assert expectedTime == actualTime

def test_process_summary_param():
    buf = input_data(blocksize * 10)
    results = list(vamp.process(buf, rate, testPluginKey, "input-summary", { "produce_output": 0 }))
    assert len(results) == 0

def test_process_multi_summary_param():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKey, [ "input-summary" ], { "produce_output": 0 }))
    assert len(results) == 0

def test_process_summary_param_bool():
    buf = input_data(blocksize * 10)
    results = list(vamp.process(buf, rate, testPluginKey, "input-summary", { "produce_output": False }))
    assert len(results) == 0

def test_process_multi_summary_param_bool():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKey, [ "input-summary" ], { "produce_output": False }))
    assert len(results) == 0

def test_process_summary():
    buf = input_data(blocksize * 10)
    results = list(vamp.process(buf, rate, testPluginKey, "input-summary", {}))
    assert len(results) == 10
    for i in range(len(results)):
        #
        # each feature has a single value, equal to the number of non-zero elts
        # in the input block (which is all of them, i.e. the blocksize) plus
        # the first elt (which is i * blockSize + 1)
        #
        expected = blocksize + i * blocksize + 1
        actual = results[i]["values"][0]
        assert actual == expected

def test_process_multi_summary():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKey, [ "input-summary" ], {}))
    assert len(results) == 10
    for i in range(len(results)):
        #
        # each feature has a single value, equal to the number of non-zero elts
        # in the input block (which is all of them, i.e. the blocksize) plus
        # the first elt (which is i * blockSize + 1)
        #
        expected = blocksize + i * blocksize + 1
        actual = results[i]["input-summary"]["values"][0]
        assert actual == expected

def test_process_freq_summary():
    buf = input_data(blocksize * 10)
    results = list(vamp.process(buf, rate, testPluginKeyFreq, "input-summary", {}))
    assert len(results) == 20
    for i in range(len(results)):
        #
        # sort of as above, but much much subtler:
        #
        # * the input block is converted to frequency domain but then converted
        # back within the plugin, so the values being reported are time-domain
        # ones but with windowing and FFT shift
        # 
        # * the effect of FFT shift is that the first element in the
        # re-converted frame is actually the one that was at the start of the
        # second half of the original frame
        #
        # * and the last block is only half-full, so the "first" elt in that
        # one, which actually comes from just after the middle of the block,
        # will be zero
        #
        # * windowing does not affect the value of the first elt, because
        # (before fft shift) it came from the peak of the window shape where
        # the window value is 1
        #
        # * but windowing does affect the number of non-zero elts, because the
        # asymmetric window used has one value very close to zero in it
        #
        # * the step size (the increment in input value from one block to the
        # next) is only half the block size
        #
        expected = i * (blocksize/2) + blocksize/2 + 1   # "first" elt
        if (i == len(results)-1):
            expected = 0
        expected = expected + blocksize - 1              # non-zero elts
        actual = results[i]["values"][0]
        eps = 1e-6
        assert abs(actual - expected) < eps

def test_process_multi_freq_summary():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKeyFreq, [ "input-summary" ], {}))
    assert len(results) == 20
    for i in range(len(results)):
        expected = i * (blocksize/2) + blocksize/2 + 1   # "first" elt
        if (i == len(results)-1):
            expected = 0
        expected = expected + blocksize - 1              # non-zero elts
        actual = results[i]["input-summary"]["values"][0]
        eps = 1e-6
        assert abs(actual - expected) < eps

def test_process_timestamps():
    buf = input_data(blocksize * 10)
    results = list(vamp.process(buf, rate, testPluginKey, "input-timestamp", {}))
    assert len(results) == 10
    for i in range(len(results)):
        # The timestamp should be the frame number of the first frame in the
        # input buffer
        expected = i * blocksize
        actual = results[i]["values"][0]
        assert actual == expected

def test_process_multi_timestamps():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKey, [ "input-timestamp" ]))
    assert len(results) == 10
    for i in range(len(results)):
        # The timestamp should be the frame number of the first frame in the
        # input buffer
        expected = i * blocksize
        actual = results[i]["input-timestamp"]["values"][0]
        assert actual == expected

def test_process_freq_timestamps():
    buf = input_data(blocksize * 10)
    results = list(vamp.process(buf, rate, testPluginKeyFreq, "input-timestamp", {}))
    assert len(results) == 20
    for i in range(len(results)):
        # The timestamp should be the frame number of the frame just beyond
        # half-way through the input buffer
        expected = i * (blocksize/2) + blocksize/2
        actual = results[i]["values"][0]
        assert actual == expected

def test_process_multi_freq_timestamps():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKeyFreq, [ "input-timestamp" ], {}))
    assert len(results) == 20
    for i in range(len(results)):
        # The timestamp should be the frame number of the frame just beyond
        # half-way through the input buffer
        expected = i * (blocksize/2) + blocksize/2
        actual = results[i]["input-timestamp"]["values"][0]
        assert actual == expected

def test_process_multiple_outputs():
    buf = input_data(blocksize * 10)
    results = list(vamp.processMultipleOutputs(buf, rate, testPluginKey, [ "input-summary", "input-timestamp" ], {}))
    assert len(results) == 20
    si = 0
    ti = 0
    for r in results:
        assert "input-summary" in r or "input-timestamp" in r
        if "input-summary" in r:
            expected = blocksize + si * blocksize + 1
            actual = r["input-summary"]["values"][0]
            assert actual == expected
            si = si + 1
        if "input-timestamp" in r:
            expected = ti * blocksize
            actual = r["input-timestamp"]["values"][0]
            assert actual == expected
            ti = ti + 1

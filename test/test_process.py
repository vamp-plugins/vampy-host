
import vamp
import numpy as np

testPluginKey = "vamp-test-plugin:vamp-test-plugin"
testPluginKeyFreq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100

def test_process():
    buf = np.zeros(10240)
    results = vamp.process(buf, rate, testPluginKey, {}, [ "input-timestamp" ])
    print("results = " + str(list(results)))
    return True

def test_process_freq():
    buf = np.zeros(10240)
    results = vamp.process(buf, rate, testPluginKeyFreq, {}, [ "input-timestamp" ])
    print("results = " + str(list(results)))
    return True

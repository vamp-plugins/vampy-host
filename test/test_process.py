
import vamp
import numpy as np

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

rate = 44100

def test_process():
    buf = np.zeros(10240)
    results = vamp.process(buf, rate, testPluginKey, {}, ["instants","curve-fsr"])
    print("results = " + str(list(results)))
    return True

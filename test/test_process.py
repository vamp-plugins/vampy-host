
import vampyhost as vh
import numpy as np

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

##!!! todo: support for, and test for, correct version of test plugin (with parameter)

rate = 44100

def test_load_unload():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    plug.unload()
    try:
        plug.unload() # should throw but not crash
        assert(False)
    except AttributeError:
        pass

def test_get_set_parameter():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    value = plug.getParameterValue("produce_output")
    assert(value == 1.0)
    plug.setParameterValue("produce_output", 0.0)
    value = plug.getParameterValue("produce_output")
    assert(value == 0.0)
    
def test_process_without_initialise():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    try:
        plug.processBlock([[1,2,3,4]], vh.RealTime(0, 0))
        assert False
    except StandardError:
        pass

def test_process_input_format():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    plug.initialise(2, 4, 4) # channels, stepsize, blocksize
    result = plug.processBlock([[1,2,3,4],[5,6,7,8]], vh.RealTime(0, 0))
    result = plug.processBlock([np.array([1,2,3,4]),np.array([5,6,7,8])], vh.RealTime(0, 0))
    result = plug.processBlock(np.array([[1,2,3,4],[5,6,7,8]]), vh.RealTime(0, 0))
    try:
        # Wrong number of channels
        result = plug.processBlock(np.array([[1,2,3,4]]), vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    try:
        # Wrong number of samples per channel
        result = plug.processBlock(np.array([[1,2,3],[4,5,6]]), vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    try:
        # Differing numbers of samples per channel
        result = plug.processBlock(np.array([[1,2,3,4],[5,6,7]]), vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass

def test_process_output_1ch():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    plug.initialise(1, 2, 2)
    try:
        # Too many channels
        result = plug.processBlock([[3,4],[5,6]], vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    result = plug.processBlock([[3,3]], vh.RealTime(0, 0))
    assert result[8] == [ { "label" : "", "values" : np.array([5.0]) } ]
    result = plug.processBlock([[3,0]], vh.RealTime(0, 0))
    assert result[8] == [ { "label" : "", "values" : np.array([4.0]) } ]

def test_process_output_2ch():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    plug.initialise(2, 2, 2)
    try:
        # Too few channels
        result = plug.processBlock([[3,4]], vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    try:
        # Too many channels
        result = plug.processBlock([[3,4],[5,6],[7,8]], vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    result = plug.processBlock([[3,3],[4,4]], vh.RealTime(0, 0))
    assert (result[8][0]["values"] == np.array([5.0,6.0])).all()
    result = plug.processBlock([[3,0],[4,0]], vh.RealTime(0, 0))
    assert (result[8][0]["values"] == np.array([4.0,5.0])).all()

def test_process_output_3ch():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    plug.initialise(3, 2, 2)
    try:
        # Too few channels
        result = plug.processBlock([[3,4],[5,6]], vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    try:
        # Too many channels
        result = plug.processBlock([[3,4],[5,6],[7,8],[9,10]], vh.RealTime(0, 0))
        assert False
    except TypeError:
        pass
    result = plug.processBlock([[3,3],[4,4],[5,5]], vh.RealTime(0, 0))
    assert (result[8][0]["values"] == np.array([5.0,6.0,7.0])).all()
    result = plug.processBlock([[3,0],[4,0],[5,0]], vh.RealTime(0, 0))
    assert (result[8][0]["values"] == np.array([4.0,5.0,6.0])).all()


    

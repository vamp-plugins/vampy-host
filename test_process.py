
import vampyhost as vh

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

rate = 44100

def test_load_unload():
    plug = vh.loadPlugin(testPluginKey, rate)
    plug.unload()
    try:
        plug.unload() # should throw but not crash
        assert(False)
    except AttributeError:
        pass

def test_get_set_parameter():
    plug = vh.loadPlugin(testPluginKey, rate)
    value = plug.getParameter("produce_output")
    assert(value == 1.0)
    plug.setParameter("produce_output", 0.0)
    value = plug.getParameter("produce_output")
    assert(value == 0.0)
    
def test_process_without_initialise():
    plug = vh.loadPlugin(testPluginKey, rate)
    try:
        plug.process([[1,2,3,4]], vh.RealTime(0, 0))
        assert(False)
    except StandardError:
        pass

def test_process():
    plug = vh.loadPlugin(testPluginKey, rate)
    plug.initialise(1, 4, 4) # channels, stepsize, blocksize
    result = plug.process([[1,2,3,4]], vh.RealTime(0, 0))



import vampyhost as vh

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

rate = 44100

def test_load_unload():
    plug = vh.loadPlugin(testPluginKey, rate)
    vh.unloadPlugin(plug)
    try:
        vh.unloadPlugin(plug) # should throw but not crash
        assert(False)
    except AttributeError:
        pass

def test_process_without_initialise():
    plug = vh.loadPlugin(testPluginKey, rate)
    try:
        vh.process(plug, [[1,2,3,4]], vh.RealTime(0, 0))
        assert(False)
    except StandardError:
        pass

    

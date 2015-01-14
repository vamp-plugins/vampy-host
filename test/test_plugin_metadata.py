
import vampyhost as vh

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

rate = 44100

def test_getoutputlist():
    outputs = vh.getOutputsOf(testPluginKey)
    assert len(outputs) == 9
    assert "input-summary" in outputs

def test_inputdomain():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    assert plug.inputDomain == vh.TimeDomain

def test_info():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    assert plug.info["identifier"] == "vamp-test-plugin"
    
def test_parameterdescriptors():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    assert plug.parameters[0]["identifier"] == "produce_output"
    
def test_setparameter():
    plug = vh.loadPlugin(testPluginKey, rate, vh.AdaptNone)
    assert plug.parameters[0]["identifier"] == "produce_output"
    assert plug.parameters[0]["defaultValue"] == 1
    assert plug.getParameterValue("produce_output") == plug.parameters[0]["defaultValue"]
    assert plug.setParameterValue("produce_output", 0) == True
    assert plug.getParameterValue("produce_output") == 0
    assert plug.setParameterValues({ "produce_output": 1 }) == True
    assert plug.getParameterValue("produce_output") == 1
    try:
        plug.setParameterValue("produce_output", "fish")
        assert False
    except TypeError:
        pass
    try:
        plug.setParameterValue(4, 0)
        assert False
    except TypeError:
        pass
    try:
        plug.setParameterValue("steak", 0)
        assert False
    except StandardError:
        pass
    try:
        plug.getParameterValue(4)
        assert False
    except TypeError:
        pass
    try:
        plug.getParameterValue("steak")
        assert False
    except StandardError:
        pass
            

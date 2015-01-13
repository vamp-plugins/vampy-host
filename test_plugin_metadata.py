
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

    


import vampyhost as vh

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

testPluginKeyFreq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100

expectedVersion = 3

def test_plugin_exists():
    assert testPluginKey in vh.list_plugins()
    plug = vh.load_plugin(testPluginKey, rate, vh.ADAPT_NONE)
    assert "pluginVersion" in plug.info
    if plug.info["pluginVersion"] != expectedVersion:
        print("Test plugin version " + str(plug.info["pluginVersion"]) + " does not match expected version " + str(expectedVersion))
    assert plug.info["pluginVersion"] == expectedVersion

def test_plugin_exists_in_freq_version():
    assert testPluginKeyFreq in vh.list_plugins()

def test_getoutputlist():
    outputs = vh.get_outputs_of(testPluginKey)
    assert len(outputs) == 10
    assert "input-summary" in outputs

def test_inputdomain():
    plug = vh.load_plugin(testPluginKey, rate, vh.ADAPT_NONE)
    assert plug.inputDomain == vh.TIME_DOMAIN

def test_info():
    plug = vh.load_plugin(testPluginKey, rate, vh.ADAPT_NONE)
    assert plug.info["identifier"] == "vamp-test-plugin"
    
def test_parameterdescriptors():
    plug = vh.load_plugin(testPluginKey, rate, vh.ADAPT_NONE)
    assert plug.parameters[0]["identifier"] == "produce_output"
    
def test_setparameter():
    plug = vh.load_plugin(testPluginKey, rate, vh.ADAPT_NONE)
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
            

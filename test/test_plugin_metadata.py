
import vampyhost as vh

plugin_key = "vamp-test-plugin:vamp-test-plugin"

plugin_key_freq = "vamp-test-plugin:vamp-test-plugin-freq"

rate = 44100

expectedVersion = 3

def test_plugin_exists():
    assert plugin_key in vh.list_plugins()
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert "plugin_version" in plug.info
    if plug.info["plugin_version"] != expectedVersion:
        print("Test plugin version " + str(plug.info["plugin_version"]) + " does not match expected version " + str(expectedVersion))
    assert plug.info["plugin_version"] == expectedVersion

def test_plugin_exists_in_freq_version():
    assert plugin_key_freq in vh.list_plugins()

def test_getoutputlist():
    outputs = vh.get_outputs_of(plugin_key)
    assert len(outputs) == 10
    assert "input-summary" in outputs

def test_getoutputlist_2():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    outputs = plug.get_outputs()
    assert len(outputs) == 10

def test_get_output_by_id():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    out = plug.get_output("input-summary")
    assert "sample_type" in out
    assert out["output_index"] == 8
    try:
        out = plug.get_output("chops")
        assert False
    except StandardError:
        pass
    try:
        out = plug.get_output("")
        assert False
    except StandardError:
        pass

def test_get_output_by_index():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    out = plug.get_output(0)
    assert "sample_type" in out
    assert out["identifier"] == "instants"
    assert out["output_index"] == 0
    try:
        out = plug.get_output(20)
        assert False
    except StandardError:
        pass
    try:
        out = plug.get_output(-1)
        assert False
    except StandardError:
        pass
    try:
        out = plug.get_output(plug)
        assert False
    except StandardError:
        pass
    
def test_inputdomain():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.input_domain == vh.TIME_DOMAIN

def test_info():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.info["identifier"] == "vamp-test-plugin"
    
def test_parameterdescriptors():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.parameters[0]["identifier"] == "produce_output"
    
def test_setparameter():
    plug = vh.load_plugin(plugin_key, rate, vh.ADAPT_NONE)
    assert plug.parameters[0]["identifier"] == "produce_output"
    assert plug.parameters[0]["default_value"] == 1
    assert plug.get_parameter_value("produce_output") == plug.parameters[0]["default_value"]
    assert plug.set_parameter_value("produce_output", 0) == True
    assert plug.get_parameter_value("produce_output") == 0
    assert plug.set_parameter_values({ "produce_output": 1 }) == True
    assert plug.get_parameter_value("produce_output") == 1
    try:
        plug.set_parameter_value("produce_output", "fish")
        assert False
    except TypeError:
        pass
    try:
        plug.set_parameter_value(4, 0)
        assert False
    except TypeError:
        pass
    try:
        plug.set_parameter_value("steak", 0)
        assert False
    except StandardError:
        pass
    try:
        plug.get_parameter_value(4)
        assert False
    except TypeError:
        pass
    try:
        plug.get_parameter_value("steak")
        assert False
    except StandardError:
        pass
            

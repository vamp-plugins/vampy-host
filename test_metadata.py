
import vampyhost as vh

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

##!!! could use: plugin version

def test_enumerate():
    plugins = vh.enumeratePlugins()
    if testPluginKey not in plugins:
        print("Test plugin " + testPluginKey + " not installed or not returned by enumerate: can't run any tests without it")
    assert testPluginKey in plugins

def test_path():
    path = vh.getPluginPath()
    assert len(path) > 0

def test_getlibrary():
    lib = vh.getLibraryForPlugin(testPluginKey)
    assert lib != ""

def test_getoutputlist():
    outputs = vh.getOutputList(testPluginKey)
    assert len(outputs) == 8
    assert "curve-vsr" in outputs
    

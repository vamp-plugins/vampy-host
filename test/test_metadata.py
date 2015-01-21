
import vampyhost as vh

testPluginKey = "vamp-test-plugin:vamp-test-plugin"

##!!! could use: plugin version

def test_list():
    plugins = vh.list_plugins()
    if testPluginKey not in plugins:
        print("Test plugin " + testPluginKey + " not installed or not returned by enumerate: can't run any tests without it")
    assert testPluginKey in plugins

def test_path():
    path = vh.get_plugin_path()
    assert len(path) > 0

def test_getlibrary():
    lib = vh.get_library_for(testPluginKey)
    assert lib.find("vamp-test-plugin") >= 0
    try:
        lib = vh.get_library_for("not a well-formatted plugin key")
        assert False
    except TypeError:
        pass
    lib = vh.get_library_for("nonexistent-library:nonexistent-plugin")
    assert lib == ""
    

/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

//include for python extension module: must be first
#include <Python.h>

// define a unique API pointer 
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <vampyhost.h>

#define HAVE_NUMPY 1 // Required

//includes for vamp host
#include "vamp-hostsdk/Plugin.h"
#include "vamp-hostsdk/PluginHostAdapter.h"
#include "vamp-hostsdk/PluginChannelAdapter.h"
#include "vamp-hostsdk/PluginInputDomainAdapter.h"
#include "vamp-hostsdk/PluginLoader.h"

#include "PyTypeConversions.h"
#include "PyRealTime.h"

#include <iostream>
#include <fstream>
#include <set>
#include <sndfile.h>

#include <cstring>
#include <cstdlib>
#include <string>

#include <cmath>

using namespace std;
using namespace Vamp;

using Vamp::Plugin;
using Vamp::PluginHostAdapter;
using Vamp::RealTime;
using Vamp::HostExt::PluginLoader;

#define HOST_VERSION "1.1"

// structure for holding plugin instance data
struct PyPluginData
{
    PyPluginData(string k, Plugin *p, float rate) :
        key(k),
        plugin(p),
        inputSampleRate(rate),
        isInitialised(false),
        channels(0),
        blockSize(0),
        stepSize(0) {
    }
    
    string key;
    Plugin *plugin;
    float inputSampleRate;
    bool isInitialised;
    size_t channels;
    size_t blockSize;
    size_t stepSize;
};

/* MODULE HELPER FUNCTIONS */
PyDoc_STRVAR(xx_foo_doc, "Some description"); //!!!

//!!! nb "The CObject API is deprecated" https://docs.python.org/2/c-api/cobject.html

PyPluginData *
getPluginData(PyObject *pyPluginHandle)
{
    PyPluginData *pd = 0;
    if (PyCObject_Check(pyPluginHandle)) {
        pd = (PyPluginData *)PyCObject_AsVoidPtr(pyPluginHandle);
    }
    if (!pd || !pd->plugin) {
        PyErr_SetString(PyExc_AttributeError,
			"Invalid or already deleted plugin handle.");
        return 0;
    } else {
        return pd;
    }
}

static PyObject *
vampyhost_enumeratePlugins(PyObject *self, PyObject *args)
{
    PluginLoader *loader = PluginLoader::getInstance();
    vector<PluginLoader::PluginKey> plugins = loader->listPlugins();
    PyTypeConversions conv;
    return conv.PyValue_From_StringVector(plugins);
}

static PyObject *
vampyhost_getPluginPath(PyObject *self, PyObject *args)
{
    vector<string> path = PluginHostAdapter::getPluginPath();
    PyTypeConversions conv;
    return conv.PyValue_From_StringVector(path);
}

static string toPluginKey(PyObject *pyPluginKey)
{
    //convert to stl string
    string pluginKey(PyString_AS_STRING(pyPluginKey));

    //check pluginKey Validity
    string::size_type ki = pluginKey.find(':');
    if (ki == string::npos) {
	PyErr_SetString(PyExc_TypeError,
			"Plugin key must be of the form library:identifier");
       	return "";
    }

    return pluginKey;
}

static PyObject *
vampyhost_getLibraryFor(PyObject *self, PyObject *args)
{
    PyObject *pyPluginKey;

    if (!PyArg_ParseTuple(args, "S", &pyPluginKey)) {
	PyErr_SetString(PyExc_TypeError,
			"getLibraryPathForPlugin() takes plugin key (string) argument");
	return 0; }

    string pluginKey = toPluginKey(pyPluginKey);
    if (pluginKey == "") return 0;
    
    PluginLoader *loader = PluginLoader::getInstance();
    string path = loader->getLibraryPathForPlugin(pluginKey);
    PyObject *pyPath = PyString_FromString(path.c_str());
    return pyPath;
}

static PyObject *
vampyhost_getPluginCategory(PyObject *self, PyObject *args)
{
    PyObject *pyPluginKey;

    if (!PyArg_ParseTuple(args, "S", &pyPluginKey)) {
	PyErr_SetString(PyExc_TypeError,
			"getPluginCategory() takes plugin key (string) argument");
	return 0; }

    string pluginKey = toPluginKey(pyPluginKey);
    if (pluginKey == "") return 0;

    PluginLoader *loader = PluginLoader::getInstance();
    PluginLoader::PluginCategoryHierarchy
	category = loader->getPluginCategory(pluginKey);

    PyTypeConversions conv;
    return conv.PyValue_From_StringVector(category);
}

static PyObject *
vampyhost_getOutputList(PyObject *self, PyObject *args)
{
    PyObject *keyOrHandle;
    Plugin::OutputList outputs;

    if (!PyArg_ParseTuple(args, "O", &keyOrHandle)) {
	PyErr_SetString(PyExc_TypeError,
			"getOutputList() takes plugin handle (object) or plugin key (string) argument");
	return 0;
    }

    if (PyString_Check(keyOrHandle) ) {

        // we have a plugin key

        string pluginKey = toPluginKey(keyOrHandle);
        if (pluginKey == "") return 0;

        PluginLoader *loader = PluginLoader::getInstance();

        Plugin *plugin = loader->loadPlugin
            (pluginKey, 48000, PluginLoader::ADAPT_ALL_SAFE);
        if (!plugin) {
            string pyerr("Failed to load plugin: "); pyerr += pluginKey;
            PyErr_SetString(PyExc_TypeError,pyerr.c_str());
            return 0;
        }

        outputs = plugin->getOutputDescriptors();

        delete plugin;
        
    } else {

        // we have a loaded plugin handle
        
        PyPluginData *pd = getPluginData(keyOrHandle);
        if (!pd) return 0;

        outputs = pd->plugin->getOutputDescriptors();
    }

    PyObject *pyList = PyList_New(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
	PyObject *pyOutputId =
	    PyString_FromString(outputs[i].identifier.c_str());
	PyList_SET_ITEM(pyList, i, pyOutputId);
    }

    return pyList;
}

static PyObject *
vampyhost_loadPlugin(PyObject *self, PyObject *args)
{
    PyObject *pyPluginKey;
    float inputSampleRate;

    if (!PyArg_ParseTuple(args, "Sf",
			  &pyPluginKey,
			  &inputSampleRate)) {
	PyErr_SetString(PyExc_TypeError,
			"loadPlugin() takes plugin key (string) and sample rate (float) arguments");
	return 0; }

    string pluginKey = toPluginKey(pyPluginKey);
    if (pluginKey == "") return 0;

    PluginLoader *loader = PluginLoader::getInstance();

    Plugin *plugin = loader->loadPlugin(pluginKey, inputSampleRate,
                                        PluginLoader::ADAPT_ALL_SAFE);
    if (!plugin) {
	string pyerr("Failed to load plugin: "); pyerr += pluginKey;
	PyErr_SetString(PyExc_TypeError,pyerr.c_str());
	return 0;
    }

    PyPluginData *pd = new PyPluginData(pluginKey, plugin, inputSampleRate);
    return PyCObject_FromVoidPtr(pd, 0);
}

static PyObject *
vampyhost_unloadPlugin(PyObject *self, PyObject *args)
{
    PyObject *pyPluginHandle;

    if (!PyArg_ParseTuple(args, "O", &pyPluginHandle)) {
	PyErr_SetString(PyExc_TypeError,
			"unloadPlugin() takes plugin handle (object) argument");
	return 0;
    }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    /* Prevent repeated calls from causing segfault since it will fail
     * type checking the 2nd time: */
    PyCObject_SetVoidPtr(pyPluginHandle, 0);

    delete pd->plugin;
    delete pd;
    return pyPluginHandle;
}

static PyObject *
vampyhost_initialise(PyObject *self, PyObject *args)
{
    PyObject *pyPluginHandle;
    size_t channels, blockSize, stepSize;

    if (!PyArg_ParseTuple (args, "Onnn",  &pyPluginHandle,
			   (size_t) &channels,
			   (size_t) &stepSize,
			   (size_t) &blockSize))
    {
	PyErr_SetString(PyExc_TypeError,
			"initialise() takes plugin handle (object), channel count, step size, and block size arguments");
	return 0;
    }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    pd->channels = channels;
    pd->stepSize = stepSize;
    pd->blockSize = blockSize;

    if (!pd->plugin->initialise(channels, stepSize, blockSize)) {
        cerr << "Failed to initialise native plugin adapter with channels = " << channels << ", stepSize = " << stepSize << ", blockSize = " << blockSize << " and ADAPT_ALL_SAFE set" << endl;
	PyErr_SetString(PyExc_TypeError,
			"Plugin initialization failed");
	return 0;
    }

    pd->isInitialised = true;

    return Py_True;
}

static PyObject *
vampyhost_reset(PyObject *self, PyObject *args)
{
    PyObject *pyPluginHandle;

    if (!PyArg_ParseTuple (args, "O",  &pyPluginHandle))
    {
	PyErr_SetString(PyExc_TypeError,
			"initialise() takes plugin handle (object) argument");
	return 0;
    }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    if (!pd->isInitialised) {
        PyErr_SetString(PyExc_StandardError,
                        "Plugin has not been initialised");
        return 0;
    }
        
    pd->plugin->reset();
    return Py_True;
}

static PyObject *
vampyhost_getParameter(PyObject *self, PyObject *args)
{
    PyObject *pyPluginHandle;
    PyObject *pyParam;

    if (!PyArg_ParseTuple(args, "OS", &pyPluginHandle, &pyParam)) {
	PyErr_SetString(PyExc_TypeError,
			"getParameter() takes plugin handle (object) and parameter id (string) arguments");
	return 0; }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    float value = pd->plugin->getParameter(PyString_AS_STRING(pyParam));
    return PyFloat_FromDouble(double(value));
}

static PyObject *
vampyhost_setParameter(PyObject *self, PyObject *args)
{
    PyObject *pyPluginHandle;
    PyObject *pyParam;
    float value;

    if (!PyArg_ParseTuple(args, "OSf", &pyPluginHandle, &pyParam, &value)) {
	PyErr_SetString(PyExc_TypeError,
			"setParameter() takes plugin handle (object), parameter id (string), and value (float) arguments");
	return 0; }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    pd->plugin->setParameter(PyString_AS_STRING(pyParam), value);
    return Py_True;
}

static PyObject *
vampyhost_process(PyObject *self, PyObject *args)
{
    PyObject *pyPluginHandle;
    PyObject *pyBuffer;
    PyObject *pyRealTime;

    if (!PyArg_ParseTuple(args, "OOO",
			  &pyPluginHandle,	// C object holding a pointer to a plugin and its descriptor
			  &pyBuffer,			// Audio data
			  &pyRealTime)) {		// TimeStamp
	PyErr_SetString(PyExc_TypeError,
			"process() takes plugin handle (object), buffer (2D array of channels * samples floats) and timestamp (RealTime) arguments");
	return 0; }

    if (!PyRealTime_Check(pyRealTime)) {
	PyErr_SetString(PyExc_TypeError,"Valid timestamp required.");
	return 0; }

    if (!PyList_Check(pyBuffer)) {
	PyErr_SetString(PyExc_TypeError, "List of NumPy Array required for process input.");
        return 0;
    }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    if (!pd->isInitialised) {
	PyErr_SetString(PyExc_StandardError,
			"Plugin has not been initialised.");
	return 0;
    }

    int channels =  pd->channels;

    if (PyList_GET_SIZE(pyBuffer) != channels) {
        cerr << "Wrong number of channels: got " << PyList_GET_SIZE(pyBuffer) << ", expected " << channels << endl;
	PyErr_SetString(PyExc_TypeError, "Wrong number of channels");
        return 0;
    }

    float **inbuf = new float *[channels];

    PyTypeConversions typeConv;
    typeConv.setNumpyInstalled(true);

    cerr << "here!"  << endl;
    
    vector<vector<float> > data;
    for (int c = 0; c < channels; ++c) {
        PyObject *cbuf = PyList_GET_ITEM(pyBuffer, c);
        data.push_back(typeConv.PyValue_To_FloatVector(cbuf));
    }
    
    for (int c = 0; c < channels; ++c) {
        if (data[c].size() != pd->blockSize) {
            cerr << "Wrong number of samples on channel " << c << ": expected " << pd->blockSize << " (plugin's block size), got " << data[c].size() << endl;
            PyErr_SetString(PyExc_TypeError, "Wrong number of samples");
            return 0;
        }
        inbuf[c] = &data[c][0];
    }

    cerr << "no, here!"  << endl;

    RealTime timeStamp = *PyRealTime_AsRealTime(pyRealTime);

    cerr << "no no, here!"  << endl;

    Plugin::FeatureSet fs = pd->plugin->process(inbuf, timeStamp);

    delete[] inbuf;

    cerr << "no no no, here!"  << endl;
    
    PyTypeConversions conv;
    conv.setNumpyInstalled(true);
    
    PyObject *pyFs = PyDict_New();

    for (Plugin::FeatureSet::const_iterator fsi = fs.begin();
         fsi != fs.end(); ++fsi) {

        int fno = fsi->first;
        const Plugin::FeatureList &fl = fsi->second;

        if (!fl.empty()) {

            PyObject *pyFl = PyList_New(fl.size());

            for (int fli = 0; fli < (int)fl.size(); ++fli) {

                const Plugin::Feature &f = fl[fli];
                PyObject *pyF = PyDict_New();

                if (f.hasTimestamp) {
                    PyDict_SetItemString
                        (pyF, "timestamp", PyRealTime_FromRealTime(f.timestamp));
                }
                if (f.hasDuration) {
                    PyDict_SetItemString
                        (pyF, "duration", PyRealTime_FromRealTime(f.duration));
                }

                PyDict_SetItemString
                    (pyF, "label", PyString_FromString(f.label.c_str()));

                if (!f.values.empty()) {
                    PyDict_SetItemString
                        (pyF, "values", conv.FloatVector_To_PyArray(f.values));
                }

                PyList_SET_ITEM(pyFl, fli, pyF);
            }

            PyObject *pyN = PyInt_FromLong(fno);
            PyDict_SetItem(pyFs, pyN, pyFl);
        }
    }

    cerr << "no you fool, here!"  << endl;
    
    return pyFs;
}


// module methods table
static PyMethodDef vampyhost_methods[] = {

    {"listPlugins",	vampyhost_enumeratePlugins,	METH_NOARGS,
     xx_foo_doc},

    {"getPluginPath",	vampyhost_getPluginPath, METH_NOARGS,
     xx_foo_doc},

    {"getCategoryOf",	vampyhost_getPluginCategory, METH_VARARGS,
     xx_foo_doc},

    {"getLibraryFor",	vampyhost_getLibraryFor, METH_VARARGS,
     xx_foo_doc},

    {"getOutputsOf",	vampyhost_getOutputList, METH_VARARGS,
     xx_foo_doc},

    {"loadPlugin",	vampyhost_loadPlugin, METH_VARARGS,
     xx_foo_doc},

    {"getParameter",	vampyhost_getParameter, METH_VARARGS,
     xx_foo_doc},

    {"setParameter",	vampyhost_setParameter, METH_VARARGS,
     xx_foo_doc},

    {"initialise",	vampyhost_initialise, METH_VARARGS,
     xx_foo_doc},

    {"reset",	vampyhost_reset, METH_VARARGS,
     xx_foo_doc},

    {"process",	vampyhost_process, METH_VARARGS,
     xx_foo_doc},

    {"unloadPlugin",	vampyhost_unloadPlugin, METH_VARARGS,
     xx_foo_doc},

    {0,		0}		/* sentinel */
};

//Documentation for our new module
PyDoc_STRVAR(module_doc, "This is a template module just for instruction.");



/* Initialization function for the module (*must* be called initxx) */

//module initialization (includes extern C {...} as necessary)
PyMODINIT_FUNC
initvampyhost(void)
{
    PyObject *m;

    /* Finalize the type object including setting type of the new type
     * object; doing it here is required for portability to Windows
     * without requiring C++. */

    if (PyType_Ready(&RealTime_Type) < 0)
	return;

    /* Create the module and add the functions */
    m = Py_InitModule3("vampyhost", vampyhost_methods, module_doc);
    if (!m) return;

    import_array();

    PyModule_AddObject(m, "RealTime", (PyObject *)&RealTime_Type);
}

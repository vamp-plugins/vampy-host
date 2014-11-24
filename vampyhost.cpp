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
    Vamp::Plugin::FeatureSet output;
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
			"loadPlugin() takes plugin key (string) and sample rate (number) arguments");
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


/* INITIALISE PLUGIN */

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
			"Wrong input arguments: requires a valid plugin handle,channels,stepSize,blockSize.");
	return 0;
    }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    pd->channels = channels;
    pd->stepSize = stepSize;
    pd->blockSize = blockSize;

    if (!pd->plugin->initialise(channels, stepSize, blockSize)) {
        std::cerr << "Failed to initialise native plugin adapter with channels = " << channels << ", stepSize = " << stepSize << ", blockSize = " << blockSize << " and ADAPT_ALL_SAFE set" << std::endl;
	PyErr_SetString(PyExc_TypeError,
			"Plugin initialization failed.");
	return 0;
    }

    pd->isInitialised = true;

    return Py_True;
}

/* RUN PROCESS */

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
			"Required: plugin handle, buffer, timestmap.");
	return 0; }

    if (!PyRealTime_Check(pyRealTime)) {
	PyErr_SetString(PyExc_TypeError,"Valid timestamp required.");
	return 0; }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    if (!pd->isInitialised) {
	PyErr_SetString(PyExc_StandardError,
			"Plugin has not been initialised.");
	return 0;
    }

    int channels =  pd->channels;
//    int blockSize = pd->blockSize;

    if (!PyList_Check(pyBuffer)) {
	PyErr_SetString(PyExc_TypeError, "List of NumPy Array required for process input.");
        return 0;
    }

    if (PyList_GET_SIZE(pyBuffer) != channels) {
        std::cerr << "Wrong number of channels: got " << PyList_GET_SIZE(pyBuffer) << ", expected " << channels << std::endl;
	PyErr_SetString(PyExc_TypeError, "Wrong number of channels");
        return 0;
    }

    float **inbuf = new float *[channels];

    PyTypeConversions typeConv;
    typeConv.setNumpyInstalled(true);
    
    vector<vector<float> > data;
    for (int c = 0; c < channels; ++c) {
        PyObject *cbuf = PyList_GET_ITEM(pyBuffer, c);
        data.push_back(typeConv.PyArray_To_FloatVector(cbuf));
    }
    
    for (int c = 0; c < channels; ++c) {
        inbuf[c] = &data[c][0];
    }

    RealTime timeStamp = *PyRealTime_AsRealTime(pyRealTime);

    //Call process and store the output
    pd->output = pd->plugin->process(inbuf, timeStamp);

    /* TODO:  DO SOMETHONG WITH THE FEATURE SET HERE */
/// convert to appropriate python objects, reuse types and conversion utilities from Vampy ...

    delete[] inbuf;

    return 0; //!!! Need to return actual features!

}

/* GET / SET OUTPUT */

//getOutput(plugin,outputNo)
static PyObject *
vampyhost_getOutput(PyObject *self, PyObject *args) {

    PyObject *pyPluginHandle;
//	PyObject *pyBuffer;
//	PyObject *pyRealTime;
    PyObject *pyOutput;

    if (!PyArg_ParseTuple(args, "OO",
			  &pyPluginHandle,	// C object holding a pointer to a plugin and its descriptor
			  &pyOutput)) {		// Output reference
	PyErr_SetString(PyExc_TypeError,
			"Required: plugin handle, buffer, timestmap.");
	return 0; }

    PyPluginData *pd = getPluginData(pyPluginHandle);
    if (!pd) return 0;

    unsigned int outputNo = (unsigned int) PyInt_AS_LONG(pyOutput);

    //Get output list: but we don't need it
    //Plugin::FeatureList features = pd->output[outputNo];

    size_t outLength = pd->output[outputNo].size();

    //New PyList for the featurelist
    PyObject *pyFeatureList = PyList_New(outLength);

    for (size_t i = 0; i < outLength; ++i) {
	// Test:
	/*
	  XxoObject *pyFeature = PyObject_New(XxoObject, &Xxo_Type);
	  if (pyFeature == 0) break; //return 0;

	  pyFeature->x_attr = 0;
	  pyFeature->feature = &pd->output[outputNo][i];

	  PyList_SET_ITEM(pyFeatureList,i,(PyObject*)pyFeature);
	*/
    }

    Py_INCREF(pyFeatureList);
    return pyFeatureList;

// EXPLAIN WHAT WE NEED TO DO HERE:
// We have the block output in pd->output
// FeatureSet[output] -> [Feature[x]] -> Feature.hasTimestamp = v
// Vamp::Plugin::FeatureSet output; = pd->output
// typedef std::vector<Feature> FeatureList;
// typedef std::map<int, FeatureList> FeatureSet; // key is output no

    // 	THIS IS FOR OUTPUT id LOOKUP LATER
    //     Plugin::OutputList outputs = plugin->getOutputDescriptors();
    //
    // if (outputs.size()<1) {
    // 	string pyerr("Plugin has no output: "); pyerr += pluginKey;
    // 	PyErr_SetString(PyExc_TypeError,pyerr.c_str());
    // 	return 0;
    // }
    //
    // //New list object
    // PyObject *pyList = PyList_New(outputs.size());
    //
    //     for (size_t i = 0; i < outputs.size(); ++i) {
    // 	PyObject *pyOutputId =
    // 	PyString_FromString(outputs[i].identifier.c_str());
    // 	PyList_SET_ITEM(pyList,i,pyOutputId);
    //     }

}




/* List of functions defined in this module */
//module methods table
static PyMethodDef vampyhost_methods[] = {

    {"enumeratePlugins",	vampyhost_enumeratePlugins,	METH_NOARGS,
     xx_foo_doc},

    {"getPluginPath",	vampyhost_getPluginPath, METH_NOARGS,
     xx_foo_doc},

    {"getLibraryForPlugin",	vampyhost_getLibraryFor, METH_VARARGS,
     xx_foo_doc},

    {"getPluginCategory",	vampyhost_getPluginCategory, METH_VARARGS,
     xx_foo_doc},

    {"getOutputList",	vampyhost_getOutputList, METH_VARARGS,
     xx_foo_doc},

    {"loadPlugin",	vampyhost_loadPlugin, METH_VARARGS,
     xx_foo_doc},

    {"process",	vampyhost_process, METH_VARARGS,
     xx_foo_doc},

    {"unloadPlugin",	vampyhost_unloadPlugin, METH_VARARGS,
     xx_foo_doc},

    {"initialise",	vampyhost_initialise, METH_VARARGS,
     xx_foo_doc},

    {"getOutput",	vampyhost_getOutput, METH_VARARGS,
     xx_foo_doc},

    /* Add RealTime Module Methods */
/*
    {"frame2RealTime",	(PyCFunction)RealTime_frame2RealTime,	METH_VARARGS,
     PyDoc_STR("frame2RealTime((int64)frame, (uint32)sampleRate ) -> returns new RealTime object from frame.")},

    {"realtime",	(PyCFunction)RealTime_new,		METH_VARARGS,
     PyDoc_STR("realtime() -> returns new RealTime object")},
*/
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
//	PyModule_AddObject(m, "Real_Time", (PyObject *)&RealTime_Type);

    /* Create the module and add the functions */
    m = Py_InitModule3("vampyhost", vampyhost_methods, module_doc);
    if (!m) return;

    import_array();

    // PyModule_AddObject(m, "realtime", (PyObject *)&RealTime_Type);

}

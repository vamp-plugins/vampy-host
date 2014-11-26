/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    VampyHost

    Use Vamp audio analysis plugins in Python

    Gyorgy Fazekas and Chris Cannam
    Centre for Digital Music, Queen Mary, University of London
    Copyright 2008-2014 Queen Mary, University of London
  
    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Except as contained in this notice, the names of the Centre for
    Digital Music; Queen Mary, University of London; and the authors
    shall not be used in advertising or otherwise to promote the sale,
    use or other dealings in this Software without prior written
    authorization.
*/

#include "PyRealTime.h"

//include for python extension module: must be first
#include <Python.h>

// define a unique API pointer 
#define PY_ARRAY_UNIQUE_SYMBOL VAMPYHOST_ARRAY_API
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
struct PyPluginObject
{
    PyObject_HEAD
    string *key;
    Plugin *plugin;
    float inputSampleRate;
    bool isInitialised;
    size_t channels;
    size_t blockSize;
    size_t stepSize;
    static PyPluginObject *create_internal();
};

PyAPI_DATA(PyTypeObject) Plugin_Type;
#define PyPlugin_Check(v) PyObject_TypeCheck(v, &Plugin_Type)
    
static void
PyPluginObject_dealloc(PyPluginObject *self)
{
    cerr << "PyPluginObject_dealloc" << endl;
    delete self->key;
    delete self->plugin;
    PyObject_Del(self);
}

PyDoc_STRVAR(xx_foo_doc, "Some description"); //!!!

PyPluginObject *
getPluginObject(PyObject *pyPluginHandle)
{
    cerr << "getPluginObject" << endl;

    PyPluginObject *pd = 0;
    if (PyPlugin_Check(pyPluginHandle)) {
        pd = (PyPluginObject *)pyPluginHandle;
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
vampyhost_enumeratePlugins(PyObject *self, PyObject *)
{
    cerr << "vampyhost_enumeratePlugins" << endl;

    PluginLoader *loader = PluginLoader::getInstance();
    vector<PluginLoader::PluginKey> plugins = loader->listPlugins();
    PyTypeConversions conv;
    return conv.PyValue_From_StringVector(plugins);
}

static PyObject *
vampyhost_getPluginPath(PyObject *self, PyObject *)
{
    cerr << "vampyhost_getPluginPath" << endl;

    vector<string> path = PluginHostAdapter::getPluginPath();
    PyTypeConversions conv;
    return conv.PyValue_From_StringVector(path);
}

static string toPluginKey(PyObject *pyPluginKey)
{
    cerr << "toPluginKey" << endl;

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
    cerr << "vampyhost_getLibraryFor" << endl;

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
    cerr << "vampyhost_getPluginCategory" << endl;

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
    cerr << "vampyhost_getOutputList" << endl;

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
        
        PyPluginObject *pd = getPluginObject(keyOrHandle);
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
    cerr << "vampyhost_loadPlugin" << endl;

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

    PyPluginObject *pd = PyPluginObject::create_internal();
    pd->key = new string(pluginKey);
    pd->plugin = plugin;
    pd->inputSampleRate = inputSampleRate;
    pd->isInitialised = false;
    pd->channels = 0;
    pd->blockSize = 0;
    pd->stepSize = 0;
    return (PyObject *)pd;
}

static PyObject *
vampyhost_initialise(PyObject *self, PyObject *args)
{
    cerr << "vampyhost_initialise" << endl;
    
    size_t channels, blockSize, stepSize;

    if (!PyArg_ParseTuple (args, "nnn",
			   (size_t) &channels,
			   (size_t) &stepSize,
			   (size_t) &blockSize)) {
	PyErr_SetString(PyExc_TypeError,
			"initialise() takes channel count, step size, and block size arguments");
	return 0;
    }

    PyPluginObject *pd = getPluginObject(self);
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
vampyhost_reset(PyObject *self, PyObject *)
{
    cerr << "vampyhost_reset" << endl;
    
    PyPluginObject *pd = getPluginObject(self);
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
    cerr << "vampyhost_getParameter" << endl;

    PyObject *pyParam;

    if (!PyArg_ParseTuple(args, "S", &pyParam)) {
	PyErr_SetString(PyExc_TypeError,
			"getParameter() takes parameter id (string) argument");
	return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    float value = pd->plugin->getParameter(PyString_AS_STRING(pyParam));
    return PyFloat_FromDouble(double(value));
}

static PyObject *
vampyhost_setParameter(PyObject *self, PyObject *args)
{
    cerr << "vampyhost_setParameter" << endl;

    PyObject *pyParam;
    float value;

    if (!PyArg_ParseTuple(args, "Sf", &pyParam, &value)) {
	PyErr_SetString(PyExc_TypeError,
			"setParameter() takes parameter id (string), and value (float) arguments");
	return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    pd->plugin->setParameter(PyString_AS_STRING(pyParam), value);
    return Py_True;
}

static PyObject *
vampyhost_process(PyObject *self, PyObject *args)
{
    cerr << "vampyhost_process" << endl;

    PyObject *pyBuffer;
    PyObject *pyRealTime;

    if (!PyArg_ParseTuple(args, "OO",
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

    PyPluginObject *pd = getPluginObject(self);
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

static PyObject *
vampyhost_unload(PyObject *self, PyObject *)
{
    cerr << "vampyhost_unloadPlugin" << endl;
    
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    delete pd->plugin;
    pd->plugin = 0; // This is checked by getPluginObject, so we 
                    // attempt to avoid repeated calls from blowing up

    return Py_True;
}

static PyMethodDef PyPluginObject_methods[] =
{
    {"getParameter",	vampyhost_getParameter, METH_VARARGS,
     xx_foo_doc}, //!!! fix all these!

    {"setParameter",	vampyhost_setParameter, METH_VARARGS,
     xx_foo_doc},

    {"initialise",	vampyhost_initialise, METH_VARARGS,
     xx_foo_doc},

    {"reset",	vampyhost_reset, METH_NOARGS,
     xx_foo_doc},

    {"process",	vampyhost_process, METH_VARARGS,
     xx_foo_doc},

    {"unload", vampyhost_unload, METH_NOARGS,
     xx_foo_doc},
    
    {0, 0}
};

static int
PyPluginObject_setattr(PyPluginObject *self, char *name, PyObject *value)
{
    return -1;
}

static PyObject *
PyPluginObject_getattr(PyPluginObject *self, char *name)
{
    return Py_FindMethod(PyPluginObject_methods, (PyObject *)self, name);
}

/* Doc:: 10.3 Type Objects */ /* static */ 
PyTypeObject Plugin_Type = 
{
    PyObject_HEAD_INIT(NULL)
    0,						/*ob_size*/
    "vampyhost.Plugin",				/*tp_name*/
    sizeof(PyPluginObject),	/*tp_basicsize*/
    0,		/*tp_itemsize*/
    (destructor)PyPluginObject_dealloc, /*tp_dealloc*/
    0,						/*tp_print*/
    (getattrfunc)PyPluginObject_getattr, /*tp_getattr*/
    (setattrfunc)PyPluginObject_setattr, /*tp_setattr*/
    0,						/*tp_compare*/
    0,			/*tp_repr*/
    0,	/*tp_as_number*/
    0,						/*tp_as_sequence*/
    0,						/*tp_as_mapping*/
    0,						/*tp_hash*/
    0,                      /*tp_call*/
    0,                      /*tp_str*/
    0,                      /*tp_getattro*/
    0,                      /*tp_setattro*/
    0,                      /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,     /*tp_flags*/
    "Plugin Object",      /*tp_doc*/
    0,                      /*tp_traverse*/
    0,                      /*tp_clear*/
    0,                      /*tp_richcompare*/
    0,                      /*tp_weaklistoffset*/
    0,                      /*tp_iter*/
    0,                      /*tp_iternext*/
    PyPluginObject_methods,       /*tp_methods*/ 
    0,                      /*tp_members*/
    0,                      /*tp_getset*/
    0,                      /*tp_base*/
    0,                      /*tp_dict*/
    0,                      /*tp_descr_get*/
    0,                      /*tp_descr_set*/
    0,                      /*tp_dictoffset*/
    0,                      /*tp_init*/
    PyType_GenericAlloc,         /*tp_alloc*/
    0,           /*tp_new*/
    PyObject_Del,			/*tp_free*/
    0,                      /*tp_is_gc*/
};

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

    {0,		0}		/* sentinel */
};

PyPluginObject *
PyPluginObject::create_internal()
{
    return (PyPluginObject *)PyType_GenericAlloc(&Plugin_Type, 0);
}

//Documentation for our new module
PyDoc_STRVAR(module_doc, "This is a template module just for instruction.");

static int
setint(PyObject *d, const char *name, int value)
{
    PyObject *v;
    int err;
    v = PyInt_FromLong((long)value);
    err = PyDict_SetItemString(d, name, v);
    Py_XDECREF(v);
    return err;
}

/* Initialization function for the module (*must* be called initxx) */

// module initialization (includes extern C {...} as necessary)
PyMODINIT_FUNC
initvampyhost(void)
{
    PyObject *m;

    if (PyType_Ready(&RealTime_Type) < 0) return;
    if (PyType_Ready(&Plugin_Type) < 0) return;

    m = Py_InitModule3("vampyhost", vampyhost_methods, module_doc);
    if (!m) {
        cerr << "ERROR: initvampyhost: Failed to initialise module" << endl;
        return;
    }

    import_array();

    PyModule_AddObject(m, "RealTime", (PyObject *)&RealTime_Type);
    PyModule_AddObject(m, "Plugin", (PyObject *)&Plugin_Type);

    // Some enum types
    PyObject *dict = PyModule_GetDict(m);
    if (!dict) {
        cerr << "ERROR: initvampyhost: Failed to obtain module dictionary" << endl;
        return;
    }

    if (setint(dict, "OneSamplePerStep",
               Plugin::OutputDescriptor::OneSamplePerStep) < 0 ||
        setint(dict, "FixedSampleRate",
               Plugin::OutputDescriptor::FixedSampleRate) < 0 ||
        setint(dict, "VariableSampleRate",
               Plugin::OutputDescriptor::VariableSampleRate) < 0 ||
        setint(dict, "TimeDomain",
               Plugin::TimeDomain) < 0 ||
        setint(dict, "FrequencyDomain",
               Plugin::FrequencyDomain) < 0) {
        cerr << "ERROR: initvampyhost: Failed to add enums to module dictionary" << endl;
        return;
    }
}

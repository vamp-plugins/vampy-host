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

//include for python extension module: must be first
#include <Python.h>

// define a unique API pointer 
#define PY_ARRAY_UNIQUE_SYMBOL VAMPYHOST_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "PyRealTime.h"
#include "PyPluginObject.h"

#include "vamp-hostsdk/PluginHostAdapter.h"
#include "vamp-hostsdk/PluginChannelAdapter.h"
#include "vamp-hostsdk/PluginInputDomainAdapter.h"
#include "vamp-hostsdk/PluginLoader.h"

#include "VectorConversion.h"
#include "PyRealTime.h"

#include <iostream>
#include <string>

#include <cmath>

using namespace std;
using namespace Vamp;
using namespace Vamp::HostExt;

PyDoc_STRVAR(xx_foo_doc, "Some description"); //!!!

//!!! todo: conv errors

static PyObject *
vampyhost_enumeratePlugins(PyObject *self, PyObject *)
{
    PluginLoader *loader = PluginLoader::getInstance();
    vector<PluginLoader::PluginKey> plugins = loader->listPlugins();
    VectorConversion conv;
    return conv.PyValue_From_StringVector(plugins);
}

static PyObject *
vampyhost_getPluginPath(PyObject *self, PyObject *)
{
    vector<string> path = PluginHostAdapter::getPluginPath();
    VectorConversion conv;
    return conv.PyValue_From_StringVector(path);
}

static string toPluginKey(PyObject *pyPluginKey)
{
    // convert to stl string
    string pluginKey(PyString_AS_STRING(pyPluginKey));

    // check pluginKey validity
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

    VectorConversion conv;
    return conv.PyValue_From_StringVector(category);
}

static PyObject *
vampyhost_getOutputList(PyObject *self, PyObject *args)
{
    PyObject *pyPluginKey;

    if (!PyArg_ParseTuple(args, "S", &pyPluginKey)) {
	PyErr_SetString(PyExc_TypeError,
			"getOutputList() takes plugin key (string) argument");
	return 0; }

    Plugin::OutputList outputs;

    string pluginKey = toPluginKey(pyPluginKey);
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

    return PyPluginObject_From_Plugin(plugin);
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

    {0,		0}		/* sentinel */
};

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

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

#include "PyPluginObject.h"

// define a unique API pointer 
#define PY_ARRAY_UNIQUE_SYMBOL VAMPYHOST_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "structmember.h"

#include "VectorConversion.h"
#include "PyRealTime.h"

#include <string>
#include <vector>
#include <cstddef>

using namespace std;
using namespace Vamp;

//!!! todo: conv errors

static
PyPluginObject *
getPluginObject(PyObject *pyPluginHandle)
{
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

static
PyObject *
pystr(const string &s)
{
    return PyString_FromString(s.c_str());
}

PyObject *
PyPluginObject_From_Plugin(Plugin *plugin)
{
    PyPluginObject *pd = 
        (PyPluginObject *)PyType_GenericAlloc(&Plugin_Type, 0);
    pd->plugin = plugin;
    pd->isInitialised = false;
    pd->channels = 0;
    pd->blockSize = 0;
    pd->stepSize = 0;

    PyObject *infodict = PyDict_New();
    PyDict_SetItemString
        (infodict, "apiVersion", PyInt_FromLong(plugin->getVampApiVersion()));
    PyDict_SetItemString
        (infodict, "pluginVersion", PyInt_FromLong(plugin->getPluginVersion()));
    PyDict_SetItemString
        (infodict, "identifier", pystr(plugin->getIdentifier()));
    PyDict_SetItemString
        (infodict, "name", pystr(plugin->getName()));
    PyDict_SetItemString
        (infodict, "description", pystr(plugin->getDescription()));
    PyDict_SetItemString
        (infodict, "maker", pystr(plugin->getMaker()));
    PyDict_SetItemString
        (infodict, "copyright", pystr(plugin->getCopyright()));
    pd->info = infodict;

    pd->inputDomain = plugin->getInputDomain();

    VectorConversion conv;

    Plugin::ParameterList pl = plugin->getParameterDescriptors();
    PyObject *params = PyList_New(pl.size());
    
    for (int i = 0; i < (int)pl.size(); ++i) {
        PyObject *paramdict = PyDict_New();
        PyDict_SetItemString
            (paramdict, "identifier", pystr(pl[i].identifier));
        PyDict_SetItemString
            (paramdict, "name", pystr(pl[i].name));
        PyDict_SetItemString
            (paramdict, "description", pystr(pl[i].description));
        PyDict_SetItemString
            (paramdict, "unit", pystr(pl[i].unit));
        PyDict_SetItemString
            (paramdict, "minValue", PyFloat_FromDouble(pl[i].minValue));
        PyDict_SetItemString
            (paramdict, "maxValue", PyFloat_FromDouble(pl[i].maxValue));
        PyDict_SetItemString
            (paramdict, "defaultValue", PyFloat_FromDouble(pl[i].defaultValue));
        if (pl[i].isQuantized) {
            PyDict_SetItemString
                (paramdict, "isQuantized", Py_True);
            PyDict_SetItemString
                (paramdict, "quantizeStep", PyFloat_FromDouble(pl[i].quantizeStep));
            if (!pl[i].valueNames.empty()) {
                PyDict_SetItemString
                    (paramdict, "valueNames", conv.PyValue_From_StringVector(pl[i].valueNames));
            }
        } else {
            PyDict_SetItemString
                (paramdict, "isQuantized", Py_False);
        }
        
        PyList_SET_ITEM(params, i, paramdict);
    }

    pd->parameters = params;

    Plugin::ProgramList prl = plugin->getPrograms();
    PyObject *progs = PyList_New(prl.size());

    for (int i = 0; i < (int)prl.size(); ++i) {
        PyList_SET_ITEM(progs, i, pystr(prl[i]));
    }

    pd->programs = progs;
    
    return (PyObject *)pd;
}

static void
PyPluginObject_dealloc(PyPluginObject *self)
{
    delete self->plugin;
    PyObject_Del(self);
}

static PyObject *
getOutputs(PyObject *self, PyObject *args)
{ 
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    Plugin::OutputList ol = pd->plugin->getOutputDescriptors();
    PyObject *outputs = PyList_New(ol.size());
    
    for (int i = 0; i < (int)ol.size(); ++i) {
        PyObject *outdict = PyDict_New();
        PyDict_SetItemString
            (outdict, "identifier", pystr(ol[i].identifier));
        PyDict_SetItemString
            (outdict, "name", pystr(ol[i].name));
        PyDict_SetItemString
            (outdict, "description", pystr(ol[i].description));
        PyDict_SetItemString
            (outdict, "binCount", PyInt_FromLong(ol[i].binCount));
        if (ol[i].binCount > 0) {
            if (ol[i].hasKnownExtents) {
                PyDict_SetItemString
                    (outdict, "hasKnownExtents", Py_True);
                PyDict_SetItemString
                    (outdict, "minValue", PyFloat_FromDouble(ol[i].minValue));
                PyDict_SetItemString
                    (outdict, "maxValue", PyFloat_FromDouble(ol[i].maxValue));
            } else {
                PyDict_SetItemString
                    (outdict, "hasKnownExtents", Py_False);
            }
            if (ol[i].isQuantized) {
                PyDict_SetItemString
                    (outdict, "isQuantized", Py_True);
                PyDict_SetItemString
                    (outdict, "quantizeStep", PyFloat_FromDouble(ol[i].quantizeStep));
            } else {
                PyDict_SetItemString
                    (outdict, "isQuantized", Py_False);
            }
        }
        PyDict_SetItemString
            (outdict, "sampleType", PyInt_FromLong((int)ol[i].sampleType));
        PyDict_SetItemString
            (outdict, "sampleRate", PyFloat_FromDouble(ol[i].sampleRate));
        PyDict_SetItemString
            (outdict, "hasDuration", ol[i].hasDuration ? Py_True : Py_False);
        
        PyList_SET_ITEM(outputs, i, outdict);
    }

    return outputs;
}

static PyObject *
initialise(PyObject *self, PyObject *args)
{
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
        cerr << "Failed to initialise native plugin adapter with channels = " << channels << ", stepSize = " << stepSize << ", blockSize = " << blockSize << endl;
        PyErr_SetString(PyExc_TypeError,
                        "Plugin initialization failed");
        return 0;
    }

    pd->isInitialised = true;

    return Py_True;
}

static PyObject *
reset(PyObject *self, PyObject *)
{
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
getParameter(PyObject *self, PyObject *args)
{
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
setParameter(PyObject *self, PyObject *args)
{
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
selectProgram(PyObject *self, PyObject *args)
{
    PyObject *pyParam;

    if (!PyArg_ParseTuple(args, "S", &pyParam)) {
        PyErr_SetString(PyExc_TypeError,
                        "selectProgram() takes parameter id (string) argument");
        return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    pd->plugin->selectProgram(PyString_AS_STRING(pyParam));
    return Py_True;
}

static
PyObject *
convertFeatureSet(const Plugin::FeatureSet &fs)
{
    VectorConversion conv;
    
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
                    (pyF, "label", pystr(f.label));

                if (!f.values.empty()) {
                    PyDict_SetItemString
                        (pyF, "values", conv.PyArray_From_FloatVector(f.values));
                }

                PyList_SET_ITEM(pyFl, fli, pyF);
            }

            PyObject *pyN = PyInt_FromLong(fno);
            PyDict_SetItem(pyFs, pyN, pyFl);
        }
    }
    
    return pyFs;
}

static vector<vector<float> >
convertPluginInput(PyObject *pyBuffer, int channels, int blockSize)
{
    vector<vector<float> > data;

    VectorConversion conv;

    if (PyArray_CheckExact(pyBuffer)) {

        data = conv.Py2DArray_To_FloatVector(pyBuffer);

        if (conv.error) {
            PyErr_SetString(PyExc_TypeError, conv.getError().str().c_str());
            return data;
        }

    } else {
        
        if (!PyList_Check(pyBuffer)) {
            PyErr_SetString(PyExc_TypeError, "List of NumPy Array required for process input.");
            return data;
        }

        if (PyList_GET_SIZE(pyBuffer) != channels) {
            cerr << "Wrong number of channels: got " << PyList_GET_SIZE(pyBuffer) << ", expected " << channels << endl;
            PyErr_SetString(PyExc_TypeError, "Wrong number of channels");
            return data;
        }

        for (int c = 0; c < channels; ++c) {
            PyObject *cbuf = PyList_GET_ITEM(pyBuffer, c);
            data.push_back(conv.PyValue_To_FloatVector(cbuf));
        }
    
        for (int c = 0; c < channels; ++c) {
            if ((int)data[c].size() != blockSize) {
                cerr << "Wrong number of samples on channel " << c << ": expected " << blockSize << " (plugin's block size), got " << data[c].size() << endl;
                PyErr_SetString(PyExc_TypeError, "Wrong number of samples");
                return vector<vector<float> >();
            }
        }
    }
    
    return data;
}

static PyObject *
process(PyObject *self, PyObject *args)
{
    PyObject *pyBuffer;
    PyObject *pyRealTime;

    if (!PyArg_ParseTuple(args, "OO",
                          &pyBuffer,                    // Audio data
                          &pyRealTime)) {               // TimeStamp
        PyErr_SetString(PyExc_TypeError,
                        "process() takes plugin handle (object), buffer (list of arrays of floats, one array per channel) and timestamp (RealTime) arguments");
        return 0; }

    if (!PyRealTime_Check(pyRealTime)) {
        PyErr_SetString(PyExc_TypeError, "Valid timestamp required.");
        return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    if (!pd->isInitialised) {
        PyErr_SetString(PyExc_StandardError,
                        "Plugin has not been initialised.");
        return 0;
    }

    int channels = pd->channels;
    vector<vector<float> > data =
        convertPluginInput(pyBuffer, channels, pd->blockSize);
    if (data.empty()) return 0;

    float **inbuf = new float *[channels];
    for (int c = 0; c < channels; ++c) {
        inbuf[c] = &data[c][0];
    }
    RealTime timeStamp = *PyRealTime_AsRealTime(pyRealTime);
    Plugin::FeatureSet fs = pd->plugin->process(inbuf, timeStamp);
    delete[] inbuf;

    return convertFeatureSet(fs);
}

static PyObject *
getRemainingFeatures(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    if (!pd->isInitialised) {
        PyErr_SetString(PyExc_StandardError,
                        "Plugin has not been initialised.");
        return 0;
    }

    Plugin::FeatureSet fs = pd->plugin->getRemainingFeatures();

    return convertFeatureSet(fs);
}

static PyObject *
getPreferredBlockSize(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyInt_FromLong(pd->plugin->getPreferredBlockSize());
}

static PyObject *
getPreferredStepSize(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyInt_FromLong(pd->plugin->getPreferredStepSize());
}

static PyObject *
getMinChannelCount(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyInt_FromLong(pd->plugin->getMinChannelCount());
}

static PyObject *
getMaxChannelCount(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyInt_FromLong(pd->plugin->getMaxChannelCount());
}
    
static PyObject *
unload(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    delete pd->plugin;
    pd->plugin = 0; // This is checked by getPluginObject, so we avoid
                    // blowing up if called repeatedly

    return Py_True;
}

static PyMemberDef PyPluginObject_members[] =
{
    {(char *)"info", T_OBJECT, offsetof(PyPluginObject, info), READONLY,
     (char *)"info -> A read-only dictionary of plugin metadata."},

    {(char *)"inputDomain", T_INT, offsetof(PyPluginObject, inputDomain), READONLY,
     (char *)"inputDomain -> The format of input audio required by the plugin, either vampyhost.TimeDomain or vampyhost.FrequencyDomain."},

    {(char *)"parameters", T_OBJECT, offsetof(PyPluginObject, parameters), READONLY,
     (char *)"parameters -> A list of metadata dictionaries describing the plugin's configurable parameters."},

    {(char *)"programs", T_OBJECT, offsetof(PyPluginObject, programs), READONLY,
     (char *)"programs -> A list of the programs available for this plugin, if any."},
    
    {0, 0}
};

static PyMethodDef PyPluginObject_methods[] =
{
    {"getOutputs", getOutputs, METH_NOARGS,
     "getOutputs() -> Obtain the output descriptors for all of the plugin's outputs."},

    {"getParameterValue", getParameter, METH_VARARGS,
     "getParameterValue(identifier) -> Return the value of the parameter with the given identifier."},

    {"setParameterValue", setParameter, METH_VARARGS,
     "setParameterValue(identifier, value) -> Set the parameter with the given identifier to the given value."},

    {"selectProgram", selectProgram, METH_VARARGS,
     "selectProgram(name) -> Select the processing program with the given name."},
    
    {"getPreferredBlockSize", getPreferredBlockSize, METH_VARARGS,
     "getPreferredBlockSize() -> Return the plugin's preferred processing block size, or 0 if the plugin accepts any block size."},

    {"getPreferredStepSize", getPreferredStepSize, METH_VARARGS,
     "getPreferredStepSize() -> Return the plugin's preferred processing step size, or 0 if the plugin allows the host to select. If this is 0, the host should normally choose the same step as block size for time-domain plugins, or half the block size for frequency-domain plugins."},

    {"getMinChannelCount", getMinChannelCount, METH_VARARGS,
     "getMinChannelCount() -> Return the minimum number of channels of audio data the plugin accepts as input."},

    {"getMaxChannelCount", getMaxChannelCount, METH_VARARGS,
     "getMaxChannelCount() -> Return the maximum number of channels of audio data the plugin accepts as input."},
    
    {"initialise", initialise, METH_VARARGS,
     "initialise(channels, stepSize, blockSize) -> Initialise the plugin for the given number of channels and processing frame sizes. This must be called before process() can be used."},

    {"reset", reset, METH_NOARGS,
     "reset() -> Reset the plugin after processing, to prepare for another processing run with the same parameters."},

    {"process", process, METH_VARARGS,
     "process(block, timestamp) -> Provide one processing frame to the plugin, with its timestamp, and obtain any features that were extracted immediately from this frame."},

    {"getRemainingFeatures", getRemainingFeatures, METH_NOARGS,
     "getRemainingFeatures() -> Obtain any features extracted at the end of processing."},

    {"unload", unload, METH_NOARGS,
     "unload() -> Dispose of the plugin. You cannot use the plugin object again after calling this. Note that unloading also happens automatically when the plugin object's reference count reaches zero; this function is only necessary if you wish to ensure the native part of the plugin is disposed of before then."},
    
    {0, 0}
};

/* Doc:: 10.3 Type Objects */ /* static */ 
PyTypeObject Plugin_Type = 
{
    PyObject_HEAD_INIT(NULL)
    0,                                  /*ob_size*/
    "vampyhost.Plugin",                 /*tp_name*/
    sizeof(PyPluginObject),             /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)PyPluginObject_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*tp_compare*/
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    PyObject_GenericGetAttr,            /*tp_getattro*/
    PyObject_GenericSetAttr,            /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                 /*tp_flags*/
    "Plugin object, providing a low-level API for running a Vamp plugin.", /*tp_doc*/
    0,                                  /*tp_traverse*/
    0,                                  /*tp_clear*/
    0,                                  /*tp_richcompare*/
    0,                                  /*tp_weaklistoffset*/
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    PyPluginObject_methods,             /*tp_methods*/ 
    PyPluginObject_members,             /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    0,                                  /*tp_init*/
    PyType_GenericAlloc,                /*tp_alloc*/
    0,                                  /*tp_new*/
    PyObject_Del,                       /*tp_free*/
    0,                                  /*tp_is_gc*/
};


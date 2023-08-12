/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    VampyHost

    Use Vamp audio analysis plugins in Python

    Gyorgy Fazekas and Chris Cannam
    Centre for Digital Music, Queen Mary, University of London
    Copyright 2008-2015 Queen Mary, University of London
  
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

#if PY_MAJOR_VERSION < 3
#include "intobject.h"
#endif

#include "structmember.h"

#include "FloatConversion.h"
#include "VectorConversion.h"
#include "StringConversion.h"
#include "PyRealTime.h"

#include "vamp-hostsdk/PluginWrapper.h"
#include "vamp-hostsdk/PluginInputDomainAdapter.h"

#include <string>
#include <vector>
#include <cstddef>
#include <set>

using namespace std;
using namespace Vamp;
using namespace Vamp::HostExt;

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


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

static int
setint(PyObject *d, const char *name, int value)
{
    PyObject *v;
    int err;
#if (PY_MAJOR_VERSION >= 3)
    v = PyLong_FromLong((long)value);
#else
    v = PyInt_FromLong((long)value);
#endif
    err = PyDict_SetItemString(d, name, v);
    Py_XDECREF(v);
    return err;
}

static int
setfloat(PyObject *d, const char *name, double value)
{
    PyObject *v;
    int err;
    v = PyFloat_FromDouble(value);
    err = PyDict_SetItemString(d, name, v);
    Py_XDECREF(v);
    return err;
}

static int
setstring(PyObject *d, const char *name, string value)
{
    PyObject *v;
    int err;
    v = StringConversion().string2py(value);
    err = PyDict_SetItemString(d, name, v);
    Py_XDECREF(v);
    return err;
}

PyObject *
PyPluginObject_From_Plugin(Plugin *plugin)
{
    PyPluginObject *pd = PyObject_New(PyPluginObject, &Plugin_Type);
    if (!pd) return 0;
    
    pd->plugin = plugin;
    pd->isInitialised = false;
    pd->channels = 0;
    pd->blockSize = 0;
    pd->stepSize = 0;
    pd->info = 0;
    pd->parameters = 0;
    pd->programs = 0;

    StringConversion strconv;
    
    PyObject *infodict = PyDict_New();
    setint(infodict, "apiVersion", plugin->getVampApiVersion());
    setint(infodict, "pluginVersion", plugin->getPluginVersion());
    setstring(infodict, "identifier", plugin->getIdentifier());
    setstring(infodict, "name", plugin->getName());
    setstring(infodict, "description", plugin->getDescription());
    setstring(infodict, "maker", plugin->getMaker());
    setstring(infodict, "copyright", plugin->getCopyright());
    pd->info = infodict;

    pd->inputDomain = plugin->getInputDomain();

    VectorConversion conv;

    Plugin::ParameterList pl = plugin->getParameterDescriptors();
    PyObject *params = PyList_New(pl.size());
    
    for (int i = 0; i < (int)pl.size(); ++i) {
        PyObject *paramdict = PyDict_New();
        setstring(paramdict, "identifier", pl[i].identifier);
        setstring(paramdict, "name", pl[i].name);
        setstring(paramdict, "description", pl[i].description);
        setstring(paramdict, "unit", pl[i].unit);
        setfloat(paramdict, "minValue", pl[i].minValue);
        setfloat(paramdict, "maxValue", pl[i].maxValue);
        setfloat(paramdict, "defaultValue", pl[i].defaultValue);
        if (pl[i].isQuantized) {
            PyDict_SetItemString(paramdict, "isQuantized", Py_True);
            setfloat(paramdict, "quantizeStep", pl[i].quantizeStep);
            if (!pl[i].valueNames.empty()) {
                PyObject *vv = conv.PyValue_From_StringVector(pl[i].valueNames);
                PyDict_SetItemString(paramdict, "valueNames", vv);
                Py_DECREF(vv);
            }
        } else {
            PyDict_SetItemString(paramdict, "isQuantized", Py_False);
        }
        
        PyList_SET_ITEM(params, i, paramdict);
    }

    pd->parameters = params;

    Plugin::ProgramList prl = plugin->getPrograms();
    PyObject *progs = PyList_New(prl.size());

    for (int i = 0; i < (int)prl.size(); ++i) {
        PyList_SET_ITEM(progs, i, strconv.string2py(prl[i]));
    }

    pd->programs = progs;
    
    return (PyObject *)pd;
}

static void
PyPluginObject_dealloc(PyPluginObject *self)
{
//    cerr << "PyPluginObject_dealloc: plugin object " << self << ", plugin " << self->plugin << endl;

    delete self->plugin;
    Py_XDECREF(self->info);
    Py_XDECREF(self->parameters);
    Py_XDECREF(self->programs);
    PyObject_Del(self);
}

static PyObject *
convertOutput(const Plugin::OutputDescriptor &desc, int ix)
{
    VectorConversion conv;
    StringConversion strconv;
    
    PyObject *outdict = PyDict_New();
    setstring(outdict, "identifier", desc.identifier);
    setstring(outdict, "name", desc.name);
    setstring(outdict, "description", desc.description);
    setstring(outdict, "unit", desc.unit);
    if (desc.hasFixedBinCount) {
        PyDict_SetItemString(outdict, "hasFixedBinCount", Py_True);
        setint(outdict, "binCount", desc.binCount);
        if (!desc.binNames.empty()) {
            PyObject *vv = conv.PyValue_From_StringVector(desc.binNames);
            PyDict_SetItemString(outdict, "binNames", vv);
            Py_DECREF(vv);
        }
    } else {
        PyDict_SetItemString(outdict, "hasFixedBinCount", Py_False);
    }
    if (!desc.hasFixedBinCount ||
        (desc.hasFixedBinCount && (desc.binCount > 0))) {
        if (desc.hasKnownExtents) {
            PyDict_SetItemString(outdict, "hasKnownExtents", Py_True);
            setfloat(outdict, "minValue", desc.minValue);
            setfloat(outdict, "maxValue", desc.maxValue);
        } else {
            PyDict_SetItemString(outdict, "hasKnownExtents", Py_False);
        }
        if (desc.isQuantized) {
            PyDict_SetItemString(outdict, "isQuantized", Py_True);
            setfloat(outdict, "quantizeStep", desc.quantizeStep);
        } else {
            PyDict_SetItemString(outdict, "isQuantized", Py_False);
        }
    }
    setint(outdict, "sampleType", (int)desc.sampleType);
    setfloat(outdict, "sampleRate", desc.sampleRate);
    PyDict_SetItemString
        (outdict, "hasDuration", desc.hasDuration ? Py_True : Py_False);
    setint(outdict, "output_index", ix);
    return outdict;
}

static PyObject *
get_output(PyObject *self, PyObject *args)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    ssize_t n = -1;
    PyObject *pyId = 0;
    
    if (!PyArg_ParseTuple(args, "n", &n) &&
        !PyArg_ParseTuple(args,
#if (PY_MAJOR_VERSION >= 3)
                          "U",
#else
                          "S",
#endif
                          &pyId)) {
        PyErr_SetString(PyExc_TypeError,
                        "get_output takes either output id (string) or output index (int) argument");
        return 0;
    }

    PyErr_Clear();
    
    Plugin::OutputList ol = pd->plugin->getOutputDescriptors();

    StringConversion strconv;
    
    if (pyId) {
        string id = strconv.py2string(pyId);
        for (int i = 0; i < int(ol.size()); ++i) {
            if (ol[i].identifier == id) {
                return convertOutput(ol[i], i);
            }
        }
    } else {
        if (n >= 0 && n < int(ol.size())) {
            return convertOutput(ol[n], n);
        }
    }

    PyErr_SetString(PyExc_Exception,
                    "unknown output id or output index out of range");
    return 0;
}

static PyObject *
get_outputs(PyObject *self, PyObject *args)
{ 
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    Plugin::OutputList ol = pd->plugin->getOutputDescriptors();
    PyObject *outputs = PyList_New(ol.size());
    
    for (int i = 0; i < (int)ol.size(); ++i) {
        PyObject *outdict = convertOutput(ol[i], i);
        PyList_SET_ITEM(outputs, i, outdict);
    }

    return outputs;
}

static PyObject *
set_process_timestamp_method(PyObject *self, PyObject *args)
{
    ssize_t method;

    if (!PyArg_ParseTuple(args,
                          "n",
                          &method)) {
        PyErr_SetString(PyExc_TypeError,
                        "set_process_timestamp_method() takes method (int) argument");
        return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    PluginWrapper *wrapper = dynamic_cast<PluginWrapper *>(pd->plugin);
    if (!wrapper) {
        PyErr_SetString(PyExc_Exception,
                        "Plugin was not loaded with ADAPT_INPUT_DOMAIN flag (no wrapper present)");
        return 0;
    }

    PluginInputDomainAdapter *adapter = wrapper->getWrapper<PluginInputDomainAdapter>();
    if (!adapter) {
        Py_RETURN_FALSE;
    }

    adapter->setProcessTimestampMethod
        (PluginInputDomainAdapter::ProcessTimestampMethod(method));
    Py_RETURN_TRUE;
}

static PyObject *
initialise(PyObject *self, PyObject *args)
{
    ssize_t channels, blockSize, stepSize;

    if (!PyArg_ParseTuple (args, "nnn",
                           &channels,
                           &stepSize,
                           &blockSize)) {
        PyErr_SetString(PyExc_TypeError,
                        "initialise() takes channel count (int), step size (int), and block size (int) arguments");
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

    Py_RETURN_TRUE;
}

static PyObject *
reset(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    if (!pd->isInitialised || !pd->plugin) {
        PyErr_SetString(PyExc_Exception,
                        "Plugin has not been initialised");
        return 0;
    }
        
    pd->plugin->reset();
    Py_RETURN_TRUE;
}

static bool
hasParameter(PyPluginObject *pd, string id)
{
    PluginBase::ParameterList pl = pd->plugin->getParameterDescriptors();
    for (int i = 0; i < (int)pl.size(); ++i) {
        if (pl[i].identifier == id) {
            return true;
        }
    }
    return false;
}

static PyObject *
get_parameter_value(PyObject *self, PyObject *args)
{
    PyObject *pyParam;

    if (!PyArg_ParseTuple(args,
#if (PY_MAJOR_VERSION >= 3)
                          "U",
#else
                          "S",
#endif
                          &pyParam)) {
        PyErr_SetString(PyExc_TypeError,
                        "get_parameter_value() takes parameter id (string) argument");
        return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    StringConversion strconv;
    
    string param = strconv.py2string(pyParam);
    
    if (!hasParameter(pd, param)) {
        PyErr_SetString(PyExc_Exception,
                        (string("Unknown parameter id \"") + param + "\"").c_str());
        return 0;
    }
    
    float value = pd->plugin->getParameter(param);
    return PyFloat_FromDouble(double(value));
}

static PyObject *
set_parameter_value(PyObject *self, PyObject *args)
{
    PyObject *pyParam;
    float value;

    if (!PyArg_ParseTuple(args,
#if (PY_MAJOR_VERSION >= 3)
                          "Uf",
#else
                          "Sf",
#endif
                          &pyParam, &value)) {
        PyErr_SetString(PyExc_TypeError,
                        "set_parameter_value() takes parameter id (string), and value (float) arguments");
        return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    StringConversion strconv;
    
    string param = strconv.py2string(pyParam);
    
    if (!hasParameter(pd, param)) {
        PyErr_SetString(PyExc_Exception,
                        (string("Unknown parameter id \"") + param + "\"").c_str());
        return 0;
    }

    pd->plugin->setParameter(param, value);
    Py_RETURN_TRUE;
}

static PyObject *
set_parameter_values(PyObject *self, PyObject *args)
{
    PyObject *dict;

    if (!PyArg_ParseTuple(args, "O", &dict)) {
        PyErr_SetString(PyExc_TypeError,
                        "set_parameter_values() takes dict argument");
        return 0;
    }

    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError,
                        "set_parameter_values() takes dict argument");
        return 0;
    }
    
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    PluginBase::ParameterList pl = pd->plugin->getParameterDescriptors();
    set<string> paramIds;
    for (int i = 0; i < (int)pl.size(); ++i) {
        paramIds.insert(pl[i].identifier);
    }

    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(dict, &pos, &key, &value)) {
#if PY_MAJOR_VERSION >= 3
        if (!key || !PyUnicode_CheckExact(key)) {
#else
        if (!key || !PyString_CheckExact(key)) {
#endif
            PyErr_SetString(PyExc_TypeError,
                            "Parameter dict keys must all have string type");
            return 0;
        }
        if (!value || !FloatConversion::check(value)) {
            PyErr_SetString(PyExc_TypeError,
                            "Parameter dict values must be convertible to float");
            return 0;
        }
        StringConversion strconv;
        string param = strconv.py2string(key);
        if (paramIds.find(param) == paramIds.end()) {
            PyErr_SetString(PyExc_Exception,
                            (string("Unknown parameter id \"") + param + "\"").c_str());
            return 0;
        }
        pd->plugin->setParameter(param, FloatConversion::convert(value));
    }

    Py_RETURN_TRUE;
}

static PyObject *
select_program(PyObject *self, PyObject *args)
{
    PyObject *pyParam;

    if (!PyArg_ParseTuple(args,
#if (PY_MAJOR_VERSION >= 3)
                          "U",
#else
                          "S",
#endif
                          &pyParam)) {
        PyErr_SetString(PyExc_TypeError,
                        "select_program() takes parameter id (string) argument");
        return 0;
    }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    StringConversion strconv;
    
    pd->plugin->selectProgram(strconv.py2string(pyParam));
    Py_RETURN_TRUE;
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
                    PyObject *rt = PyRealTime_FromRealTime(f.timestamp);
                    PyDict_SetItemString(pyF, "timestamp", rt);
                    Py_DECREF(rt);
                }
                if (f.hasDuration) {
                    PyObject *rt = PyRealTime_FromRealTime(f.duration);
                    PyDict_SetItemString(pyF, "duration", rt);
                    Py_DECREF(rt);
                }

                StringConversion strconv;

                setstring(pyF, "label", f.label);

                if (!f.values.empty()) {
                    PyObject *vv = conv.PyArray_From_FloatVector(f.values);
                    PyDict_SetItemString(pyF, "values", vv);
                    Py_DECREF(vv);
                }

                PyList_SET_ITEM(pyFl, fli, pyF);
            }

            PyObject *pyN = PyLong_FromLong(fno);
            PyDict_SetItem(pyFs, pyN, pyFl);
            Py_DECREF(pyN);
            Py_DECREF(pyFl);
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

        if ((int)data.size() != channels) {
//            cerr << "Wrong number of channels: got " << data.size() << ", expected " << channels << endl;
            PyErr_SetString(PyExc_TypeError, "Wrong number of channels");
            return vector<vector<float> >();
        }
        
    } else {
        
        if (!PyList_Check(pyBuffer)) {
            PyErr_SetString(PyExc_TypeError, "List of NumPy arrays or lists of numbers required for process input");
            return data;
        }
        
        if (PyList_GET_SIZE(pyBuffer) != channels) {
//            cerr << "Wrong number of channels: got " << PyList_GET_SIZE(pyBuffer) << ", expected " << channels << endl;
            PyErr_SetString(PyExc_TypeError, "Wrong number of channels");
            return data;
        }

        for (int c = 0; c < channels; ++c) {
            PyObject *cbuf = PyList_GET_ITEM(pyBuffer, c);
            data.push_back(conv.PyValue_To_FloatVector(cbuf));
            if (conv.error) {
                PyErr_SetString(PyExc_TypeError, conv.getError().str().c_str());
                return vector<vector<float> >();
            }
        }
    }
    
    for (int c = 0; c < channels; ++c) {
        if ((int)data[c].size() != blockSize) {
//            cerr << "Wrong number of samples on channel " << c << ": expected " << blockSize << " (plugin's block size), got " << data[c].size() << endl;
            PyErr_SetString(PyExc_TypeError, "Wrong number of samples for process block");
            return vector<vector<float> >();
        }
    }
    
    return data;
}

static PyObject *
process_block(PyObject *self, PyObject *args)
{
    PyObject *pyBuffer;
    PyObject *pyRealTime;

    if (!PyArg_ParseTuple(args, "OO",
                          &pyBuffer,                    // Audio data
                          &pyRealTime)) {               // TimeStamp
        PyErr_SetString(PyExc_TypeError,
                        "process_block() takes buffer (2D array or list of arrays, one row per channel) and timestamp (RealTime) arguments");
        return 0; }

    if (!PyRealTime_Check(pyRealTime)) {
        PyErr_SetString(PyExc_TypeError, "Valid timestamp required.");
        return 0; }

    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    if (!pd->isInitialised) {
        PyErr_SetString(PyExc_Exception,
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
get_remaining_features(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

    if (!pd->isInitialised) {
        PyErr_SetString(PyExc_Exception,
                        "Plugin has not been initialised.");
        return 0;
    }

    Plugin::FeatureSet fs = pd->plugin->getRemainingFeatures();

    return convertFeatureSet(fs);
}

static PyObject *
get_preferred_block_size(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyLong_FromLong(pd->plugin->getPreferredBlockSize());
}

static PyObject *
get_preferred_step_size(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyLong_FromLong(pd->plugin->getPreferredStepSize());
}

static PyObject *
get_min_channel_count(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyLong_FromLong(pd->plugin->getMinChannelCount());
}

static PyObject *
get_max_channel_count(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;
    return PyLong_FromLong(pd->plugin->getMaxChannelCount());
}
    
static PyObject *
unload(PyObject *self, PyObject *)
{
    PyPluginObject *pd = getPluginObject(self);
    if (!pd) return 0;

//    cerr << "unload: unloading plugin object " << pd << ", plugin " << pd->plugin << endl;
    
    delete pd->plugin;
    pd->plugin = 0; // This is checked by getPluginObject, so we avoid
                    // blowing up if called repeatedly

    Py_RETURN_TRUE;
}

static PyMemberDef PyPluginObject_members[] =
{
    {(char *)"info", T_OBJECT, offsetof(PyPluginObject, info), READONLY,
     (char *)"info -> A read-only dictionary of plugin metadata."},

    {(char *)"inputDomain", T_INT, offsetof(PyPluginObject, inputDomain), READONLY,
     (char *)"inputDomain -> The format of input audio required by the plugin, either vampyhost.TIME_DOMAIN or vampyhost.FREQUENCY_DOMAIN."},

    {(char *)"parameters", T_OBJECT, offsetof(PyPluginObject, parameters), READONLY,
     (char *)"parameters -> A list of metadata dictionaries describing the plugin's configurable parameters."},

    {(char *)"programs", T_OBJECT, offsetof(PyPluginObject, programs), READONLY,
     (char *)"programs -> A list of the programs available for this plugin, if any."},
    
    {0, 0}
};

static PyMethodDef PyPluginObject_methods[] =
{
    {"get_outputs", get_outputs, METH_NOARGS,
     "get_outputs() -> Obtain the output descriptors for all of the plugin's outputs."},

    {"get_output", get_output, METH_VARARGS,
     "get_output(out) -> Obtain the output descriptor for a single output, by either id (string) or index (int)."},

    {"get_parameter_value", get_parameter_value, METH_VARARGS,
     "get_parameter_value(identifier) -> Return the value of the parameter with the given identifier."},

    {"set_parameter_value", set_parameter_value, METH_VARARGS,
     "set_parameter_value(identifier, value) -> Set the parameter with the given identifier to the given value."},

    {"set_parameter_values", set_parameter_values, METH_VARARGS,
     "set_parameter_values(dict) -> Set multiple parameters to values corresponding to the key/value pairs in the dict. Any parameters not mentioned in the dict are unchanged."},
    
    {"select_program", select_program, METH_VARARGS,
     "select_program(name) -> Select the processing program with the given name."},
    
    {"get_preferred_block_size", get_preferred_block_size, METH_VARARGS,
     "get_preferred_block_size() -> Return the plugin's preferred processing block size, or 0 if the plugin accepts any block size."},

    {"get_preferred_step_size", get_preferred_step_size, METH_VARARGS,
     "get_preferred_step_size() -> Return the plugin's preferred processing step size, or 0 if the plugin allows the host to select. If this is 0, the host should normally choose the same step as block size for time-domain plugins, or half the block size for frequency-domain plugins."},

    {"get_min_channel_count", get_min_channel_count, METH_VARARGS,
     "get_min_channel_count() -> Return the minimum number of channels of audio data the plugin accepts as input."},

    {"get_max_channel_count", get_max_channel_count, METH_VARARGS,
     "get_max_channel_count() -> Return the maximum number of channels of audio data the plugin accepts as input."},

    {"set_process_timestamp_method", set_process_timestamp_method, METH_VARARGS,
     "set_process_timestamp_method(method) -> Set the method used for timestamp adjustment in plugins using frequency-domain input, where that input is being automatically converted for a plugin loaded with the ADAPT_INPUT_DOMAIN flag set (or one of ADAPT_ALL_SAFE or ADAPT_ALL). The method must be one of SHIFT_TIMESTAMP, SHIFT_DATA, or NO_SHIFT. The default is SHIFT_TIMESTAMP."},
    
    {"initialise", initialise, METH_VARARGS,
     "initialise(channels, stepSize, blockSize) -> Initialise the plugin for the given number of channels and processing frame sizes. This must be called before process_block() can be used."},

    {"reset", reset, METH_NOARGS,
     "reset() -> Reset the plugin after processing, to prepare for another processing run with the same parameters."},

    {"process_block", process_block, METH_VARARGS,
     "process_block(block, timestamp) -> Provide one processing frame to the plugin, with its timestamp, and obtain any features that were extracted immediately from this frame."},

    {"get_remaining_features", get_remaining_features, METH_NOARGS,
     "get_remaining_features() -> Obtain any features extracted at the end of processing."},

    {"unload", unload, METH_NOARGS,
     "unload() -> Dispose of the plugin. You cannot use the plugin object again after calling this. Note that unloading also happens automatically when the plugin object's reference count reaches zero; this function is only necessary if you wish to ensure the native part of the plugin is disposed of before then."},
    
    {0, 0}
};

/* Doc:: 10.3 Type Objects */ /* static */ 
PyTypeObject Plugin_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
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
    0,                                  /*tp_alloc*/
    0,                                  /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
};


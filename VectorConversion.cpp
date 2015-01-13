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

#include <Python.h>

#include "FloatConversion.h"
#include "VectorConversion.h"

#include <math.h>
#include <float.h>

using namespace std;

/*  Note: NO FUNCTION IN THIS CLASS SHOULD ALTER REFERENCE COUNTS
    (EXCEPT FOR TEMPORARY PYTHON OBJECTS)! */

VectorConversion::VectorConversion() : 
    m_error(false),
    error(m_error) // const public reference for easy access
{
}

VectorConversion::~VectorConversion()
{
}

/// floating point numbers (TODO: check numpy.float128)
float 
VectorConversion::PyValue_To_Float(PyObject* pyValue) const
{
    if (FloatConversion::check(pyValue)) {
        return FloatConversion::convert(pyValue);
    }

    setValueError("Conversion error: object" + PyValue_Get_TypeName(pyValue) +" is not float, int, or long.");
    return 0.0;
}

vector<float> 
VectorConversion::PyValue_To_FloatVector (PyObject *pyValue) const 
{
    /// numpy array
    if (PyArray_CheckExact(pyValue)) 
        return PyArray_To_FloatVector(pyValue);

    /// python list of floats (backward compatible)
    if (PyList_Check(pyValue)) {
        return PyList_To_FloatVector(pyValue);
    }

    string msg = "Value is not list or array of floats";
    setValueError(msg);
#ifdef _DEBUG
    cerr << "VectorConversion::PyValue_To_FloatVector failed. " << msg << endl;
#endif
    return vector<float>();
}

vector<float> 
VectorConversion::PyList_To_FloatVector (PyObject *inputList) const 
{
    vector<float> v;
	
    if (!PyList_Check(inputList)) {
        setValueError("Value is not a list");
        return v;
    } 

    PyObject **pyObjectArray = PySequence_Fast_ITEMS(inputList);
    int n = PyList_GET_SIZE(inputList);

    for (int i = 0; i < n; ++i) {
        v.push_back(PyValue_To_Float(pyObjectArray[i]));
    }
    
    return v;
}

vector<float> 
VectorConversion::PyArray_To_FloatVector (PyObject *pyValue) const 
{
    vector<float> v;
	
    if (!PyArray_Check(pyValue)) {
        setValueError("Value is not an array");
        return v;
    } 

    PyArrayObject* pyArray = (PyArrayObject*) pyValue;
    PyArray_Descr* descr = PyArray_DESCR(pyArray);
	
    if (PyArray_DATA(pyArray) == 0 || descr == 0) {
        string msg = "NumPy array with NULL data or descriptor pointer encountered.";
        setValueError(msg);
        return v;
    }

    if (PyArray_NDIM(pyArray) != 1) {
        string msg = "NumPy array must be a one-dimensional vector.";
        setValueError(msg);
        return v;
    }

    /// check strides (useful if array is not continuous)
    size_t strides =  *((size_t*) PyArray_STRIDES(pyArray));
    
    /// convert the array
    switch (descr->type_num) {
        
    case NPY_FLOAT : // dtype='float32'
        return PyArray_Convert<float,float>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
    case NPY_DOUBLE : // dtype='float64'
        return PyArray_Convert<float,double>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
    case NPY_INT : // dtype='int'
        return PyArray_Convert<float,int>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
    case NPY_LONG : // dtype='long'
        return PyArray_Convert<float,long>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
    default :
        string msg = "Unsupported value type in NumPy array object.";
        setValueError(msg);
#ifdef _DEBUG
        cerr << "VectorConversion::PyArray_To_FloatVector failed. Error: " << msg << endl;
#endif			
        return v;
    }
}

vector<vector<float> >
VectorConversion::Py2DArray_To_FloatVector (PyObject *pyValue) const 
{
    vector<vector<float> > v;
	
    if (!PyArray_Check(pyValue)) {
        setValueError("Value is not an array");
        return v;
    } 

    PyArrayObject* pyArray = (PyArrayObject*) pyValue;
    PyArray_Descr* descr = PyArray_DESCR(pyArray);
	
    if (PyArray_DATA(pyArray) == 0 || descr == 0) {
        string msg = "NumPy array with NULL data or descriptor pointer encountered.";
        setValueError(msg);
        return v;
    }

    if (PyArray_NDIM(pyArray) != 2) {
        string msg = "NumPy array must be a two-dimensional matrix.";
        setValueError(msg);
        return v;
    }

    /// check strides (useful if array is not continuous)
    size_t *strideptr =  (size_t*) PyArray_STRIDES(pyArray);
    
    /// convert the array
    for (int i = 0; i < PyArray_DIMS(pyArray)[0]; ++i) {

        vector<float> vv;
        
        switch (descr->type_num) {
        
        case NPY_FLOAT : // dtype='float32'
            vv = PyArray_Convert<float,float>(PyArray_GETPTR2(pyArray, i, 0),PyArray_DIMS(pyArray)[1],strideptr[1]);
            break;
        case NPY_DOUBLE : // dtype='float64'
            vv = PyArray_Convert<float,double>(PyArray_GETPTR2(pyArray, i, 0),PyArray_DIMS(pyArray)[1],strideptr[1]);
            break;
        case NPY_INT : // dtype='int'
            vv = PyArray_Convert<float,int>(PyArray_GETPTR2(pyArray, i, 0),PyArray_DIMS(pyArray)[1],strideptr[1]);
            break;
        case NPY_LONG : // dtype='long'
            vv = PyArray_Convert<float,long>(PyArray_GETPTR2(pyArray, i, 0),PyArray_DIMS(pyArray)[1],strideptr[1]);
            break;
        default :
            string msg = "Unsupported value type in NumPy array object.";
            cerr << "VectorConversion::PyArray_To_FloatVector failed (value type = " << descr->type_num << "). Error: " << msg << endl;
            setValueError(msg);
            return v;
        }

        v.push_back(vv);
    }

    return v;
}

PyObject *
VectorConversion::PyArray_From_FloatVector(const vector<float> &v) const
{
    npy_intp ndims[1];
    ndims[0] = (int)v.size();
    PyObject *arr = PyArray_SimpleNew(1, ndims, NPY_FLOAT);
    float *data = (float *)PyArray_DATA((PyArrayObject *)arr);
    for (int i = 0; i < ndims[0]; ++i) {
        data[i] = v[i];
    }
    return arr;
}

PyObject *
VectorConversion::PyValue_From_StringVector(const vector<string> &v) const
{
    PyObject *pyList = PyList_New(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        PyObject *pyStr = PyString_FromString(v[i].c_str());
        PyList_SET_ITEM(pyList, i, pyStr);
    }
    return pyList;
}


/* Error handling */

void
VectorConversion::setValueError (string message) const
{
    m_error = true;
    m_errorQueue.push(ValueError(message));
}

/// return a reference to the last error or creates a new one.
ValueError&
VectorConversion::lastError() const 
{
    m_error = false;
    if (!m_errorQueue.empty()) return m_errorQueue.back();
    else {
        m_errorQueue.push(ValueError("Type conversion error."));
        return m_errorQueue.back();
    }
}

/// helper function to iterate over the error message queue:
/// pops the oldest item
ValueError 
VectorConversion::getError() const
{
    if (!m_errorQueue.empty()) {
        ValueError e = m_errorQueue.front();
        m_errorQueue.pop();
        if (m_errorQueue.empty()) m_error = false;
        return e;
    }
    else {
        m_error = false;
        return ValueError();
    }
}

/* Utilities */

/// get the type name of an object
string
VectorConversion::PyValue_Get_TypeName(PyObject* pyValue) const
{
    PyObject *pyType = PyObject_Type(pyValue);
    if (!pyType) 
    {
        cerr << "Warning: Object type name could not be found." << endl;
        if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
        return string ("< unknown type >");
    }
    PyObject *pyString = PyObject_Str(pyType);
    if (!pyString)
    {
        cerr << "Warning: Object type name could not be found." << endl;
        if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
        Py_CLEAR(pyType);
        return string ("< unknown type >");
    }
    char *cstr = PyString_AS_STRING(pyString);
    if (!cstr)
    {
        cerr << "Warning: Object type name could not be found." << endl;
        if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
        Py_DECREF(pyType);
        Py_CLEAR(pyString);
        return string("< unknown type >");
    }
    Py_DECREF(pyType);
    Py_DECREF(pyString);
    return string(cstr);
}

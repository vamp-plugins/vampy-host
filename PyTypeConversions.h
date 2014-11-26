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

/*
  PyTypeConversions: Type safe conversion utilities between Python
  types and basic C/C++ types.
*/

#ifndef PY_TYPE_CONVERSIONS_H
#define PY_TYPE_CONVERSIONS_H

#include <Python.h>

// NumPy is required here
#define PY_ARRAY_UNIQUE_SYMBOL VAMPYHOST_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
	
// Data
class ValueError
{
public:
    ValueError() {}
    ValueError(std::string m) : message(m) {}
    std::string location;
    std::string message;
    std::string str() const { 
        return (location.empty()) ? message : message + "\nLocation: " + location;}
    template<typename V> ValueError &operator<< (const V& v)
    {
        std::ostringstream ss;
        ss << v;
        location += ss.str();
        return *this;
    }
};

class PyTypeConversions
{
public:
    PyTypeConversions();
    ~PyTypeConversions();

    ValueError getError() const;

    std::vector<float> PyValue_To_FloatVector (PyObject*) const;
    std::vector<float> PyArray_To_FloatVector (PyObject *) const;
    std::vector<float> PyList_To_FloatVector (PyObject*) const;

    PyObject *PyValue_From_StringVector(const std::vector<std::string> &) const;
    PyObject *PyArray_From_FloatVector(const std::vector<float> &) const;

private:
    std::string PyValue_Get_TypeName(PyObject*) const;
    float PyValue_To_Float(PyObject*) const;

    /// Convert DTYPE type 1D NumpyArray to std::vector<RET>
    template<typename RET, typename DTYPE>
    std::vector<RET> PyArray_Convert(void* raw_data_ptr,
                                     int length,
                                     size_t strides) const {
        
        std::vector<RET> v(length);
		
        /// check if the array is continuous, if not use strides info
        if (sizeof(DTYPE) != strides) {
#ifdef _DEBUG_VALUES
            cerr << "Warning: discontinuous numpy array. Strides: " << strides << " bytes. sizeof(dtype): " << sizeof(DTYPE) << endl;
#endif
            char* data = (char*) raw_data_ptr;
            for (int i = 0; i < length; ++i){
                v[i] = (RET)(*((DTYPE*)data));
                data += strides;
            }
            return v;
        }

        DTYPE* data = (DTYPE*) raw_data_ptr;
        for (int i = 0; i < length; ++i){
            v[i] = (RET)data[i];
        }

        return v;
    }

private:
    mutable bool m_error;
    mutable std::queue<ValueError> m_errorQueue;
	
    void setValueError(std::string) const;
    ValueError& lastError() const;
    
public:
    const bool& error;
};

#endif

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

#ifdef HAVE_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#endif

#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>

using std::cerr;
using std::endl;

#ifdef HAVE_NUMPY
enum eArrayDataType {
	dtype_float32 = (int) NPY_FLOAT,
	dtype_complex64 = (int) NPY_CFLOAT 
	};
#endif 

/* C++ mapping of PyNone Type */
struct NoneType {};
	
// Data
class ValueError
{
public:
	ValueError() {}
	ValueError(std::string m, bool s) : message(m),strict(s) {}
	std::string location;
	std::string message;
	bool strict;
	std::string str() const { 
		return (location.empty()) ? message : message + "\nLocation: " + location;}
	void print() const { cerr << str() << endl; }
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
	
	// Utilities
	void setStrictTypingFlag(bool b) {m_strict = b;}
	void setNumpyInstalled(bool b) {m_numpyInstalled = b;}
	ValueError getError() const;
	std::string PyValue_Get_TypeName(PyObject*) const;

	// Basic type conversion: Python to C++ 
	float 	PyValue_To_Float(PyObject*) const;
	size_t 	PyValue_To_Size_t(PyObject*) const;
	bool 	PyValue_To_Bool(PyObject*) const;
	std::string PyValue_To_String(PyObject*) const;
	long 	PyValue_To_Long(PyObject*) const;
	// int 	PyValue_To_Int(PyObject* pyValue) const;
	
	// C++ to Python
	PyObject *PyValue_From_CValue(const char*) const;
	PyObject *PyValue_From_CValue(const std::string& x) const { return PyValue_From_CValue(x.c_str()); }
	PyObject *PyValue_From_CValue(size_t) const;
	PyObject *PyValue_From_CValue(double) const;
	PyObject *PyValue_From_CValue(float x) const { return PyValue_From_CValue((double)x); }
	PyObject *PyValue_From_CValue(bool) const;
	
	// Sequence types
	std::vector<std::string> PyValue_To_StringVector (PyObject*) const;
	std::vector<float> PyValue_To_FloatVector (PyObject*) const;
	std::vector<float> PyList_To_FloatVector (PyObject*) const;

	PyObject *PyValue_From_StringVector(const std::vector<std::string> &) const;
	
	// Numpy types
#ifdef HAVE_NUMPY
	std::vector<float> PyArray_To_FloatVector (PyObject *pyValue) const;
	PyObject *FloatVector_To_PyArray(const std::vector<float> &) const; // Copying the data
#endif

	/// Convert DTYPE type 1D NumpyArray to std::vector<RET>
	template<typename RET, typename DTYPE>
	std::vector<RET> PyArray_Convert(void* raw_data_ptr, long length, size_t strides) const
	{
		std::vector<RET> rValue;
		
		/// check if the array is continuous, if not use strides info
		if (sizeof(DTYPE)!=strides) {
#ifdef _DEBUG_VALUES
			cerr << "Warning: discontinuous numpy array. Strides: " << strides << " bytes. sizeof(dtype): " << sizeof(DTYPE) << endl;
#endif
			char* data = (char*) raw_data_ptr;
			for (long i = 0; i<length; ++i){
				rValue.push_back((RET)(*((DTYPE*)data)));
#ifdef _DEBUG_VALUES
				cerr << "value: " << (RET)(*((DTYPE*)data)) << endl;
#endif				
				data+=strides;
			}
			return rValue;
		}

		DTYPE* data = (DTYPE*) raw_data_ptr;
		for (long i = 0; i<length; ++i){
#ifdef _DEBUG_VALUES
			cerr << "value: " << (RET)data[i] << endl;
#endif
			rValue.push_back((RET)data[i]);
		}
		return rValue;
	}

	/// this is a special case. numpy.float64 has an array conversions but no array descriptor
	std::vector<float> PyArray0D_Convert(PyArrayInterface *ai) const
	{
		std::vector<float> rValue;
		if ((ai->typekind) == *"f") 
			rValue.push_back((float)*(double*)(ai->data));
		else { 
			setValueError("Unsupported NumPy data type.",m_strict); 
			return rValue;
		}
#ifdef _DEBUG_VALUES
		cerr << "value: " << rValue[0] << endl;
#endif
		return rValue;
	}

private:
	bool m_strict;
	mutable bool m_error;
	mutable std::queue<ValueError> m_errorQueue;
	bool m_numpyInstalled;
	
	void setValueError(std::string,bool) const;
	ValueError& lastError() const;

public:
	const bool& error;

};

#endif

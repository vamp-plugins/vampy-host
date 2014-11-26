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

#include "PyTypeConversions.h"

#include <math.h>
#include <float.h>
#include <limits.h>
#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;

/*  Note: NO FUNCTION IN THIS CLASS SHOULD ALTER REFERENCE COUNTS
	(EXCEPT FOR TEMPORARY PYTHON OBJECTS)! */

PyTypeConversions::PyTypeConversions() : 
	m_strict(false),
	m_error(false),
	error(m_error) // const public reference for easy access
{
}

PyTypeConversions::~PyTypeConversions()
{
}


/// floating point numbers (TODO: check numpy.float128)
float 
PyTypeConversions::PyValue_To_Float(PyObject* pyValue) const
{
	// convert float
	if (pyValue && PyFloat_Check(pyValue)) 
		//TODO: check for limits here (same on most systems)
		return (float) PyFloat_AS_DOUBLE(pyValue);
	
	if (pyValue == NULL)
	{
		setValueError("Error while converting object " + PyValue_Get_TypeName(pyValue) + " to float. ",m_strict);
		return 0.0;		
	}
		
	// in strict mode we will not try harder
	if (m_strict) {
		setValueError("Strict conversion error: object" + PyValue_Get_TypeName(pyValue) +" is not float.",m_strict);
		return 0.0;
	}

	// convert other objects supporting the number protocol
	if (PyNumber_Check(pyValue))
	{
		PyObject* pyFloat = PyNumber_Float(pyValue); // new ref
		if (!pyFloat)
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			setValueError("Error while converting " + PyValue_Get_TypeName(pyValue) + " object to float.",m_strict);
			return 0.0;
		}
		float rValue = (float) PyFloat_AS_DOUBLE(pyFloat);
		Py_DECREF(pyFloat);
		return rValue;
	}
/*	
	// convert other objects supporting the number protocol
	if (PyNumber_Check(pyValue)) 
	{	
		// PEP353: Py_ssize_t is size_t but signed !
		// This will work up to numpy.float64
		Py_ssize_t rValue = PyNumber_AsSsize_t(pyValue,NULL);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting integer object.",m_strict);
			return 0.0;
		}
		if (rValue > (Py_ssize_t)FLT_MAX || rValue < (Py_ssize_t)FLT_MIN)
		{
			setValueError("Overflow error. Object can not be converted to float.",m_strict);
			return 0.0;
		}
		return (float) rValue;
	}
*/	
    // convert string
	if (PyString_Check(pyValue))
	{
		PyObject* pyFloat = PyFloat_FromString(pyValue,NULL);
		if (!pyFloat) 
		{
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			setValueError("String value can not be converted to float.",m_strict);
			return 0.0;
		}
		float rValue = (float) PyFloat_AS_DOUBLE(pyFloat);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear(); 
			Py_CLEAR(pyFloat);
			setValueError("Error while converting float object.",m_strict);
			return 0.0;
		}
		Py_DECREF(pyFloat);
		return rValue;
	}
	
	// convert the first element of any iterable sequence (for convenience and backwards compatibility)
	if (PySequence_Check(pyValue) && PySequence_Size(pyValue) > 0) 
	{
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			float rValue = this->PyValue_To_Float(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				std::string msg = "Could not convert sequence element to float. ";
				setValueError(msg,m_strict);
				return 0.0;
			}
		}
	}

    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + PyValue_Get_TypeName(pyValue) + " to float is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeConversions::PyValue_To_Float failed. " << msg << endl;
#endif	
	return 0.0;
}


/*			 			Sequence Types to C++ Types	    		  	*/

//convert PyFeature.value (typically a list or numpy array) to C++ vector of floats
std::vector<float> 
PyTypeConversions::PyValue_To_FloatVector (PyObject *pyValue) const 
{
	// there are four types of values we may receive from a numpy process:
	// * a python scalar, 
	// * an array scalar, (e.g. numpy.float32)
	// * an array with nd = 0  (0D array)
	// * an array with nd > 0

	/// check for scalars
	if (PyArray_CheckScalar(pyValue) || PyFloat_Check(pyValue)) {

		std::vector<float> Output;

		// we rely on the behaviour the scalars are either floats
		// or support the number protocol
		// TODO: a potential optimisation is to handle them directly
		Output.push_back(PyValue_To_Float(pyValue));
		return Output;
	}

	/// numpy array
	if (PyArray_CheckExact(pyValue)) 
		return PyArray_To_FloatVector(pyValue);

	/// python list of floats (backward compatible)
	if (PyList_Check(pyValue)) {
		return PyList_To_FloatVector(pyValue);
	}

	std::vector<float> Output;
	
	/// finally assume a single value supporting the number protocol 
	/// this allows to write e.g. Feature.values = 5 instead of [5.00]
	Output.push_back(PyValue_To_Float(pyValue));
	if (m_error) {
		std::string msg = "Value is not list or array of floats nor can be casted as float. ";
		setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeConversions::PyValue_To_FloatVector failed. " << msg << endl;
#endif
	}
	return Output;
}

//convert a list of python floats
std::vector<float> 
PyTypeConversions::PyList_To_FloatVector (PyObject *inputList) const 
{
	std::vector<float> Output;
	
#ifdef _DEBUG
	// This is a low level function normally called from 
	// PyValue_To_FloatVector(). Checking for list is not required.
	if (!PyList_Check(inputList)) {
		std::string msg = "Value is not list.";
		setValueError(msg,true);
		cerr << "PyTypeConversions::PyList_To_FloatVector failed. " << msg << endl;
		return Output; 
	} 
#endif

	float ListElement;
	PyObject *pyFloat = NULL;
	PyObject **pyObjectArray = PySequence_Fast_ITEMS(inputList);

	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(inputList); ++i) {

		// pyFloat = PyList_GET_ITEM(inputList,i);
		pyFloat = pyObjectArray[i];

#ifdef _DEBUG
		if (!pyFloat) {
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			cerr << "PyTypeConversions::PyList_To_FloatVector: Could not obtain list element: " 
			<< i << " PyList_GetItem returned NULL! Skipping value." << endl;
			continue;
		}
#endif		

		// ListElement = (float) PyFloat_AS_DOUBLE(pyFloat);
		ListElement = PyValue_To_Float(pyFloat);
		

#ifdef _DEBUG_VALUES
		cerr << "value: " << ListElement << endl;
#endif
		Output.push_back(ListElement);
	}
	return Output;
}

// if numpy is not installed this will not be called, 
// therefor we do not check again
std::vector<float> 
PyTypeConversions::PyArray_To_FloatVector (PyObject *pyValue) const 
{
	std::vector<float> Output;
	
#ifdef _DEBUG
	// This is a low level function, normally called from 
	// PyValue_To_FloatVector(). Checking the array here is not required.
	if (!PyArray_Check(pyValue)) {
		std::string msg = "Object has no array conversions.";
		setValueError(msg,true);
		cerr << "PyTypeConversions::PyArray_To_FloatVector failed. " << msg << endl;
		return Output; 
	} 
#endif

	PyArrayObject* pyArray = (PyArrayObject*) pyValue;
	PyArray_Descr* descr = PyArray_DESCR(pyArray);
	
	/// check raw data and descriptor pointers
	if (PyArray_DATA(pyArray) == 0 || descr == 0) {
		std::string msg = "NumPy array with NULL data or descriptor pointer encountered.";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyArray_To_FloatVector failed. Error: " << msg << endl;
#endif		
		return Output;
	}

	/// check dimensions
	if (PyArray_NDIM(pyArray) != 1) {
		std::string msg = "NumPy array must be a one dimensional vector.";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyArray_To_FloatVector failed. Error: " << msg << " Dims: " << (int) PyArray_NDIM(pyArray) << endl;
#endif	
		return Output;
	}

#ifdef _DEBUG_VALUES
	cerr << "PyTypeConversions::PyArray_To_FloatVector: Numpy array verified." << endl;
#endif
	
	/// check strides (useful if array is not continuous)
	size_t strides =  *((size_t*) PyArray_STRIDES(pyArray));
    
	/// convert the array
	switch (descr->type_num)
	{
		case NPY_FLOAT : // dtype='float32'
			return PyArray_Convert<float,float>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
		case NPY_DOUBLE : // dtype='float64'
			return PyArray_Convert<float,double>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
		case NPY_INT : // dtype='int'
			return PyArray_Convert<float,int>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
		case NPY_LONG : // dtype='long'
			return PyArray_Convert<float,long>(PyArray_DATA(pyArray),PyArray_DIMS(pyArray)[0],strides);
		default :
			std::string msg = "Unsupported value type in NumPy array object.";
			setValueError(msg,m_strict);
#ifdef _DEBUG
			cerr << "PyTypeConversions::PyArray_To_FloatVector failed. Error: " << msg << endl;
#endif			
			return Output;
	}
}

PyObject *
PyTypeConversions::FloatVector_To_PyArray(const vector<float> &v) const
{
	npy_intp ndims[1];
	ndims[0] = (int)v.size();
	PyObject *arr = PyArray_SimpleNew(1, ndims, dtype_float32);
	float *data = (float *)PyArray_DATA((PyArrayObject *)arr);
	for (int i = 0; i < ndims[0]; ++i) {
		data[i] = v[i];
	}
	return arr;
}

PyObject *
PyTypeConversions::PyValue_From_StringVector(const std::vector<std::string> &v) const
{
	PyObject *pyList = PyList_New(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		PyObject *pyStr = PyString_FromString(v[i].c_str());
		PyList_SET_ITEM(pyList, i, pyStr);
	}
	return pyList;
}


/*			   			  	Error handling		   			  		*/

void
PyTypeConversions::setValueError (std::string message, bool strict) const
{
	m_error = true;
	m_errorQueue.push(ValueError(message,strict));
}

/// return a reference to the last error or creates a new one.
ValueError&
PyTypeConversions::lastError() const 
{
	m_error = false;
	if (!m_errorQueue.empty()) return m_errorQueue.back();
	else {
		m_errorQueue.push(ValueError("Type conversion error.",m_strict));
		return m_errorQueue.back();
	}
}

/// helper function to iterate over the error message queue:
/// pops the oldest item
ValueError 
PyTypeConversions::getError() const
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

/*			   			  	Utilities						  		*/

/// get the type name of an object
std::string
PyTypeConversions::PyValue_Get_TypeName(PyObject* pyValue) const
{
	PyObject *pyType = PyObject_Type(pyValue);
	if (!pyType) 
	{
		cerr << "Warning: Object type name could not be found." << endl;
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		return std::string ("< unknown type >");
	}
	PyObject *pyString = PyObject_Str(pyType);
	if (!pyString)
	{
		cerr << "Warning: Object type name could not be found." << endl;
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		Py_CLEAR(pyType);
		return std::string ("< unknown type >");
	}
	char *cstr = PyString_AS_STRING(pyString);
	if (!cstr)
	{
		cerr << "Warning: Object type name could not be found." << endl;
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		Py_DECREF(pyType);
		Py_CLEAR(pyString);
		return std::string("< unknown type >");
	}
	Py_DECREF(pyType);
	Py_DECREF(pyString);
	return std::string(cstr);
	
}

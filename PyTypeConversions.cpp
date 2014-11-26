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
	m_numpyInstalled(false),
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

/// size_t (unsigned integer types)
size_t 
PyTypeConversions::PyValue_To_Size_t(PyObject* pyValue) const
{
	// convert objects supporting the number protocol 
	if (PyNumber_Check(pyValue)) 
	{	
		if (m_strict && !PyInt_Check(pyValue) && !PyLong_Check(pyValue)) 
			setValueError("Strict conversion error: object is not integer type.",m_strict);
		// Note: this function handles Bool,Int,Long,Float
		// speed is not critical in the use of this type by Vamp
		// PEP353: Py_ssize_t is size_t but signed ! 
		Py_ssize_t rValue = PyInt_AsSsize_t(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting integer object.",m_strict);
			return 0;
		}
		// this test is nonsense -- neither part can occur
		// owing to range of data types -- size_t is at least
		// as big as long, and unsigned is always non-negative
/*
		if ((unsigned long)rValue > SIZE_T_MAX || (unsigned long)rValue < 0)
		{
			setValueError("Overflow error. Object can not be converted to size_t.",m_strict);
			return 0;
		}
*/
		return (size_t) rValue;
	}
	
	// in strict mode we will not try harder and throw an exception
	// then the caller should decide what to do with it
	if (m_strict) {
		setValueError("Strict conversion error: object is not integer.",m_strict);
		return 0;
	}
	
	// convert string
	if (PyString_Check(pyValue))
	{
		PyObject* pyLong = PyNumber_Long(pyValue);
		if (!pyLong) 
		{
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			setValueError("String object can not be converted to size_t.",m_strict);
			return 0;
		}
		size_t rValue = this->PyValue_To_Size_t(pyLong);
		if (!m_error) {
			Py_DECREF(pyLong);
			return rValue;
		} else {
			Py_CLEAR(pyLong);
			setValueError ("Error converting string to size_t.",m_strict);
			return 0;
		}
	}
	
	// convert the first element of iterable sequences
	if (PySequence_Check(pyValue) && PySequence_Size(pyValue) > 0) 
	{
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			size_t rValue = this->PyValue_To_Size_t(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				setValueError("Could not convert sequence element to size_t. ",m_strict);
				return 0;
			}
		}
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to size_t is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeConversions::PyValue_To_Size_t failed. " << msg << endl;
#endif	
	return 0;
}

/// long and int
long 
PyTypeConversions::PyValue_To_Long(PyObject* pyValue) const
{
	// most common case: convert int (faster)
	if (pyValue && PyInt_Check(pyValue)) {
		// if the object is not NULL and verified, this macro just extracts the value.
		return PyInt_AS_LONG(pyValue);
	} 
	
	// long
	if (PyLong_Check(pyValue)) {
		long rValue = PyLong_AsLong(pyValue);
		if (PyErr_Occurred()) { 
			PyErr_Print(); PyErr_Clear(); 
			setValueError("Error while converting long object.",m_strict);
			return 0;
		}
		return rValue;
	}
	
	if (m_strict) {
		setValueError("Strict conversion error: object is not integer or long integer.",m_strict);
		return 0;
	}
	
	// convert all objects supporting the number protocol
	if (PyNumber_Check(pyValue)) 
	{	
		// Note: this function handles Bool,Int,Long,Float
		// PEP353: Py_ssize_t is size_t but signed ! 
		Py_ssize_t rValue = PyInt_AsSsize_t(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting integer object.",m_strict);
			return 0;
		}
		if (rValue > LONG_MAX || rValue < LONG_MIN)
		{
			setValueError("Overflow error. Object can not be converted to size_t.",m_strict);
			return 0;
		}
		return (long) rValue;
	}
	
	// convert string
	if (PyString_Check(pyValue))
	{
		PyObject* pyLong = PyNumber_Long(pyValue);
		if (!pyLong) 
		{
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			setValueError("String object can not be converted to long.",m_strict);
			return 0;
		}
		long rValue = this->PyValue_To_Long(pyLong);
		if (!m_error) {
			Py_DECREF(pyLong);
			return rValue;
		} else {
			Py_CLEAR(pyLong);
			setValueError ("Error converting string to long.",m_strict);
			return 0;
		}
	}
	
	// convert the first element of iterable sequences
	if (PySequence_Check(pyValue) && PySequence_Size(pyValue) > 0) 
	{
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			size_t rValue = this->PyValue_To_Long(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				setValueError("Could not convert sequence element to long. ",m_strict);
				return 0;
			}
		}
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to long is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeConversions::PyValue_To_Long failed. " << msg << endl;
#endif	
	return 0;
}


bool 
PyTypeConversions::PyValue_To_Bool(PyObject* pyValue) const
{
	// convert objects supporting the number protocol
	// Note: PyBool is a subclass of PyInt
	if (PyNumber_Check(pyValue)) 
	{	
		if (m_strict && !PyBool_Check(pyValue)) 
			setValueError
			("Strict conversion error: object is not boolean type.",m_strict);

		// Note: this function handles Bool,Int,Long,Float
		Py_ssize_t rValue = PyInt_AsSsize_t(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError ("Error while converting boolean object.",m_strict);
		}
		if (rValue != 1 && rValue != 0)
		{
			setValueError ("Overflow error. Object can not be converted to boolean.",m_strict);
		}
		return (bool) rValue;
	}
	
	if (m_strict) {
		setValueError ("Strict conversion error: object is not numerical type.",m_strict);
		return false;
	}
	
	// convert iterables: the rule is the same as in the interpreter:
	// empty sequence evaluates to False, anything else is True
	if (PySequence_Check(pyValue)) 
	{
		return PySequence_Size(pyValue)?true:false;
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to boolean is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeConversions::PyValue_To_Bool failed. " << msg << endl;
#endif	
	return false;
}

/// string and objects that support .__str__() 
/// TODO: check unicode objects
std::string 
PyTypeConversions::PyValue_To_String(PyObject* pyValue) const
{
	// convert string
	if (PyString_Check(pyValue)) 
	{	
		char *cstr = PyString_AS_STRING(pyValue);
		if (!cstr) 
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			setValueError("Error while converting string object.",m_strict);
			return std::string();
		}
		return std::string(cstr);
	}
	// TODO: deal with unicode here (argh!)
	
	// in strict mode we will not try harder
	if (m_strict) {
		setValueError("Strict conversion error: object is not string.",m_strict);
		return std::string();
	}
	
	// accept None as empty string
	if (pyValue == Py_None) return std::string();
			
	// convert list or tuple: empties are turned into empty strings conventionally
	if (PyList_Check(pyValue) || PyTuple_Check(pyValue)) 
	{
		if (!PySequence_Size(pyValue)) return std::string();
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			std::string rValue = this->PyValue_To_String(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				setValueError("Could not convert sequence element to string.",m_strict);
				return std::string();
			}
		}
	}

	// convert any other object that has .__str__() or .__repr__()
	PyObject* pyString = PyObject_Str(pyValue);
	if (pyString && !PyErr_Occurred())
	{
		std::string rValue = this->PyValue_To_String(pyString);
		if (!m_error) {
			Py_DECREF(pyString);
			return rValue;
		} else {
			Py_CLEAR(pyString);
			std::string msg = "Object " + this->PyValue_Get_TypeName(pyValue) +" can not be represented as string. ";
			setValueError (msg,m_strict);
			return std::string();
		}
	}

	// give up
	PyErr_Print(); PyErr_Clear();
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to string is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeConversions::PyValue_To_String failed. " << msg << endl;
#endif	
	return std::string();
}

/*			 			C Values to Py Values				  		*/


PyObject*
PyTypeConversions::PyValue_From_CValue(const char* cValue) const
{
	// returns new reference
#ifdef _DEBUG
	if (!cValue) {
		std::string msg = "PyTypeConversions::PyValue_From_CValue: Null pointer encountered while converting from const char* .";
		cerr << msg << endl;
		setValueError(msg,m_strict);
		return NULL;
	}
#endif
	PyObject *pyValue = PyString_FromString(cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from char* or string.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyValue_From_CValue: Interpreter failed to convert from const char*" << endl;
#endif
		return NULL;
	}
	return pyValue;
}

PyObject*
PyTypeConversions::PyValue_From_CValue(size_t cValue) const
{
	// returns new reference
	PyObject *pyValue = PyInt_FromSsize_t((Py_ssize_t)cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from size_t.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyValue_From_CValue: Interpreter failed to convert from size_t" << endl;
#endif
		return NULL;
	}
	return pyValue;
}

PyObject*
PyTypeConversions::PyValue_From_CValue(double cValue) const
{
	// returns new reference
	PyObject *pyValue = PyFloat_FromDouble(cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from float or double.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyValue_From_CValue: Interpreter failed to convert from float or double" << endl;
#endif
		return NULL;
	}
	return pyValue;
}

PyObject*
PyTypeConversions::PyValue_From_CValue(bool cValue) const
{
	// returns new reference
	PyObject *pyValue = PyBool_FromLong((long)cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from bool.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyValue_From_CValue: Interpreter failed to convert from bool" << endl;
#endif
		return NULL;
	}
	return pyValue;
}


/*			 			Sequence Types to C++ Types	    		  	*/

//convert Python list to C++ vector of strings
std::vector<std::string> 
PyTypeConversions::PyValue_To_StringVector (PyObject *pyList) const 
{
	
	std::vector<std::string> Output;
	std::string ListElement;
	PyObject *pyString = NULL;
	
	if (PyList_Check(pyList)) {

		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
			//Get next list item (Borrowed Reference)
			pyString = PyList_GET_ITEM(pyList,i);
			ListElement = (string) PyString_AsString(PyObject_Str(pyString));
			Output.push_back(ListElement);
		}
		return Output;
	}

// #ifdef _DEBUG
// 	cerr << "PyTypeConversions::PyValue_To_StringVector: Warning: Value is not list of strings." << endl;
// #endif

	/// Assume a single value that can be casted as string 
	/// this allows to write e.g. Feature.label = 5.2 instead of ['5.2']
	Output.push_back(PyValue_To_String(pyList));
	if (m_error) {
		std::string msg = "Value is not list of strings nor can be casted as string. ";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeConversions::PyValue_To_StringVector failed. " << msg << endl;
#endif
	}
	return Output;
}

//convert PyFeature.value (typically a list or numpy array) to C++ vector of floats
std::vector<float> 
PyTypeConversions::PyValue_To_FloatVector (PyObject *pyValue) const 
{

#ifdef HAVE_NUMPY
if (m_numpyInstalled) 
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
}
#endif

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
#ifdef HAVE_NUMPY 
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
#endif

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

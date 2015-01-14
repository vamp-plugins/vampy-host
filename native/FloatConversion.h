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

#ifndef VAMPYHOST_FLOAT_CONVERSION_H
#define VAMPYHOST_FLOAT_CONVERSION_H

class FloatConversion
{
public:
    static bool check(PyObject *pyValue) {
	if (pyValue && PyFloat_Check(pyValue)) {
	    return true;
	}
	if (pyValue && PyLong_Check(pyValue)) {
	    return true;
	}
	if (pyValue && PyInt_Check(pyValue)) {
	    return true;
	}
	return false;
    }
	
    static float convert(PyObject* pyValue) {
	
	if (pyValue && PyFloat_Check(pyValue)) {
	    return (float) PyFloat_AS_DOUBLE(pyValue);
	}

	if (pyValue && PyLong_Check(pyValue)) {
	    return (float) PyLong_AsDouble(pyValue);
	}

	if (pyValue && PyInt_Check(pyValue)) {
	    return (float) PyInt_AsLong(pyValue);
	}

	return 0.0;
    }
};

#endif


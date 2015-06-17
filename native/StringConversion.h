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

/*
  StringConversion: A couple of type safe conversion utilities between
  Python types and C++ strings.
*/

#ifndef VAMPYHOST_STRING_CONVERSION_H
#define VAMPYHOST_STRING_CONVERSION_H

#include <Python.h>
#include <string>

class StringConversion
{
public:
    StringConversion() {}
    ~StringConversion() {}

    PyObject *string2py(const std::string &s) {
#if PY_MAJOR_VERSION < 3
	return PyString_FromString(s.c_str());
#else
	return PyUnicode_FromString(s.c_str());
#endif
    }

    std::string py2string(PyObject *obj) {
#if PY_MAJOR_VERSION < 3
	char *cstr = PyString_AsString(obj);
	if (!cstr) return std::string();
	else return cstr;
#else
	PyObject *uobj = PyUnicode_AsUTF8String(obj);
	if (!uobj) return std::string();
	char *cstr = PyBytes_AsString(uobj);
	if (!cstr) return std::string();
	else return cstr;
#endif
    }
};

#endif



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

#include "PyRealTime.h"

#include <string>

using namespace std;
using namespace Vamp;

/* CONSTRUCTOR: New RealTime object from sec and nsec */
static PyObject*
RealTime_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    unsigned int sec = 0;
    unsigned int nsec = 0;
    double unary = 0;
    const char *fmt = NULL;

    if (
        /// new RealTime from ('format',float) e.g. ('seconds',2.34123)   
        !PyArg_ParseTuple(args, "|sd:RealTime.new ", 
                          (const char *) &fmt, 
                          (double *) &unary)    &&

        /// new RealTime from (sec{int},nsec{int}) e.g. (2,34)
        !PyArg_ParseTuple(args, "|II:RealTime.new ", 
                          (unsigned int*) &sec, 
                          (unsigned int*) &nsec) 
                
        ) { 
        PyErr_SetString(PyExc_TypeError, 
                        "RealTime initialised with wrong arguments.");
        return NULL; 
    }

    // PyErr_Clear();

    // RealTimeObject *self = PyObject_New(RealTimeObject, &RealTime_Type); 
    RealTimeObject *self = (RealTimeObject*)type->tp_alloc(type, 0);
        
    if (self == NULL) return NULL;

    self->rt = NULL;

    if (sec == 0 && nsec == 0 && fmt == 0) 
        self->rt = new RealTime();
    else if (fmt == 0)
        self->rt = new RealTime(sec,nsec);
    else { 
        /// new RealTime from seconds or milliseconds: i.e. >>>RealTime('seconds',12.3)
        if (!string(fmt).compare("float") ||
            !string(fmt).compare("seconds"))  
            self->rt = new RealTime( 
                RealTime::fromSeconds((double) unary)); 

        if (!string(fmt).compare("milliseconds")) {
            self->rt = new RealTime( 
                RealTime::fromSeconds((double) unary / 1000.0)); }
    }

    if (!self->rt) { 
        PyErr_SetString(PyExc_TypeError, 
                        "RealTime initialised with wrong arguments.");
        return NULL; 
    }

    return (PyObject *) self;
}

/* DESTRUCTOR: delete type object */
static void
RealTimeObject_dealloc(RealTimeObject *self)
{
    if (self->rt) delete self->rt;      //delete the C object
    PyObject_Del(self); //delete the Python object (original)
    /// this requires PyType_Ready() which fills ob_type
    // self->ob_type->tp_free((PyObject*)self); 
}

/* RealTime Object's Methods */ 
//these are internals not exposed by the module but the object

/* Returns a Tuple containing sec and nsec values */
static PyObject *
RealTime_values(RealTimeObject *self)
{
    return Py_BuildValue("(ii)",self->rt->sec,self->rt->nsec);
}

/* Returns a Text representation */
static PyObject *
RealTime_toString(RealTimeObject *self, PyObject *args)
{
    return Py_BuildValue("s",self->rt->toText().c_str());
}

/* Frame representation */
static PyObject *
RealTime_toFrame(PyObject *self, PyObject *args)
{
    unsigned int samplerate;
        
    if ( !PyArg_ParseTuple(args, "I:realtime.toFrame object ", 
                           (unsigned int *) &samplerate )) {
        PyErr_SetString(PyExc_ValueError,"Integer Sample Rate Required.");
        return NULL;
    }
        
    return Py_BuildValue("k", 
                         RealTime::realTime2Frame( 
                             *(const RealTime*) ((RealTimeObject*)self)->rt, 
                             (unsigned int) samplerate));
}

/* Conversion of realtime to a double precision floating point value */
/* ...in Python called by e.g. float(realtime) */
static PyObject *
RealTime_float(PyObject *s)
{
    double drt = ((double) ((RealTimeObject*)s)->rt->sec + 
                  (double)((double) ((RealTimeObject*)s)->rt->nsec)/1000000000);
    return PyFloat_FromDouble(drt);     
}


/* Type object's (RealTime) methods table */
static PyMethodDef RealTime_methods[] = 
{
    {"values",  (PyCFunction)RealTime_values,   METH_NOARGS,
     PyDoc_STR("values() -> Tuple of sec,nsec representation.")},

    {"toString",        (PyCFunction)RealTime_toString, METH_NOARGS,
     PyDoc_STR("toString() -> Return a user-readable string to the nearest millisecond in a form like HH:MM:SS.mmm")},

    {"toFrame", (PyCFunction)RealTime_toFrame,  METH_VARARGS,
     PyDoc_STR("toFrame(samplerate) -> Sample count for given sample rate.")},

    {"toFloat", (PyCFunction)RealTime_float,    METH_NOARGS,
     PyDoc_STR("toFloat() -> Floating point representation.")},
        
    {NULL,              NULL}           /* sentinel */
};


/* Methods implementing protocols */ 
// these functions are called by the interpreter 

/* Object Protocol */

static int
RealTime_setattr(RealTimeObject *self, char *name, PyObject *value)
{
    if ( !string(name).compare("sec")) { 
        self->rt->sec= (int) PyInt_AS_LONG(value);
        return 0;
    }

    if ( !string(name).compare("nsec")) { 
        self->rt->nsec= (int) PyInt_AS_LONG(value);
        return 0;
    }

    return -1;
}

static PyObject *
RealTime_getattr(RealTimeObject *self, char *name)
{
    if ( !string(name).compare("sec") ) { 
        return PyInt_FromSsize_t(
            (Py_ssize_t) self->rt->sec); 
    } 

    if ( !string(name).compare("nsec") ) { 
        return PyInt_FromSsize_t(
            (Py_ssize_t) self->rt->nsec); 
    } 

    return Py_FindMethod(RealTime_methods, 
                         (PyObject *)self, name);
}

static int
RealTime_compare(PyObject *self, PyObject *other)
{
    if (!PyRealTime_Check(self) || !PyRealTime_Check(other)) {
        PyErr_SetString(PyExc_TypeError, "RealTime Object Expected.");
        return -1;
    }

    RealTime *rt1 = PyRealTime_AS_REALTIME(self);
    RealTime *rt2 = PyRealTime_AS_REALTIME(other);

    if (*rt1 == *rt2) return 0;
    else if (*rt1 > *rt2) return 1;
    else return -1;
}

/* String representation called by e.g. str(realtime), print realtime*/
static PyObject *
RealTime_repr(PyObject *self)
{
    return Py_BuildValue("s",
                         ((RealTimeObject*)self)->rt->toString().c_str());
}


/* Number Protocol */
/// TODO: implement all methods available in Vamp::RealTime() objects

static PyObject *
RealTime_add(PyObject *s, PyObject *w)
{
    RealTimeObject *result = 
        PyObject_New(RealTimeObject, &RealTime_Type); 
    if (result == NULL) return NULL;

    result->rt = new RealTime(
        *((RealTimeObject*)s)->rt + *((RealTimeObject*)w)->rt);
    return (PyObject*)result;
}

static PyObject *
RealTime_subtract(PyObject *s, PyObject *w)
{
    RealTimeObject *result = 
        PyObject_New(RealTimeObject, &RealTime_Type); 
    if (result == NULL) return NULL;

    result->rt = new RealTime(
        *((RealTimeObject*)s)->rt - *((RealTimeObject*)w)->rt);
    return (PyObject*)result;
}

static PyNumberMethods realtime_as_number = 
{
    RealTime_add,                       /*nb_add*/
    RealTime_subtract,          /*nb_subtract*/
    0,                                          /*nb_multiply*/
    0,                                          /*nb_divide*/
    0,                                          /*nb_remainder*/
    0,                      /*nb_divmod*/
    0,                          /*nb_power*/
    0,                          /*nb_neg*/
    0,                          /*nb_pos*/
    0,                          /*(unaryfunc)array_abs,*/
    0,                          /*nb_nonzero*/
    0,                          /*nb_invert*/
    0,                                  /*nb_lshift*/
    0,                                  /*nb_rshift*/
    0,                                  /*nb_and*/
    0,                                  /*nb_xor*/
    0,                                  /*nb_or*/
    0,                      /*nb_coerce*/
    0,                                          /*nb_int*/
    0,                                  /*nb_long*/
    (unaryfunc)RealTime_float,/*nb_float*/
    0,                          /*nb_oct*/
    0,                          /*nb_hex*/
};

/* REAL-TIME TYPE OBJECT */

#define RealTime_alloc PyType_GenericAlloc
#define RealTime_free PyObject_Del

/* Doc:: 10.3 Type Objects */ /* static */ 
PyTypeObject RealTime_Type = 
{
    PyObject_HEAD_INIT(NULL)
    0,                          /*ob_size*/
    "vampy.RealTime",           /*tp_name*/
    sizeof(RealTimeObject),     /*tp_basicsize*/
    0,                          /*tp_itemsize*/
    /*          methods         */
    (destructor)RealTimeObject_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    (getattrfunc)RealTime_getattr, /*tp_getattr*/
    (setattrfunc)RealTime_setattr, /*tp_setattr*/
    (cmpfunc)RealTime_compare,     /*tp_compare*/
    RealTime_repr,                 /*tp_repr*/
    &realtime_as_number,        /*tp_as_number*/
    0,                          /*tp_as_sequence*/
    0,                          /*tp_as_mapping*/
    0,                          /*tp_hash*/
    0,                      /*tp_call*/
    0,                      /*tp_str*/
    0,                      /*tp_getattro*/
    0,                      /*tp_setattro*/
    0,                      /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,     /*tp_flags*/
    "RealTime object, used for Vamp plugin timestamps.",      /*tp_doc*/
    0,                      /*tp_traverse*/
    0,                      /*tp_clear*/
    0,                      /*tp_richcompare*/
    0,                      /*tp_weaklistoffset*/
    0,                      /*tp_iter*/
    0,                      /*tp_iternext*/
    RealTime_methods,       /*tp_methods*/ //TypeObject Methods
    0,                      /*tp_members*/
    0,                      /*tp_getset*/
    0,                      /*tp_base*/
    0,                      /*tp_dict*/
    0,                      /*tp_descr_get*/
    0,                      /*tp_descr_set*/
    0,                      /*tp_dictoffset*/
    0,                      /*tp_init*/
    RealTime_alloc,         /*tp_alloc*/
    RealTime_new,           /*tp_new*/
    RealTime_free,          /*tp_free*/
    0,                      /*tp_is_gc*/
};



/* PyRealTime C++ API */

/*PyRealTime from RealTime*/
PyObject* 
PyRealTime_FromRealTime(const Vamp::RealTime& rt) {

    RealTimeObject *self =
        PyObject_New(RealTimeObject, &RealTime_Type); 
    if (self == NULL) return NULL;

    self->rt = new RealTime(rt);
    return (PyObject*) self;
}

/*RealTime* from PyRealTime*/
const Vamp::RealTime*
PyRealTime_AsRealTime (PyObject *self) { 

    RealTimeObject *s = (RealTimeObject*) self; 

    if (!PyRealTime_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "RealTime Object Expected.");
        cerr << "in call PyRealTime_AsPointer(): RealTime Object Expected. " << endl;
        return NULL; }
    return s->rt; 
};


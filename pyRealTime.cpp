/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */
/*

  This module exposes a Type Object wrapping Vamp::RealTime
  together with module level functions to create 
  new pyRealTime objects from frame count, samplerate or sec,nsec tuples.

  A small API is provided for the C/C++ programmer and relevant
  functions are exposed to Python.

  TODO: implement number protocol (i.e. wrap arithmetic operators) 
  partly done

*/

#include <Python.h>
#include <pyRealTime.h>
#include "vamp-hostsdk/Plugin.h"
#include <string>

using namespace std;
using namespace Vamp;

using Vamp::Plugin;
using Vamp::RealTime;

/* REAL-TIME TYPE OBJECT */


/* Documentation for our new module */
PyDoc_STRVAR(module_doc,
             "This module is a thin wrapper around Vamp::RealTime.");


/* RealTime Object's Methods */ 
//Note: these are internals, not exposed by the module but the object

/* Returns a Tuple containing sec and nsec values */
static PyObject *
RealTime_values(RealTimeObject *self)
{
    return Py_BuildValue("(ii)",
                         self->rt->sec,self->rt->nsec);
}

/* Returns a Text representation */
static PyObject *
RealTime_toText(RealTimeObject *self, PyObject *args)
{
    return Py_BuildValue("s",
                         self->rt->toText().c_str());
}

/* String representation called by e.g. str(realtime), print realtime*/
static PyObject *
RealTime_repr(PyObject *self)
{
    return Py_BuildValue("s",
                         ((RealTimeObject*)self)->rt->toString().c_str());
}


/* Frame representation */
static PyObject *
RealTime_toFrame(PyObject *self, PyObject *args)
{
    unsigned int samplerate;
	
    if ( !PyArg_ParseTuple(args, "I:realtime.toFrame object ", 
                           (unsigned int *) &samplerate )) {
        PyErr_SetString(PyExc_ValueError, 
                        "Sample Rate Required.");
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

/* test */
static PyObject *
RealTime_test(PyObject *self)
{

    long frame = 100;
    unsigned int sampleRate = 22050;
	
    const RealTime t = RealTime::frame2RealTime(frame,sampleRate);
    long back = RealTime::realTime2Frame(t,sampleRate);
    cerr << "Reverse Conversion: " << back << endl;

    return Py_BuildValue("s",
                         ((RealTimeObject*)self)->rt->toString().c_str());
}


/* Type object's (RealTime) methods table */
static PyMethodDef RealTime_methods[] = {

    {"toText",	(PyCFunction)RealTime_toText,	METH_NOARGS,
     PyDoc_STR("toText() -> Return a user-readable string to the nearest millisecond in a form like HH:MM:SS.mmm")},

    {"values",	(PyCFunction)RealTime_values,	METH_NOARGS,
     PyDoc_STR("values() -> Tuple of sec,nsec representation.")},

    {"toFrame",	(PyCFunction)RealTime_toFrame,	METH_VARARGS,
     PyDoc_STR("toFrame(samplerate) -> Sample count for given sample rate.")},

    {"toFloat",	(PyCFunction)RealTime_float,	METH_NOARGS,
     PyDoc_STR("float() -> Floating point representation.")},

    {"test",	(PyCFunction)RealTime_test,	METH_VARARGS,
     PyDoc_STR("test() -> .")},
	
    {NULL,		NULL}		/* sentinel */
};



/* Function to set basic attributes */
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

/* Function to get basic attributes */
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


/* DESTRUCTOR: delete type object */
static void
RealTimeObject_dealloc(RealTimeObject *self)
{
    delete self->rt; 	//delete the C object
    PyObject_Del(self); //delete the Python object
}

/*					 Number Protocol 					*/


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


static PyNumberMethods realtime_as_number = {
    RealTime_add,			/*nb_add*/
    RealTime_subtract,		/*nb_subtract*/
    0,						/*nb_multiply*/
    0,				 		/*nb_divide*/
    0,						/*nb_remainder*/
    0,      	            /*nb_divmod*/
    0,                   	/*nb_power*/
    0,                  	/*nb_neg*/
    0,                		/*nb_pos*/
    0,                  	/*(unaryfunc)array_abs,*/
    0,                    	/*nb_nonzero*/
    0,                    	/*nb_invert*/
    0,       				/*nb_lshift*/
    0,      				/*nb_rshift*/
    0,      				/*nb_and*/
    0,      				/*nb_xor*/
    0,       				/*nb_or*/
    0,                      /*nb_coerce*/
    0,						/*nb_int*/
    0,				        /*nb_long*/
    (unaryfunc)RealTime_float,             /*nb_float*/
    0,               		/*nb_oct*/
    0,               		/*nb_hex*/
};

/*					 pyRealTime TypeObject 					*/


/* Doc:: 10.3 Type Objects */
/* static */ PyTypeObject RealTime_Type = {
    /* The ob_type field must be initialized in the module init function
     * to be portable to Windows without using C++. */
    PyObject_HEAD_INIT(NULL)
    0,						/*ob_size*/
    "pyRealTime.realtime",				/*tp_name*/
    sizeof(RealTimeObject),	/*tp_basicsize*/
    sizeof(RealTime),		/*tp_itemsize*/
    /*	 	methods	 	*/
    (destructor)RealTimeObject_dealloc, /*tp_dealloc*/
    0,						/*tp_print*/
    (getattrfunc)RealTime_getattr, /*tp_getattr*/
    (setattrfunc)RealTime_setattr, /*tp_setattr*/
    0,						/*tp_compare*/
    RealTime_repr,			/*tp_repr*/
    &realtime_as_number,	/*tp_as_number*/
    0,						/*tp_as_sequence*/
    0,						/*tp_as_mapping*/
    0,						/*tp_hash*/
    0,//(ternaryfunc)RealTime_new,                      /*tp_call*/
    0,                      /*tp_str*/
    0,                      /*tp_getattro*/
    0,                      /*tp_setattro*/
    0,                      /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,     /*tp_flags*/
    0,                      /*tp_doc*/
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
    0,                      /*tp_alloc*/
    0,                      /*tp_new*/
    0,			            /*tp_free*/
    0,                      /*tp_is_gc*/
};


/*		 Remaining Functions Exposed by the MODULE 					*/


/* New RealTime object from Frame (with given samplerate) */
/*static*/ PyObject *
RealTime_frame2RealTime(PyObject *ignored, PyObject *args)
{

    long frame;
    unsigned int sampleRate;

    if (!PyArg_ParseTuple(args, "lI:realtime.fame2RealTime ", 
                          &frame, 
                          &sampleRate))
        return NULL;
    /*Doc:: 5.5 Parsing arguments and building values*/

    RealTimeObject *self;
    self = PyObject_New(RealTimeObject, &RealTime_Type); 
    if (self == NULL)
        return NULL;

    self->rt = new RealTime(
	RealTime::frame2RealTime(frame,sampleRate));

    return (PyObject *) self;
}

/* New RealTime object from sec and nsec */
/*static*/ PyObject *
RealTime_new(PyObject *ignored, PyObject *args)
{

    unsigned int sec = 0;
    unsigned int nsec = 0;
    double unary = 0;
    const char *fmt = NULL;

    /*Doc:: 5.5 Parsing arguments and building values*/
    if (
		
	!PyArg_ParseTuple(args, "|sd:realtime.new ", 
                          (const char *) &fmt, 
                          (double *) &unary) 	&&

	!PyArg_ParseTuple(args, "|II:realtime.new ", 
                          (unsigned int*) &sec, 
                          (unsigned int*) &nsec) 
		
	) { 
        PyErr_SetString(PyExc_TypeError, 
                        "RealTime initialised with wrong arguments.");
        return NULL; }

    PyErr_Clear();

    RealTimeObject *self = 
	PyObject_New(RealTimeObject, &RealTime_Type); 
    if (self == NULL) return NULL;

    self->rt = NULL;

    if (sec == 0 && nsec == 0 && fmt == 0) 
        self->rt = new RealTime();
    else if (fmt == 0)
        self->rt = new RealTime(sec,nsec);
    else { 

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


/* pyRealTime Module's methods table */
static PyMethodDef Module_methods[] = {

    {"frame2RealTime",	(PyCFunction)RealTime_frame2RealTime,	METH_VARARGS,
     PyDoc_STR("frame2RealTime((int64)frame, (uint32)sampleRate ) -> returns new RealTime object from frame.")},

    {"realtime",	RealTime_new,		METH_VARARGS,
     PyDoc_STR("realtime() -> returns new RealTime object")},
	
    {NULL,		NULL}		/* sentinel */
};


/*				 PyRealTime C API functions 					*/



/*RealTime from PyRealTime*/
RealTime*
PyRealTime_AsPointer (PyObject *self) { 

    RealTimeObject *s = (RealTimeObject*) self; 

    if (!PyRealTime_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "RealTime Object Expected.");
        cerr << "in call PyRealTime_AsPointer(): RealTime Object Expected. " << endl;
        return NULL; }
    return s->rt; };

/*PyRealTime from RealTime*/
PyObject* 
PyRealTime_FromRealTime(Vamp::RealTime *rt) {

    RealTimeObject *self =
	PyObject_New(RealTimeObject, &RealTime_Type); 
    if (self == NULL) return NULL;

    self->rt = new RealTime(*rt);
    return (PyObject*) self;
    //TODO: check if we need to INCREF here
}


/* Module initialization (includes extern "C" {...}) */
PyMODINIT_FUNC
initpyRealTime(void)
{
    PyObject *m;

    /* Finalize the type object including setting type of the new type
     * object; doing it here is required for portability to Windows 
     * without requiring C++. */
    if (PyType_Ready(&RealTime_Type) < 0)
        return;

    /* Create the module and add the functions */
    m = Py_InitModule3("pyRealTime", Module_methods, module_doc);
    if (m == NULL)
        return;

//	PyModule_AddObject(m, "realtime", (PyObject *)&RealTime_Type);

}

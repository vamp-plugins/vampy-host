/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

// Moving stuff around 

static PyObject *
vampyhost_process(PyObject *self, PyObject *args)
{
//check if the plugin has been initialised
//obtain sample Rate: maybe library:identifier:channels:stepSize:blockSize
    PyObject *pyPluginHandle;
    PyObject *pyBuffer;

    if (!PyArg_ParseTuple(args, "OO", 
			  &pyPluginHandle,	// C object holding a pointer to a plugin and its descriptor
			  &pyBuffer)) {		// Audio data
	PyErr_SetString(PyExc_TypeError,
			"Required: plugin handle, buffer, timestmap.");
	return NULL; }

    string *key;	
    Plugin *plugin; 
    long frame = 0;

    if ( !getPluginHandle(pyPluginHandle, &plugin, &key) ) {
	PyErr_SetString(PyExc_AttributeError,
			"Invalid or already deleted plugin handle.");
	return NULL; }

    PyPluginDescriptor *pd = (PyPluginDescriptor*) key;

    if (!pd->isInitialised) {
	PyErr_SetString(PyExc_StandardError,
			"Plugin has not been initialised.");
	return NULL; }

    size_t channels =  pd->channels;	
    size_t blockSize = pd->blockSize;

/*
  Handle the case when we get the data as a character buffer
  Handle SampleFormats: int16, float32
	
*/
		
    if (PyString_Check(pyBuffer)) {
	cerr << ">>> String obj passed in." << endl;
    }

//	size_t chlen = sizeof(short) / sizeof(char); 

    //Assume interleaved signed 16-bit PCM data

    //int *intch = new int*[buflen/2];
    //int *intch = (int*) PyString_AS_STRING(pyBuffer);
    //short *tmpch = 
    //reinterpret_cast <short*> (PyString_AS_STRING(pyBuffer));

    typedef char int16[2]; //can we trust sizeof(short) = 2 ?
    size_t sample_size = sizeof(int16);

    long buflen = (long) PyString_GET_SIZE(pyBuffer);

    size_t input_length = 
	static_cast <size_t> (buflen/channels/sample_size);

    if (input_length == pd->blockSize) {
	cerr << ">>> A full block has been passed in." << endl; }

    int16 *input = 
	reinterpret_cast <int16*> (PyString_AS_STRING(pyBuffer));
	
    // int16 *input = new int16[buflen/sample_size];
    // input = reinterpret_cast <int16*> (PyString_AS_STRING(pyBuffer));
	
    // short *input = 
    // reinterpret_cast <short*> (PyString_AS_STRING(pyBuffer));
	
    //float ffirst = 
    //static_cast <float> (*input[1000]) /
    //static_cast <float> (SHRT_MAX);

//	int *proba[10]; -> pointer array
    int *proba = new int[10]; // -> actual array of ints
    int p = 234;
    proba[1]=p;
    size_t chlen = (size_t) buflen/2;
    //short smax = SHRT_MAX;
    cerr 
	<< " c: " << sizeof(char) 	
	<< " s: " << sizeof(short) 
	//<< " i16: " << sizeof(int16) 
	<< " i:" << sizeof(int) 
	<< " float:" << sizeof(float)
	<< " [proba]: " << proba[1]
	//<< " ffirst: " << ffirst 
	<< endl; 

    //vector<int> *intch = (vector<int>*) PyString_AS_STRING(pyBuffer);
    //size_t chlen = intch->size();
    //cerr << ">>>Size of ch buffer: " << chlen << endl;
	
    //convert int16 PCM data to 32-bit floats
    float **plugbuf = new float*[channels];
    float smax = static_cast <float> (SHRT_MAX);
		
    for (size_t c = 0; c < channels; ++c) {

	plugbuf[c] = new float[blockSize+2];
  
      	size_t j = 0;
        while (j < input_length) {
	    //int *v = (*int) input[j * channels + c];
	    //int value = 5;//input[j * channels + c];
	    // short *v = (short*) input+j;
	    // short value = *v;
	    //int *v = (int*) input+j;
	    int *v = new int;
	    *v = 0;
	    char *wc = (char*) v;
	    char *ic = (char*) input[j];
	    wc=wc+2;
	    *wc = *ic; 
	    wc++; ic++;
	    *wc = *ic; 

	    int value = *v;

	    plugbuf[c][j] =  static_cast <float> (value/100000);
// works if short=2	static_cast <float> (*input[j * channels + c]) / smax;
//			static_cast <float> (input[j * channels + c]) / smax;
	    ++j; 
        }
        while (j < blockSize) {
            plugbuf[c][j] = 0.0f;
            ++j;
        }

	//}
    }	

    const char *output = reinterpret_cast <const char*> (plugbuf[0]);
    Py_ssize_t len = (Py_ssize_t) channels*blockSize*4;
	
    PyObject* pyReturnBuffer = 
	PyString_FromStringAndSize(output,len);

    return pyReturnBuffer;


/* NOW return the data in a PyBuffer

 */

/*	
	char* test =  PyString_AS_STRING(pyBuffer);
	cerr << "Passed in: " << buflen << " str: " << test << endl;

//convert the buffer to plugbuf
		
//plugin->process
// (plugbuf, RealTime::frame2RealTime(frame, samplerate))

for(size_t k=0; k<channels; k++){
delete[] plugbuf[k];
}
delete[] plugbuf;
*/
    return pyReturnBuffer;

}

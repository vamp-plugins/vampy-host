/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

#ifndef _VAMPYHOST_H_
#define _VAMPYHOST_H_

#include "vamp-hostsdk/Plugin.h"
#include <string>

// structure of NumPy array intrface (just a hack, shouldn't be needed here...)
typedef struct {
    int two;              /* contains the integer 2 -- simple sanity check */
    int nd;               /* number of dimensions */
    char typekind;        /* kind in array --- character code of typestr */
    int itemsize;         /* size of each element */
    int flags;            /* flags indicating how the data should be interpreted */
                          /*   must set ARR_HAS_DESCR bit to validate descr */
    Py_intptr_t *shape;   /* A length-nd array of shape information */
    Py_intptr_t *strides; /* A length-nd array of stride information */
    void *data;           /* A pointer to the first element of the array */
    PyObject *descr;      /* NULL or data-description (same as descr key */
                          /*        of __array_interface__) -- must set ARR_HAS_DESCR */
                          /*        flag or this will be ignored. */
} PyArrayInterface;

//structure for holding plugin instance data
typedef struct {
    std::string key;
    std::string identifier;
    bool isInitialised;
    float inputSampleRate;
    size_t channels;
    size_t blockSize;
    size_t stepSize;
    size_t sampleSize;
    bool mixChannels;
    enum InputSampleType {
	int16,
	float32 }; 
    InputSampleType inputSampleType;
    Vamp::Plugin::FeatureSet output;
} PyPluginDescriptor;

#endif

/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

#ifndef _VAMPYHOST_H_
#define _VAMPYHOST_H_

#include "vamp-hostsdk/Plugin.h"
#include <string>

//structure for holding plugin instance data
typedef struct {
    std::string key;
    std::string identifier;
    bool isInitialised;
    float inputSampleRate;
    size_t channels;
    size_t blockSize;
    size_t stepSize;
    Vamp::Plugin::FeatureSet output;
} PyPluginDescriptor;

#endif

'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import numpy

def frames_from_array(arr, step_size, frameSize):
    """Generate a list of frames of size frameSize, extracted from the input array arr at step_size intervals"""
    # presumably such a function exists in many places, but I need practice
    assert(step_size > 0)
    if arr.ndim == 1: # turn 1d into 2d array with 1 channel
        arr = numpy.reshape(arr, (1, arr.shape[0]))
    assert(arr.ndim == 2)
    n = arr.shape[1]
    i = 0
    while (i < n):
        frame = arr[:, i : i + frameSize]
        w = frame.shape[1]
        if (w < frameSize):
            pad = numpy.zeros((frame.shape[0], frameSize - w))
            frame = numpy.concatenate((frame, pad), 1)
        yield frame
        i = i + step_size


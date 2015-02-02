'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import numpy

def frames_from_array(arr, step_size, frame_size):
    """Generate a list of frames of size frame_size, extracted from the input array arr at step_size intervals"""
    # presumably such a function exists in many places, but I need practice
    assert(step_size > 0)
    if arr.ndim == 1: # turn 1d into 2d array with 1 channel
        arr = numpy.reshape(arr, (1, arr.shape[0]))
    assert(arr.ndim == 2)
    n = arr.shape[1]
    i = 0
    while (i < n):
        frame = arr[:, i : i + frame_size]
        w = frame.shape[1]
        if (w < frame_size):
            pad = numpy.zeros((frame.shape[0], frame_size - w))
            frame = numpy.concatenate((frame, pad), 1)
        yield frame
        i = i + step_size


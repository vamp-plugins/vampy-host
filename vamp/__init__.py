#!/usr/bin/env python

#   Python Vamp Host
#   Copyright (c) 2008-2015 Queen Mary, University of London
#
#   Permission is hereby granted, free of charge, to any person
#   obtaining a copy of this software and associated documentation
#   files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies
#   of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be
#   included in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
#   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#   Except as contained in this notice, the names of the Centre for
#   Digital Music and Queen Mary, University of London shall not be
#   used in advertising or otherwise to promote the sale, use or other
#   dealings in this Software without prior written authorization.

'''
Python Vamp plugin host
=======================

This module allows Python code to load and use native-code Vamp
plugins (http://vamp-plugins.org) for audio feature analysis.

The module consists of a native-code extension ("vampyhost") that
provides a low-level wrapper for the Vamp plugin SDK, along with a
Python wrapper ("vamp") that provides a higher-level abstraction.

No code for loading audio files is included; you'll need to use some
other module for that. This code expects to receive decoded audio data
of one or more channels, either as a series of frames or as a single
whole buffer.

Written by Chris Cannam and George Fazekas at the Centre for Digital
Music, Queen Mary University of London. Copyright 2008-2015 Queen
Mary, University of London. Refer to COPYING.rst for licence details.

See home page at https://code.soundsoftware.ac.uk/projects/vampy-host
for more details.


A simple example
----------------

Using librosa (http://bmcfee.github.io/librosa/) to read an audio
file, and the NNLS Chroma Vamp plugin
(https://code.soundsoftware.ac.uk/projects/nnls-chroma/) for
analysis::

    >>> import vamp
    >>> import librosa
    >>> data, rate = librosa.load("example.wav")
    >>> chroma = vamp.collect(data, rate, "nnls-chroma:nnls-chroma")
    >>> chroma
    {'matrix': ( 0.092879819, array([[  61.0532608 ,   60.27478409,   59.3938446 , ...,  182.13394165,
              42.40084457,  116.55457306],
           [  68.8901825 ,   63.98115921,   60.77633667, ...,  245.88218689,
              68.51251984,  164.70120239],
           [  58.59794617,   50.3429184 ,   45.44804764, ...,  258.02362061,
              83.95749664,  179.91200256],
           ..., 
           [   0.        ,    0.        ,    0.        , ...,    0.        ,
               0.        ,    0.        ],
           [   0.        ,    0.        ,    0.        , ...,    0.        ,
               0.        ,    0.        ],
           [   0.        ,    0.        ,    0.        , ...,    0.        ,
               0.        ,    0.        ]], dtype=float32))}
    >>> stepsize, chromadata = chroma["matrix"]
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(chromadata)
    <matplotlib.image.AxesImage object at 0x7fe9e0043fd0>
    >>> plt.show()

And a pitch-chroma plot appears.


High-level interface (vamp)
---------------------------

This module contains three sorts of function:

1. Basic info and lookup functions
""""""""""""""""""""""""""""""""""

   * ``vamp.list_plugins``
   * ``vamp.get_outputs_of``
   * ``vamp.get_parameters_of``
   * ``vamp.get_category_of``

   These retrieve the installed plugin keys and get basic information
   about each plugin. (For more detailed information, load a plugin
   and inspect it using the low-level interface described below.)

2. Process functions
""""""""""""""""""""

   * ``vamp.process_audio``
   * ``vamp.process_frames``
   * ``vamp.process_audio_multiple_outputs``
   * ``vamp.process_frames_multiple_outputs``

   These accept audio input, and produce output in the form of a list
   of feature sets structured similarly to those in the C++ Vamp
   plugin SDK. The plugin to be used is specified by its key (the
   identifier as returned by ``vamp.list_plugins``). A dictionary of
   plugin parameter settings may optionally be supplied.

   The ``_audio`` versions take a single (presumably long) array of
   audio samples as input, and chop it into frames according to the
   plugin's preferred step and block sizes. The ``_frames`` versions
   instead accept an enumerable sequence of audio frame arrays.

3. The process-and-collect function
"""""""""""""""""""""""""""""""""""
   
   * ``vamp.collect``

   This accepts a single array of audio samples as input, and returns
   an output structure that reflects the underlying structure of the
   feature output (depending on whether it is a curve, grid, etc). The
   plugin to be used is specified by its key. A dictionary of plugin
   parameter settings may optionally be supplied.

   The ``collect`` function processes the whole input before returning
   anything; if you need to supply a streamed input, or retrieve
   results as they are calculated, then you must use one of the
   ``process`` functions (above) or else the low-level interface
   (below).


Low-level interface (vampyhost)
-------------------------------

This extension contains facilities that operate on Vamp plugins in a
way analogous to the existing C++ Vamp Host SDK: ``list_plugins``,
``get_plugin_path``, ``get_category_of``, ``get_library_for``,
``get_outputs_of``, ``load_plugin``, and a utility function
``frame_to_realtime``.

Calling ``load_plugin`` gets you a ``vampyhost.Plugin`` object, which
then exposes all of the methods found in the Vamp SDK Plugin class.

(Note that methods wrapped directly from the Vamp SDK are named using
camelCase, so as to match the names found in the C++ SDK. Elsewhere
this module follows Python PEP8 naming.)

See the individual module and function documentation for further
details.

'''

import vampyhost

from vamp.load import list_plugins, get_outputs_of, get_parameters_of, get_category_of
from vamp.process import process_audio, process_frames, process_audio_multiple_outputs, process_frames_multiple_outputs
from vamp.collect import collect


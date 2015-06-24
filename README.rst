
Python Vamp plugin host
=======================

This module allows Python code to load and use Vamp plugins for audio
feature analysis.

It consists of a native-code extension ("vampyhost") which provides a
low-level wrapper for the Vamp plugin SDK, along with a Python module
("vamp") that provides a higher-level abstraction.

No code for loading audio files etc is included; you'll need to use
some other module for that.

Written by Chris Cannam and George Fazekas at the Centre for Digital
Music, Queen Mary University of London. Copyright 2008-2015 Queen
Mary, University of London. Refer to COPYING.rst for licence details.

See home page at https://code.soundsoftware.ac.uk/projects/vampy-host
for more details.


High-level interface (vamp)
---------------------------

This module contains three sorts of function:

 * Lookup functions: list_plugins, get_outputs_of, get_category_of

   These retrieve the installed plugin identifiers and get basic
   information about each plugin. For more detailed information,
   load a plugin and inspect it (using the low-level interface).

 * Process functions: process_audio, process_frames,
   process_audio_multiple_outputs, process_frames_multiple_outputs

   These accept audio input, and produce output in the form of a list
   of feature sets structured similarly to those in the C++ Vamp
   plugin SDK. The plugin to be used is specified by its identifier.

   The _audio versions take a single (presumably long) array of audio
   samples as input, and chop it into frames according to the plugin's
   preferred step and block sizes. The _frames versions instead accept
   an enumerable sequence of audio frame arrays.

 * The process-and-collect function: collect

   This accepts a single array of audio samples as input, and returns
   an output structure that reflects the underlying structure of the
   feature output (depending on whether it is a curve, grid, etc). The
   plugin to be used is specified by its identifier.

   It processes the whole input before returning anything; if you need
   to supply a streamed input, or retrieve results as they are
   calculated, then you must use one of the process functions (above)
   or the low-level interface (below) instead.


Low-level interface (vampyhost)
-------------------------------

This module contains functions that operate on Vamp plugins in a way
analogous to the existing C++ Vamp Host SDK: list_plugins,
get_plugin_path, get_category_of, get_library_for, get_outputs_of,
load_plugin, and a utility function frame_to_realtime.

Calling load_plugin gets you a vampyhost.Plugin object, which then
exposes all of the methods found in the Vamp SDK Plugin class.

(Note that methods wrapped directly from the Vamp SDK are named using
camelCase, so as to match the SDK. Elsewhere this module follows
Python PEP8 naming.)


See the individual module and function documentation for more details.


A simple example
----------------

Using librosa (http://bmcfee.github.io/librosa/) for audio file I/O,
and the NNLS Chroma Vamp plugin
(https://code.soundsoftware.ac.uk/projects/nnls-chroma/)::

    $ python
    >>> import vamp
    >>> import librosa
    >>> data, rate = librosa.load("example.wav")
    >>> collected = vamp.collect(data, rate, "nnls-chroma:nnls-chroma")
    >>> collected
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
    >>> stepsize, chromadata = collected["matrix"]
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(chromadata)
    <matplotlib.image.AxesImage object at 0x7fe9e0043fd0>
    >>> plt.show()

And a pitch-chroma plot appears (though it's rotated 90 degrees
compared with its more usual orientation).


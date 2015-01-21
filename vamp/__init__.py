'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

from load import list_plugins, load_and_configure
from frames import frames_from_array
from process import process, process_multiple_outputs
from collect import collect

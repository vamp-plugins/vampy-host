'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

from load import list_plugins
from process import process, process_frames, process_multiple_outputs
from collect import collect


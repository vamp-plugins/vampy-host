'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

from load import list_plugins, get_outputs_of, get_category_of
from process import process, process_frames, process_multiple_outputs, process_frames_multiple_outputs
from collect import collect


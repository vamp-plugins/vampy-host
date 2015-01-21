'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

from load import list_plugins, loadAndConfigureFor
from frames import framesFromArray
from process import process, processMultipleOutputs
from collect import collect

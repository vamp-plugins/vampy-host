'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

def list_plugins():
    """Obtain a list of plugin keys for all currently installed Vamp plugins.

    The returned value is a list of strings, each of which is the key
    for one plugin. (Note that a plugin may have multiple outputs, if
    it computes more than one type of feature.)

    To query the available outputs and category of a plugin, you may
    use vamp.get_outputs_of() and vamp.get_category_of(). Further
    information may be retrieved by loading the plugin and querying
    its info dictionary using the low-level functions in the
    vamp.vampyhost extension module.

    To make use of a plugin to extract features from audio data, pass
    the plugin key and optionally an output identifier to
    vamp.process() or vamp.collect().
    """
    return vampyhost.list_plugins()

def get_outputs_of(key):
    """Obtain a list of the output identifiers for the given plugin key.
    """
    return vampyhost.get_outputs_of(key)

def get_category_of(key):
    """Obtain the category descriptor, if any, for the given plugin key.

    The returned value is a list of descriptor terms, from least
    specific to most specific. The list may be empty if no category
    information is found for the plugin.
    """
    return vampyhost.get_category_of(key)

def load_and_configure(data, sample_rate, key, parameters):
    """Load the plugin with the given key at a given sample rate,
    configure it with the parameter keys and values in the given
    parameter dictionary, and initialise it with its preferred step
    and block size. The channel count is taken from the shape of the
    data array provided.
    """

    plug = vampyhost.load_plugin(key, sample_rate,
                                 vampyhost.ADAPT_INPUT_DOMAIN +
                                 vampyhost.ADAPT_CHANNEL_COUNT)

    plug.set_parameter_values(parameters)

    step_size = plug.get_preferred_step_size()
    block_size = plug.get_preferred_block_size()

    if block_size == 0:
        block_size = 1024
    if step_size == 0:
        step_size = block_size ##!!! or block_size/2, but check this with input domain adapter

    channels = 1
    if data.ndim > 1:
        channels = data.shape[0]

    if plug.initialise(channels, step_size, block_size):
        return (plug, step_size, block_size)
    else:
        raise "Failed to initialise plugin"

'''A high-level interface to the vampyhost extension module, for quickly and easily running Vamp audio analysis plugins on audio files and buffers.'''

import vampyhost

def list_plugins():
    return vampyhost.list_plugins()

def load_and_configure(data, sample_rate, key, parameters):

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

    plug.initialise(channels, step_size, block_size)
    return (plug, step_size, block_size)


from distutils.core import setup, Extension
import numpy as np

sdkdir = 'vamp-plugin-sdk/src/vamp-hostsdk/'
vpydir = 'native/'

sdkfiles = [ 'Files', 'PluginBufferingAdapter', 'PluginChannelAdapter',
             'PluginHostAdapter', 'PluginInputDomainAdapter', 'PluginLoader',
             'PluginSummarisingAdapter', 'PluginWrapper', 'RealTime' ]
vpyfiles = [ 'PyPluginObject', 'PyRealTime', 'VectorConversion', 'vampyhost' ]

srcfiles = [ sdkdir + f + '.cpp' for f in sdkfiles ] + [ vpydir + f + '.cpp' for f in vpyfiles ]

vampyhost = Extension('vampyhost',
                      sources = srcfiles,
                      define_macros = [ ('_USE_MATH_DEFINES', 1) ],
                      include_dirs = [ 'vamp-plugin-sdk', np.get_include() ])

setup (name = 'vamp',
       version = '1.0',
       description = 'This module allows Python code to load and use Vamp plugins for audio feature analysis.',
       requires = [ 'numpy' ],
       ext_modules = [ vampyhost ])

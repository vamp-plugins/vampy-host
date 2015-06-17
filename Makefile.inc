
LIBRARY_EXT	?= .so
LIBRARY		?= vampyhost$(LIBRARY_EXT)
CXX		?= g++
CC		?= gcc

SRC_DIR		:= native

PLUGIN_EXT	?= $(LIBRARY_EXT)
TESTPLUG_DIR	:= test/vamp-test-plugin
TESTPLUG	:= $(TESTPLUG_DIR)/vamp-test-plugin$(PLUGIN_EXT)

HEADERS		:= $(SRC_DIR)/PyPluginObject.h $(SRC_DIR)/PyRealTime.h $(SRC_DIR)/FloatConversion.h $(SRC_DIR)/VectorConversion.h
SOURCES		:= $(SRC_DIR)/PyPluginObject.cpp $(SRC_DIR)/PyRealTime.cpp $(SRC_DIR)/VectorConversion.cpp $(SRC_DIR)/vampyhost.cpp

PY		:= $(wildcard vamp/*.py)
TESTS		:= $(wildcard test/test_*.py)

OBJECTS		:= $(SOURCES:.cpp=.o)
OBJECTS		:= $(OBJECTS:.c=.o)

default:	$(LIBRARY)

all:		$(LIBRARY) .tests

.tests:		$(LIBRARY) $(PY) $(TESTPLUG) $(TESTS)
		VAMP_PATH=$(TESTPLUG_DIR) $(NOSE)
		@touch $@

.PHONY:		test
test:		
		VAMP_PATH=$(TESTPLUG_DIR) $(NOSE)

$(LIBRARY):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

$(TESTPLUG):	
		$(MAKE) -C $(TESTPLUG_DIR) -f Makefile$(MAKEFILE_EXT) VAMPSDK_DIR=../../../vamp-plugin-sdk
 
clean:		
		$(MAKE) -C $(TESTPLUG_DIR) -f Makefile$(MAKEFILE_EXT) clean
		rm -f $(OBJECTS) .tests

distclean:	clean
		$(MAKE) -C $(TESTPLUG_DIR) -f Makefile$(MAKEFILE_EXT) distclean
		rm -f $(LIBRARY)

depend:
		$(MAKE) -C $(TESTPLUG_DIR) -f Makefile$(MAKEFILE_EXT) depend
		makedepend -Y -fMakefile.inc $(SOURCES) $(HEADERS)

# DO NOT DELETE

native/PyPluginObject.o: native/PyPluginObject.h native/FloatConversion.h
native/PyPluginObject.o: native/VectorConversion.h native/StringConversion.h
native/PyPluginObject.o: native/PyRealTime.h
native/PyRealTime.o: native/PyRealTime.h
native/VectorConversion.o: native/VectorConversion.h native/FloatConversion.h
native/VectorConversion.o: native/StringConversion.h
native/vampyhost.o: native/PyRealTime.h native/PyPluginObject.h
native/vampyhost.o: native/VectorConversion.h native/StringConversion.h

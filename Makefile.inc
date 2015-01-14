
LIBRARY_EXT	?= .so
LIBRARY		?= vampyhost$(LIBRARY_EXT)
CXX		?= g++
CC		?= gcc

SRC_DIR		:= native

HEADERS		:= $(SRC_DIR)/PyPluginObject.h $(SRC_DIR)/PyRealTime.h $(SRC_DIR)/FloatConversion.h $(SRC_DIR)/VectorConversion.h
SOURCES		:= $(SRC_DIR)/PyPluginObject.cpp $(SRC_DIR)/PyRealTime.cpp $(SRC_DIR)/VectorConversion.cpp $(SRC_DIR)/vampyhost.cpp
TESTS		:= $(wildcard test/test_*.py)

OBJECTS		:= $(SOURCES:.cpp=.o)
OBJECTS		:= $(OBJECTS:.c=.o)

all:		$(LIBRARY) .tests

$(LIBRARY):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

.tests:		$(LIBRARY) $(TESTS)
		$(NOSE)
		@touch $@

clean:		
		rm -f $(OBJECTS)

distclean:	clean
		rm -f $(LIBRARY)

depend:
	makedepend -Y -fMakefile.inc $(SOURCES) $(HEADERS)

# DO NOT DELETE

native/PyPluginObject.o: native/PyPluginObject.h native/FloatConversion.h
native/PyPluginObject.o: native/VectorConversion.h native/PyRealTime.h
native/PyRealTime.o: native/PyRealTime.h
native/VectorConversion.o: native/FloatConversion.h native/VectorConversion.h
native/vampyhost.o: native/PyRealTime.h native/PyPluginObject.h
native/vampyhost.o: native/VectorConversion.h

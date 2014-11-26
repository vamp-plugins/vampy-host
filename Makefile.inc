
LIBRARY_EXT	?= .so
LIBRARY		?= vampyhost$(LIBRARY_EXT)
CXX		?= g++
CC		?= gcc

HEADERS		:= PyPluginObject.h PyRealTime.h VectorConversion.h
SOURCES		:= PyPluginObject.cpp PyRealTime.cpp VectorConversion.cpp vampyhost.cpp

OBJECTS		:= $(SOURCES:.cpp=.o)
OBJECTS		:= $(OBJECTS:.c=.o)

$(LIBRARY):	$(OBJECTS)
		$(CXX) -o $@ $^ $(LDFLAGS)

clean:		
		rm -f $(OBJECTS)

distclean:	clean
		rm -f $(LIBRARY)

depend:
	makedepend -Y -fMakefile.inc $(SOURCES) $(HEADERS)

# DO NOT DELETE

PyRealTime.o: PyRealTime.h
vampyhost.o: PyRealTime.h VectorConversion.h
VectorConversion.o: VectorConversion.h

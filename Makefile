
PY_INCLUDE_PATH	:= /usr/include/python2.7
NUMPY_INCLUDE_PATH := /usr/lib/python2.7/site-packages/numpy/core/include

CFLAGS		:= -DHAVE_NUMPY -g -fPIC -Wall -Werror -I$(PY_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I.
CXXFLAGS	:= -DHAVE_NUMPY -g -fPIC -Wall -Werror -I$(PY_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I.

LDFLAGS		:= -shared -Wl,-Bstatic -lvamp-hostsdk -Wl,-Bdynamic -Wl,-z,defs -lpython2.7 -ldl

OBJECTS	:= PyRealTime.o VectorConversion.o vampyhost.o

all: vampyhost.so

vampyhost.so: $(OBJECTS)
	g++ -o $@ -shared $^ $(LDFLAGS)

clean:	
	rm -f *.o *.so *.a

depend:
	makedepend -Y -fMakefile *.cpp *.h


# DO NOT DELETE

PyRealTime.o: PyRealTime.h
vampyhost.o: PyRealTime.h VectorConversion.h
VectorConversion.o: VectorConversion.h

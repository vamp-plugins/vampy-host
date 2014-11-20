
PY_INCLUDE_PATH	:= /usr/include/python2.7
NUMPY_INCLUDE_PATH := /usr/lib/python2.7/site-packages/numpy/core/include

CFLAGS		:= -DHAVE_NUMPY -D_VAMP_PLUGIN_IN_HOST_NAMESPACE=1 -O2 -fPIC -Wall -Werror -Ivampy -I$(PY_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I.
CXXFLAGS	:= -DHAVE_NUMPY -D_VAMP_PLUGIN_IN_HOST_NAMESPACE=1 -O2 -fPIC -Wall -Werror -Ivampy -I$(PY_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I.

LDFLAGS		:= -shared -Wl,-Bstatic -lvamp-hostsdk -Wl,-Bdynamic -Wl,-z,defs -lpython2.7 -ldl

OBJECTS	:= vampy/PyRealTime.o vampy/PyFeature.o vampy/PyFeatureSet.o vampy/PyTypeConversions.o vampyhost.o

all: vampyhost.so

vampyhost.so: $(OBJECTS)
	g++ -o $@ -shared $^ $(LDFLAGS)

clean:	
	rm -f vampy/*.o *.o *.so *.a

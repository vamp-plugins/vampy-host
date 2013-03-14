
PY_INCLUDE_PATH	:= /usr/include/python2.7
NUMPY_INCLUDE_PATH := /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include

CFLAGS		:= -O2 -fPIC -Wall -I$(PY_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I.
CXXFLAGS	:= -O2 -fPIC -Wall -I$(PY_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I. -I../vamp-plugin-sdk

LDFLAGS		:= -shared -L../vamp-plugin-sdk -lpython2.7 -lvamp-hostsdk
#LDFLAGS		:= -dynamiclib -lpython2.5 /usr/lib/libvamp-hostsdk.a

all: pyRealTime.so vampyhost.so

pyRealTime.a: pyRealTime.o
	ar r $@ pyRealTime.o

pyRealTime.so: pyRealTime.o
	g++ -shared $^ -o $@ $(LDFLAGS)

vampyhost.so: vampyhost.o pyRealTime.a
	g++ -o $@ -shared $^ $(LDFLAGS)

clean:
	rm *.o
	rm *.so
	rm *.a

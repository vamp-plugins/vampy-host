
PY_INCLUDE_PATH	:= /usr/include/python2.7

CFLAGS		:= -O2 -Wall -I$(PY_INCLUDE_PATH) -I.
CXXFLAGS	:= -O2 -Wall -I$(PY_INCLUDE_PATH) -I.

LDFLAGS		:= -shared -lpython2.7 -lvamp-hostsdk
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

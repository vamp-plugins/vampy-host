
CFLAGS		:= -O2 -Wall -I/usr/include/python2.5 -I/usr/include/vamp-sdk/hostext/ -I/usr/include/vamp-sdk/ -I/Users/Shared/Development/vampy-host-experiments/
CXXFLAGS	:= -O2 -Wall -I/usr/include/python2.5 -I/usr/include/vamp-sdk/hostext/ -I/usr/include/vamp-sdk/ -I/Users/Shared/Development/vampy-host-experiments/
LDFLAGS		:= -dynamiclib -lpython2.5 /usr/lib/libvamp-hostsdk.a


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
		

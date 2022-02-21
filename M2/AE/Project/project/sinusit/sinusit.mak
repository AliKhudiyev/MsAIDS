

UNAME := $(shell uname)

ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifneq ("$(OS)","")
	EZ_PATH=../../
endif

sinusitLIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =  -std=c++14 -fopenmp -O2 -g -Wall -fmessage-length=0 -I$(sinusitLIB_PATH)include

OBJS = sinusit.o sinusitIndividual.o 

LIBS = -lpthread -fopenmp 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :


#END OF USER MAKEFILE OPTIONS

TARGET =	sinusit

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(sinusitLIB_PATH)/libeasea.a $(LIBS)

	
#%.o:%.cpp
#	$(CXX) -c $(CXXFLAGS) $^

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET).exe
else
	rm -f $(OBJS) $(TARGET)
endif
easeaclean:
ifneq ("$(OS)","")
	-del $(TARGET).exe *.o *.cpp *.hpp sinusit.png sinusit.dat sinusit.prm sinusit.mak Makefile sinusit.vcproj sinusit.csv sinusit.r sinusit.plot sinusit.pop
else
	rm -f $(TARGET) *.o *.cpp *.hpp sinusit.png sinusit.dat sinusit.prm sinusit.mak Makefile sinusit.vcproj sinusit.csv sinusit.r sinusit.plot sinusit.pop
endif




UNAME := $(shell uname)

ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifneq ("$(OS)","")
	EZ_PATH=../../
endif

mini_projectLIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =  -std=c++14 -fopenmp -O2 -g -Wall -fmessage-length=0 -I$(mini_projectLIB_PATH)include

OBJS = mini_project.o mini_projectIndividual.o 

LIBS = -lpthread -fopenmp 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :


#END OF USER MAKEFILE OPTIONS

TARGET =	mini_project

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(mini_projectLIB_PATH)/libeasea.a $(LIBS)

	
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
	-del $(TARGET).exe *.o *.cpp *.hpp mini_project.png mini_project.dat mini_project.prm mini_project.mak Makefile mini_project.vcproj mini_project.csv mini_project.r mini_project.plot mini_project.pop
else
	rm -f $(TARGET) *.o *.cpp *.hpp mini_project.png mini_project.dat mini_project.prm mini_project.mak Makefile mini_project.vcproj mini_project.csv mini_project.r mini_project.plot mini_project.pop
endif


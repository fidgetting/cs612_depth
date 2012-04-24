
BIN_DIR = ../bin
INC_DIR = ../include

CXX     = g++
CFLAGS  = -g -O3 -std=c++0x `pkg-config opencv --cflags` \
                            `pkg-config gio-2.0 --cflags` \
                            `pkg-config glib-2.0 --cflags`
INCPATH = -I$(INC_DIR) -I../segment
EXE     = ../depth
LIBS    = `pkg-config opencv --libs` \
          `pkg-config gio-2.0 --libs` \
          `pkg-config glib-2.0 --libs` \
          -lboost_program_options -lboost_filesystem -lboost_system

OBJS = $(BIN_DIR)/main.o     \
       $(BIN_DIR)/util.o     \
       $(BIN_DIR)/super.o    \
       $(BIN_DIR)/ground.o   \
       $(BIN_DIR)/knearest.o

HEAD = Makefile              \
       $(INC_DIR)/util.h     \
       $(INC_DIR)/super.h    \
       $(INC_DIR)/ground.h   \
       $(INC_DIR)/knearest.h

all: $(EXE)
$(EXE): $(OBJS) $(BIN_DIR)/*.o $(HEAD)
	$(CXX) -o $(EXE) $(BIN_DIR)/*.o $(LIBS)

$(OBJS): $(BIN_DIR)/%.o: %.cpp $(HEAD)
	@echo "    build" $<
	$(CXX) -c $(DEF) $(INCPATH) $(CFLAGS) $< -o $@

clean:
	rm -f $(EXE) $(OBJS)


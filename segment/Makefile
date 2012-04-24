
BIN_DIR = ../bin
INC_DIR = .

CXX     = g++
CFLAGS  = -g -O3
INCPATH = -I$(INC_DIR)

OBJS = $(BIN_DIR)/convolve.o      \
       $(BIN_DIR)/disjoint-set.o   \
       $(BIN_DIR)/filter.o        \
       $(BIN_DIR)/imconv.o        \
       $(BIN_DIR)/segment-image.o \

all: $(OBJS)
$(OBJS): $(BIN_DIR)/%.o: %.cpp
	@echo "    build" $<
	$(CXX) -c $(DEF) $(INCPATH) $(CFLAGS) $< -o $@ 

clean:
	rm -f $(OBJS)

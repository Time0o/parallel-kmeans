# directories
C_SRC_DIR = c
CPP_SRC_DIR = cpp

C_OBJ_DIR = $(C_SRC_DIR)/obj
CPP_OBJ_DIR = $(CPP_SRC_DIR)/obj

INCLUDE_DIR = include
BUILD_DIR = build
IMAGE_DIR = images

# files
_DEPS = kmeans.h
DEPS = $(patsubst %, $(INCLUDE_DIR)/%, $(_DEPS))

_C_OBJ = kmeans.o
C_OBJ = $(patsubst %, $(C_OBJ_DIR)/%, $(_C_OBJ))

_CPP_OBJ = kmeans_wrapper.o
CPP_OBJ = $(patsubst %, $(CPP_OBJ_DIR)/%, $(_CPP_OBJ))

# compilation options
CFLAGS_COMMON = -Wall -g -O3 -I$(INCLUDE_DIR) `pkg-config --cflags opencv` 
CFLAGS_CPP = -std=c++11

LFLAGS = `pkg-config --libs opencv` 

# demo
DEMO_IMAGE = stallman_small.jpg
DEMO_CLUSTERS = 10

# rules
kmeans_demo: $(C_OBJ) $(CPP_OBJ)
	g++ -o $(BUILD_DIR)/$@ $(C_OBJ) $(CPP_OBJ) $(LFLAGS)
	./$(BUILD_DIR)/$@ $(IMAGE_DIR)/$(DEMO_IMAGE) $(DEMO_CLUSTERS)

$(C_OBJ_DIR)/%.o: $(C_SRC_DIR)/%.c $(DEPS)
	gcc -c -o $@ $< $(CFLAGS_COMMON)

$(CPP_OBJ_DIR)/%.o: $(CPP_SRC_DIR)/%.cc $(DEPS)
	g++ -c -o $@ $< $(CFLAGS_COMMON) $(CFLAGS_CPP)

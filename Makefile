# directories
C_SRC_DIR=c/src
C_OBJ_DIR=c/obj
C_INCLUDE_DIR=c/include

CPP_SRC_DIR=cpp/src
CPP_OBJ_DIR=cpp/obj
CPP_INCLUDE_DIR=cpp/include

CONFIG_DIR=config
BUILD_DIR=build
IMAGE_DIR=images

CC_C=gcc
CC_CPP=g++

# files
_CONFIG_DEPS=kmeans_config.h
CONFIG_DEPS=$(patsubst %, $(CONFIG_DIR)/%, $(_CONFIG_DEPS))

_C_DEPS=kmeans.h
C_DEPS=$(patsubst %, $(C_INCLUDE_DIR)/%, $(_C_DEPS))

_CPP_DEPS=kmeans_wrapper.h
CPP_DEPS=$(patsubst %, $(CPP_INCLUDE_DIR)/%, $(_CPP_DEPS))

DEPS=$(CONFIG_DEPS) $(C_DEPS) $(CPP_DEPS)

_C_OBJ=kmeans.o
C_OBJ=$(patsubst %, $(C_OBJ_DIR)/%, $(_C_OBJ))

_CPP_OBJ=kmeans_demo.o kmeans_wrapper.o
CPP_OBJ=$(patsubst %, $(CPP_OBJ_DIR)/%, $(_CPP_OBJ))

# flags
COMMON_CFLAGS=-Wall -g -O0 -I$(CONFIG_DIR) -I$(C_INCLUDE_DIR) \
  -I$(CPP_INCLUDE_DIR) -fopenmp

C_CFLAGS=-std=c99 $(COMMON_CFLAGS)
CPP_CFLAGS=-std=c++11 $(COMMON_CFLAGS) `pkg-config --cflags opencv`

LFLAGS=`pkg-config --libs opencv` -fopenmp

# demo parameters
DEMO_CLUSTERS=3
DEMO_IMAGE=demo_image.jpg

# rules
all: $(C_OBJ) $(CPP_OBJ)

demo: all
	$(CC_CPP) -o $(BUILD_DIR)/$@ $(C_OBJ) $(CPP_OBJ) $(LFLAGS)
	@chmod +x $(BUILD_DIR)/$@
	OMP_NUM_THREADS=4 ./$(BUILD_DIR)/$@ $(IMAGE_DIR)/$(DEMO_IMAGE) $(DEMO_CLUSTERS)

$(C_OBJ_DIR)/%.o: $(C_SRC_DIR)/%.c $(C_DEPS) $(CONFIG_DEPS)
	$(CC_C) -c -o $@ $< $(C_CFLAGS)

$(CPP_OBJ_DIR)/%.o: $(CPP_SRC_DIR)/%.cc $(DEPS)
	$(CC_CPP) -c -o $@ $< $(CPP_CFLAGS)

.PHONY: clean
clean:
	-@rm $(C_OBJ_DIR)/*.o 2> /dev/null || true
	-@rm $(CPP_OBJ_DIR)/*.o 2> /dev/null || true
	-@rm $(BUILD_DIR)/* 2> /dev/null || true

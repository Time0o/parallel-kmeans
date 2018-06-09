# directories
C_SRC_DIR=c/src
C_OBJ_DIR=c/obj
C_INCLUDE_DIR=c/include

CPP_SRC_DIR=cpp/src
CPP_OBJ_DIR=cpp/obj
CPP_INCLUDE_DIR=cpp/include

CONFIG_DIR=config
BENCHMARK_OUT_DIR=benchmarks
BUILD_DIR=build
IMAGE_DIR=images
REPORT_DIR=report
REPORT_AUX_DIR=$(REPORT_DIR)/aux
REPORT_PDF_DIR=$(REPORT_DIR)/pdf
REPORT_RESOURCE_DIR=$(REPORT_DIR)/resources
REPORT_TEX_DIR=$(REPORT_DIR)/tex

# files
_CONFIG_DEPS=kmeans_config.h
CONFIG_DEPS=$(patsubst %, $(CONFIG_DIR)/%, $(_CONFIG_DEPS))

_C_DEPS=kmeans.h
C_DEPS=$(patsubst %, $(C_INCLUDE_DIR)/%, $(_C_DEPS))

_CPP_DEPS=kmeans_wrapper.h
CPP_DEPS=$(patsubst %, $(CPP_INCLUDE_DIR)/%, $(_CPP_DEPS))

DEPS=$(CONFIG_DEPS) $(C_DEPS) $(CPP_DEPS)

# compilation settings
CC_C=gcc-8
CC_CPP=g++

COMMON_CFLAGS=-Wall -g -O0 -I$(CONFIG_DIR) -I$(C_INCLUDE_DIR) -I$(CPP_INCLUDE_DIR) -fopenmp

C_CFLAGS=-std=c99 $(COMMON_CFLAGS)
CPP_CFLAGS=-std=c++11 $(COMMON_CFLAGS) `pkg-config --cflags opencv`

LFLAGS=`pkg-config --libs opencv` -fopenmp

#functions
define run_pdflatex
	pdflatex -halt-on-error -shell-escape -output-directory $(REPORT_AUX_DIR) $< > /dev/null
endef

# profiling parameters
PROFILE_IMAGE=$(IMAGE_DIR)/profile_image.jpg
PROFILE_CLUSTERS=5

# benchmark parameters
BENCHMARK_DIM_MIN=10
BENCHMARK_DIM_MAX=100
BENCHMARK_DIM_STEP=10
BENCHMARK_CLUSTER_MIN=10
BENCHMARK_CLUSTER_MAX=10
BENCHMARK_CLUSTER_STEP=1
BENCHMARK_N_EXEC=100
BENCHMARK_PLOT=tool/plot.py

# demo parameters
DEMO_IMAGE=$(IMAGE_DIR)/demo_image.jpg
DEMO_CLUSTERS=5

# rules
all: $(BUILD_DIR)/demo $(BUILD_DIR)/profile $(BUILD_DIR)/benchmark

$(BUILD_DIR)/demo: $(CPP_OBJ_DIR)/kmeans_demo.o $(C_OBJ_DIR)/kmeans.o $(CPP_OBJ_DIR)/kmeans_wrapper.o
	$(CC_CPP) -o $@ $^ $(LFLAGS)

$(BUILD_DIR)/profile: $(CPP_OBJ_DIR)/kmeans_profile.o $(C_OBJ_DIR)/kmeans_profile.o $(CPP_OBJ_DIR)/kmeans_wrapper.o
	$(CC_CPP) -o $@ $^ $(LFLAGS)

$(BUILD_DIR)/benchmark: $(CPP_OBJ_DIR)/kmeans_benchmark.o $(C_OBJ_DIR)/kmeans.o $(CPP_OBJ_DIR)/kmeans_wrapper.o
	$(CC_CPP) -o $@ $^ $(LFLAGS)

$(C_OBJ_DIR)/kmeans_profile.o: $(C_SRC_DIR)/kmeans.c $(C_DEPS) $(CONFIG_DEPS)
	$(CC_C) -c -o $@ $< $(C_CFLAGS) -DPROFILE

$(C_OBJ_DIR)/%.o: $(C_SRC_DIR)/%.c $(C_DEPS) $(CONFIG_DEPS)
	$(CC_C) -c -o $@ $< $(C_CFLAGS)

$(CPP_OBJ_DIR)/%.o: $(CPP_SRC_DIR)/%.cc $(DEPS)
	$(CC_CPP) -c -o $@ $< $(CPP_CFLAGS)

report: $(REPORT_PDF_DIR)/report.pdf

$(REPORT_PDF_DIR)/report.pdf: $(REPORT_TEX_DIR)/report.tex $(REPORT_RESOURCE_DIR)/*
	$(call run_pdflatex)
	$(call run_pdflatex)
	-@mv $(REPORT_AUX_DIR)/report.pdf $(REPORT_PDF_DIR)

.PHONY: demo, profile, benchmark, clean

demo: $(BUILD_DIR)/demo $(DEMO_IMAGE)
	./$(BUILD_DIR)/demo $(DEMO_IMAGE) $(DEMO_CLUSTERS)

profile: $(BUILD_DIR)/profile $(PROFILE_IMAGE)
	./$(BUILD_DIR)/profile $(PROFILE_IMAGE) $(PROFILE_CLUSTERS)

benchmark: $(BUILD_DIR)/benchmark $(BENCHMARK_PLOT)
	./$(BUILD_DIR)/benchmark \
	$(BENCHMARK_DIM_MIN) $(BENCHMARK_DIM_MAX) $(BENCHMARK_DIM_STEP) \
	$(BENCHMARK_CLUSTER_MIN) $(BENCHMARK_CLUSTER_MAX) $(BENCHMARK_CLUSTER_STEP) \
	$(BENCHMARK_N_EXEC) $(BENCHMARK_OUT_DIR)
	./$(BENCHMARK_PLOT) $(BENCHMARK_OUT_DIR)

clean:
	rm $(C_OBJ_DIR)/*.o 2> /dev/null || true
	rm $(CPP_OBJ_DIR)/*.o 2> /dev/null || true
	rm $(BUILD_DIR)/* 2> /dev/null || true
	rm $(REPORT_AUX_DIR)/* 2> /dev/null || true
	rm $(REPORT_PDF_DIR)/* 2> /dev/null || true

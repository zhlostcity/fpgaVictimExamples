############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO)  "  make test DEVICE=<FPGA platform>"
	$(ECHO)  "     Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO)  ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""

############################## Setting up Project Variables ##############################
# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%ocl_kernels/cl_hotspot/*}')
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGET := hw
HOST_ARCH := x86
SYSROOT := 

include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

# SoC variables
RUN_APP_SCRIPT = ./run_app.sh
PACKAGE_OUT = ./package.$(TARGET)

LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh
RESULT_STRING = TEST PASSED

VPP := v++
SDCARD := sd_card

include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11
LDFLAGS += $(opencl_LDFLAGS)

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I$(ABS_COMMON_REPO)/common/includes/xcl2
HOST_SRCS += $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.cpp ./src/host.cpp 
# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += 
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps 
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif




# Adding config files to linker

EXECUTABLE = ./cl_hotspot
CMD_ARGS = -x $(BUILD_DIR)/hotspot.xclbin
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/hotspot.xclbin
BINARY_CONTAINER_hotspot_OBJS += $(TEMP_DIR)/hotspot.xo
BINARY_CONTAINER_hotspot_OBJS += $(TEMP_DIR)/comm_kernel.xo

############################## Setting Targets ##############################
CP = cp -rf
DATA = ./data

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis $(BINARY_CONTAINERS)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/hotspot.xo: src/hotspot.cl
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k hotspot $(CLFLAGS) --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/comm_kernel.xo: src/hotspot.cl
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k comm_kernel $(CLFLAGS) --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(BUILD_DIR)/hotspot.xclbin: $(BINARY_CONTAINER_hotspot_OBJS)
	mkdir -p $(BUILD_DIR)
ifeq ($(HOST_ARCH), x86)
	$(VPP) $(VPP_FLAGS) -l $(LDCLFLAGS) $(CLFLAGS) --temp_dir $(BUILD_DIR)  -o'$(BUILD_DIR)/hotspot.link.xclbin' $(+)
	$(VPP) -p $(BUILD_DIR)/hotspot.link.xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/hotspot.xclbin
else
	$(VPP) -l $(LDCLFLAGS) $(CLFLAGS) --temp_dir $(BUILD_DIR) -o'$(BUILD_DIR)/hotspot.xclbin' $(+)
endif

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(findstring u50, $(DEVICE)), u50)
$(error This example is not supported for $(DEVICE))
endif
ifeq ($(findstring vck, $(DEVICE)), vck)
$(error This example is not supported for $(DEVICE))
endif
ifeq ($(findstring zc, $(DEVICE)), zc)
$(error This example is not supported for $(DEVICE))
endif
ifeq ($(findstring samsung, $(DEVICE)), samsung)
$(error This example is not supported for $(DEVICE))
endif

ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) -x $(BUILD_DIR)/hotspot.xclbin
else
	$(ABS_COMMON_REPO)/common/utility/run_emulation.pl "${LAUNCH_EMULATOR} | tee run_app.log" "${RUN_APP_SCRIPT} $(TARGET)" "${RESULT_STRING}" "7"
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) -x $(BUILD_DIR)/hotspot.xclbin
endif
endif


.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) -x $(BUILD_DIR)/hotspot.xclbin
else
	$(ABS_COMMON_REPO)/common/utility/run_emulation.pl "${LAUNCH_EMULATOR} | tee embedded_run.log" "${RUN_APP_SCRIPT} $(TARGET)" "${RESULT_STRING}" "7"
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) -x $(BUILD_DIR)/hotspot.xclbin
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin
	-$(RMDIR) ./output.bmp 
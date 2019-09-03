#define GLFW_EXPOSE_NATIVE_WGL true
#define GLFW_EXPOSE_NATIVE_WIN32 true
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true

#pragma comment(lib, "OpenCL.lib")

#pragma OPENCL EXTENSION cl_khr_icd : enable

#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable

#include <CL/cl.h>
#include <glew.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <CL/cl_gl_ext.h>
#include <CL/cl_gl.h>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

struct SIM {
	int PLATE_WIDTH, PLATE_HEIGHT;
	float externalTemperature, sourceTemperature;
	int sourceX, sourceY;
};

static SIM setup;

static float tempScale = 1000;

static cl_int err;
static cl_platform_id platform_id;
static cl_device_id device_id;
static cl_context context;
static cl_program program;
static cl_command_queue command_queue;
static cl_event event;
static cl_kernel kernel;

static cl_mem sourceMemory;
static cl_mem destMemory;
static cl_sampler sampler;

void generateSetup(int WINDOW_WIDTH, int WINDOW_HEIGHT, float extTemp, float srcTemp, int sourceX, int sourceY);

void updateBuffer(float * buffer);

void updateSourcePos(int x, int y);

void fillBuffer(float* buffer, float value);

void setupOpenCLContext(cl_device_type dev_type, GLFWwindow* window);

void prepareCLKernel(cl_GLuint sourceTexture, cl_GLuint destTexture);

void updateBufferGPU(cl_int xRes, cl_int yRes);
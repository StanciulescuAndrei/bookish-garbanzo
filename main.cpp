// ThermalPropagation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define GLFW_EXPOSE_NATIVE_WGL true
#define GLFW_EXPOSE_NATIVE_WIN32 true
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#pragma comment(lib, "OpenCL.lib")
#pragma OPENCL EXTENSION cl_khr_icd : enable
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable

#include <glew.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#define PRINT_DEVICE_TYPE(type_str) { printf("- Device type: %s" "\n", (type_str)); break; }
#define SAFE_OCL_CALL(call)			do { cl_int status = (call); if (status != CL_SUCCESS) { printf("Error calling \"%s\" (%s:%d): %s\n", #call, __FILE__, __LINE__, GetErrorString(status));} } while (0)

#include <CL/cl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_gl.h>

//OpenGL variables:
int scaling = 1;
GLuint texture;
GLuint backTexture;
GLfloat* textureBuffer = NULL;
GLuint VBO, VAO;
GLuint gl_program;

//OpenCL variables:
cl_int err;
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_command_queue command_queue;
cl_event event;
cl_kernel kernel;

cl_mem image;
cl_mem dataBuffer;
cl_mem clTexture;

cl_sampler sampler;
cl_int res;
size_t global_work_items[2], local_work_items[2];
size_t num_workers, origin[3] = { 0, 0, 0 }, region[3] = { 0, 0, 1 };


const char* GetErrorString(cl_int status)
{
	switch (status)
	{
	case CL_SUCCESS:                            return "Success\n";
	case CL_DEVICE_NOT_FOUND:                   return "Device not found\n";
	case CL_DEVICE_NOT_AVAILABLE:               return "Device not available\n";
	case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available\n";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure\n";
	case CL_OUT_OF_RESOURCES:                   return "Out of resources\n";
	case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory\n";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available\n";
	case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap\n";
	case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch\n";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported\n";
	case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure\n";
	case CL_MAP_FAILURE:                        return "Map failure\n";
	case CL_INVALID_VALUE:                      return "Invalid value\n";
	case CL_INVALID_DEVICE_TYPE:                return "Invalid device type\n";
	case CL_INVALID_PLATFORM:                   return "Invalid platform\n";
	case CL_INVALID_DEVICE:                     return "Invalid device\n";
	case CL_INVALID_CONTEXT:                    return "Invalid context\n";
	case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties\n";
	case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue\n";
	case CL_INVALID_HOST_PTR:                   return "Invalid host pointer\n";
	case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object\n";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor\n";
	case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size\n";
	case CL_INVALID_SAMPLER:                    return "Invalid sampler\n";
	case CL_INVALID_BINARY:                     return "Invalid binary\n";
	case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options\n";
	case CL_INVALID_PROGRAM:                    return "Invalid program\n";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable\n";
	case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name\n";
	case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition\n";
	case CL_INVALID_KERNEL:                     return "Invalid kernel\n";
	case CL_INVALID_ARG_INDEX:                  return "Invalid argument index\n";
	case CL_INVALID_ARG_VALUE:                  return "Invalid argument value\n";
	case CL_INVALID_ARG_SIZE:                   return "Invalid argument size\n";
	case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments\n";
	case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension\n";
	case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size\n";
	case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size\n";
	case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset\n";
	case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list\n";
	case CL_INVALID_EVENT:                      return "Invalid event\n";
	case CL_INVALID_OPERATION:                  return "Invalid operation\n";
	case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object\n";
	case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size\n";
	case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level\n";
	default:                                    return "Unknown error\n";
	}
}

//Simulation variables:
float tempScaling = 1000;
float extTemp = 30;
float sourceTemp = 900;
volatile int sourceX = 80, sourceY = 100;

void glfwErrorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		fprintf(stdout, "Closing GLFW Window...\n");
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

void OCLContextCallback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
	size_t i;
	char* data = (char*)private_info;

	printf("Context callback: <%s>\n", errinfo);
	printf("Data: <");

	for (i = 0; i < cb; ++i)
	{
		printf("%c", data[i]);
	}

	printf(">\n");
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		double x, y;
		int ww, wh;
		glfwGetFramebufferSize(window, &ww, &wh);
		glfwGetCursorPos(window, &x, &y);
		y = wh - y;
		//fprintf(stdout, "%f, %f\n", x, y);
		sourceX = x / scaling;
		sourceY = y / scaling;
		//updateSourcePos((int)(x / scaling),(int)(y / scaling));
	}
}

void setupGLFWWindow(GLFWwindow** window, int WINDOW_WIDTH, int WINDOW_HEIGHT) {
	*window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Thermal Propagation", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Could not create GLFW Window...\n");
	}

	//Create GLFW callbacks
	glfwMakeContextCurrent(*window);

	glewExperimental = GL_TRUE;
	glewInit();

	glfwSetErrorCallback(glfwErrorCallback);
	glfwSetKeyCallback(*window, glfwKeyCallback);
	glfwSetMouseButtonCallback(*window, mouse_button_callback);

	glfwGetFramebufferSize(*window, &WINDOW_WIDTH, &WINDOW_HEIGHT);
	//glfwSwapInterval(1);

	//GL view settings
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glOrtho(0, 0, -1.f, 1.f, 1.f, -1.f);
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_TEXTURE_2D);
	glLoadIdentity();
}

void glAllocateTexture(int textureWidth, int textureHeight) {
	int textureSize = textureHeight * textureWidth;

	textureBuffer = (float*)malloc(sizeof(float) * textureSize);
	if (textureBuffer == NULL) {
		fprintf(stderr, "Could not allocate texture memory...\n");
	}
	fprintf(stdout, "Texture Size: %d\n", textureSize);

	for (int i = 0; i < textureSize; i++) {
		textureBuffer[i] = extTemp / tempScaling;
	}
	textureBuffer[sourceY * textureWidth + sourceX] = sourceTemp / tempScaling;

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, textureBuffer);
	glGenerateMipmap(GL_TEXTURE_2D);
	glFinish();
	free(textureBuffer);
}

std::string readFile(const char* filePath) {
	std::string content;
	std::ifstream fileStream(filePath, std::ios::in);

	if (!fileStream.is_open()) {
		std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
		return "";
	}

	std::string line = "";
	while (!fileStream.eof()) {
		std::getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}

GLuint LoadShader(const char* vertex_path, const char* fragment_path) {
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Read shaders
	std::string vertShaderStr = readFile(vertex_path);
	std::string fragShaderStr = readFile(fragment_path);
	const char* vertShaderSrc = vertShaderStr.c_str();
	const char* fragShaderSrc = fragShaderStr.c_str();

	GLint result = GL_FALSE;
	int logLength;

	// Compile vertex shader
	std::cout << "Compiling vertex shader." << std::endl;
	glShaderSource(vertShader, 1, &vertShaderSrc, NULL);
	glCompileShader(vertShader);

	// Check vertex shader
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength);
	std::vector<char> vertShaderError((logLength > 1) ? logLength : 1);
	glGetShaderInfoLog(vertShader, logLength, NULL, &vertShaderError[0]);
	std::cout << &vertShaderError[0] << std::endl;

	// Compile fragment shader
	std::cout << "Compiling fragment shader." << std::endl;
	glShaderSource(fragShader, 1, &fragShaderSrc, NULL);
	glCompileShader(fragShader);

	// Check fragment shader
	glGetShaderiv(fragShader, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength);
	std::vector<char> fragShaderError((logLength > 1) ? logLength : 1);
	glGetShaderInfoLog(fragShader, logLength, NULL, &fragShaderError[0]);
	std::cout << &fragShaderError[0] << std::endl;

	std::cout << "Linking program" << std::endl;
	GLuint program = glCreateProgram();
	glAttachShader(program, vertShader);
	glAttachShader(program, fragShader);
	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &result);
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
	std::vector<char> programError((logLength > 1) ? logLength : 1);
	glGetProgramInfoLog(program, logLength, NULL, &programError[0]);
	std::cout << &programError[0] << std::endl;

	glDeleteShader(vertShader);
	glDeleteShader(fragShader);

	return program;
}

void setupDisplayData(int WINDOW_WIDTH, int WINDOW_HEIGHT) {
	//Texture setup:

	glAllocateTexture(WINDOW_WIDTH, WINDOW_HEIGHT);

	//Generate quad coodrinates and buffers
	float vertices[] = {
		-1.0, -1.0, 0.0, 0.0,
		 1.0, -1.0, 1.0, 0.0,
		 1.0,  1.0, 1.0, 1.0,
		-1.0,  1.0, 0.0, 1.0
	};

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	//Position:
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	//Color:
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	//Load shaders and compile graphics pipeline
	gl_program = LoadShader("C:\\Users\\Stanciu\\source\\repos\\ThermalPropagation\\shader.vs", "C:\\Users\\Stanciu\\source\\repos\\ThermalPropagation\\shader.fs");
	glUseProgram(gl_program);
}

void prepareCLKernel(int textureWidth, int textureHeight)
{

	//Allocate image memory buffers
	cl_image_format format = { CL_R, CL_FLOAT };
	cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, textureWidth, textureHeight, 0, 0, 0, 0, 0, 0, NULL };
	image = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &res);
	SAFE_OCL_CALL(res);
	dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 16 * sizeof(float), NULL, &res);
	SAFE_OCL_CALL(res);

	//Shared OpenGL- OpenCL texture
	clTexture = clCreateFromGLTexture(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, texture, &res);
	SAFE_OCL_CALL(res);

	region[0] = global_work_items[0] = textureWidth;
	region[1] = global_work_items[1] = textureHeight;

	//define and calibrate workgroup sizes
	local_work_items[0] = local_work_items[1] = 16;
	for (int i = 0; i < 2; ++i)
	{
		global_work_items[i] = ((global_work_items[i] + local_work_items[i] - 1) / local_work_items[i]) * local_work_items[i];
	}
	sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &res);
	SAFE_OCL_CALL(res);

	command_queue = clCreateCommandQueue(context, device_id, 0, &res);
	SAFE_OCL_CALL(res);

#pragma region loadKernel
	FILE* fileptr;
	size_t src_size;
	long int file_size;
	char* file_source;
	cl_int status;

	fopen_s(&fileptr, "kernel.cl", "r");
	if (fileptr == NULL) {
		fprintf(stderr, "Error opening OpenCL source file...\n");
		return;
	}

	fseek(fileptr, 0L, SEEK_END);
	file_size = ftell(fileptr) + 1;
	rewind(fileptr);

	file_source = (char*)calloc(file_size, 1);
	src_size = fread(file_source, 1, file_size, fileptr);
	fclose(fileptr);

	program = clCreateProgramWithSource(context, 1, (const char**)& file_source, (const size_t*)& src_size, &status);
	free(file_source);

	if (status != CL_SUCCESS) {
		fprintf(stderr, "Error creating OpenCL program...\n");
		return;
	}

	status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		size_t log_len;
		char* log_buffer;

		fprintf(stderr, "Error building OpenCL program...\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
		++log_len;
		log_buffer = (char*)calloc(log_len, 1);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_len, log_buffer, NULL);
		fprintf(stderr, "Build log:\n<%s>\n", log_buffer);
		free(log_buffer);
		return;
	}

	kernel = clCreateKernel(program, "ThermalPropagation", &status);
	SAFE_OCL_CALL(res);
#pragma endregion loadKernel

	res = clSetKernelArg(kernel, 0, sizeof(clTexture), &clTexture);
	SAFE_OCL_CALL(res);
	/*res = clSetKernelArg(kernel, 0, sizeof(image), &image);
	SAFE_OCL_CALL(res);*/
	res = clSetKernelArg(kernel, 1, sizeof(image), &image);
	SAFE_OCL_CALL(res);
	res = clSetKernelArg(kernel, 3, sizeof(sampler), &sampler);

}

void setupOpenCLContext(int textureWidth, int textureHeight, cl_device_type dev_type) {
	cl_platform_id* platforms = NULL;
	cl_device_id* devices = NULL;
	cl_uint num_platforms, num_devices, i;
	cl_int status = CL_SUCCESS;

	static cl_context_properties context_properties[] = { CL_CONTEXT_PLATFORM, 0,
														  CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
														  CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
														  0 };

	clGetPlatformIDs(0, NULL, &num_platforms);
	if (num_platforms == 0) {
		printf("No OpenCL platforms found...\n");
		return;
	}

	platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	clGetPlatformIDs(num_platforms, platforms, NULL);

	for (int i = 0; i < num_platforms; i++) {
		clGetDeviceIDs(platforms[i], dev_type, 0, NULL, &num_devices);
		if (num_devices == 0)
			continue;
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		clGetDeviceIDs(platforms[i], dev_type, num_devices, devices, NULL);
		context_properties[1] = (cl_context_properties)platforms[i];
		context = clCreateContext(context_properties, 1, devices, &OCLContextCallback, NULL, &status);
		if (status == CL_SUCCESS) {
			platform_id = platforms[i];
			device_id = devices[0];
			break;
		}
		else {
			printf(GetErrorString(status));
		}
	}
	free(devices);
	free(platforms);

	const cl_device_info CL_DEVICE_INFOS[] = {
		CL_DEVICE_NAME,		CL_DEVICE_VENDOR,
		CL_DEVICE_VERSION };
	const char* CL_DEVICE_INFOS_NAMES[] = {
		"Name", "Vendor", "Version", "Driver version", "Profile", "Extensions" };
	char* info = NULL;
	size_t info_size = 0;
	cl_device_type type_data = 0;
	cl_ulong ulong_data = 0;
	cl_uint uint_data = 0;
	cl_bool bool_data = 0;
	size_t size_data = 0;

	printf("OpenCL device <%p>: \n", device_id);

	// Get the device type (CPU, GPU, accelerator or default)
	clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_data, NULL);
	switch (type_data)
	{
	case CL_DEVICE_TYPE_CPU: 			PRINT_DEVICE_TYPE("CPU");
	case CL_DEVICE_TYPE_GPU: 			PRINT_DEVICE_TYPE("GPU");
	case CL_DEVICE_TYPE_ACCELERATOR: 	PRINT_DEVICE_TYPE("Accelerator");
	case CL_DEVICE_TYPE_DEFAULT: 		PRINT_DEVICE_TYPE("Default");
	}

	// Get the device info strings and display them
	for (i = 0; i < sizeof(CL_DEVICE_INFOS) / sizeof(CL_DEVICE_INFOS[0]); ++i)
	{
		clGetDeviceInfo(device_id, CL_DEVICE_INFOS[i], 0, NULL, &info_size);
		info = (char*)malloc(info_size + 1);

		clGetDeviceInfo(device_id, CL_DEVICE_INFOS[i], info_size, info, NULL);
		printf("- %s: %s" "\n", CL_DEVICE_INFOS_NAMES[i], info);
		free(info);
	}

	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &uint_data, NULL);
	printf("- Available compute units: %d"  "\n", uint_data);

	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size_data, NULL);
	printf("- Maximum device work group size: %lu"  "\n", size_data);

}

void processTexture(int textureWidth, int textureHeight) {
	//Convert texture buffer to CL image
	glFinish();

	res = clEnqueueAcquireGLObjects(command_queue, 1, &clTexture, 0, NULL, NULL);
	SAFE_OCL_CALL(res);

	//Setup data:
	float data[] = {textureWidth, textureHeight, sourceTemp, extTemp, sourceX, sourceY, tempScaling};
	res = clEnqueueWriteBuffer(command_queue, dataBuffer, CL_FALSE, 0, sizeof(data), data, 0, NULL, NULL);
	SAFE_OCL_CALL(res);
	res = clSetKernelArg(kernel, 2, sizeof(dataBuffer), &dataBuffer);
	SAFE_OCL_CALL(res);

	
	//Run kernel:
	res = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_items, local_work_items, 0, NULL, NULL);
	SAFE_OCL_CALL(res);

	res = clEnqueueCopyImage(command_queue, image, clTexture, origin, origin, region, 0, NULL, NULL);
	SAFE_OCL_CALL(res);

	res = clEnqueueReleaseGLObjects(command_queue, 1, &clTexture, 0, NULL, NULL);
	SAFE_OCL_CALL(res);

	res = clFinish(command_queue);
	SAFE_OCL_CALL(res);

}

int main(void)
{
	srand(time(NULL));
	float frameStart;
	//Init GL context
	if (!glfwInit()) {
		return -1;
	}

	int WINDOW_HEIGHT = 180, WINDOW_WIDTH = 200;
	scaling = 5;

	//Setup GLFW Window
	GLFWwindow* glwindow = NULL;
	setupGLFWWindow(&glwindow, WINDOW_WIDTH * scaling, WINDOW_HEIGHT * scaling);

	setupOpenCLContext(WINDOW_WIDTH, WINDOW_HEIGHT, CL_DEVICE_TYPE_GPU);

	setupDisplayData(WINDOW_WIDTH, WINDOW_HEIGHT);

	prepareCLKernel(WINDOW_WIDTH, WINDOW_HEIGHT);
	
	

	glBindVertexArray(VAO);

	while (!glfwWindowShouldClose(glwindow)) {
		frameStart = clock();
		glClear(GL_COLOR_BUFFER_BIT);

		processTexture(WINDOW_WIDTH, WINDOW_HEIGHT);

		glBindTexture(GL_TEXTURE_2D, texture);

		glDrawArrays(GL_QUADS, 0, 4);

		frameStart = clock() - frameStart;
		frameStart /= 1000;

		glfwSwapBuffers(glwindow);
		glfwPollEvents();

		//fprintf(stdout, "Frame time: %f\tFPS: %f\n", frameStart, 1 / frameStart);
	}

	//Clean up GLFW

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteTextures(1, &texture);

	clReleaseCommandQueue(command_queue);
	clReleaseMemObject(image);
	clReleaseMemObject(clTexture);
	clReleaseSampler(sampler);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);

	glfwDestroyWindow(glwindow);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
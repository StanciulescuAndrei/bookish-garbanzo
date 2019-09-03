#include "simulation.h"
#define PRINT_DEVICE_TYPE(type_str) { printf("- Device type: %s" "\n", (type_str)); break; }

void ContextCallback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
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

bool inBounds(int x, int y) {
	return (x >= 0 && y >= 0 && x < setup.PLATE_WIDTH && y < setup.PLATE_HEIGHT);
}

void generateSetup(int WINDOW_WIDTH, int WINDOW_HEIGHT, float extTemp, float srcTemp, int sourceX, int sourceY) {
	if (WINDOW_WIDTH <= 0 || WINDOW_HEIGHT <= 0 || extTemp < 0 || srcTemp < 0) {
		fprintf(stderr, "Invalid setup parameters...\n");
	}
	if (sourceX < 0 || sourceX >= WINDOW_WIDTH || sourceY < 0 || sourceY >= WINDOW_HEIGHT) {
		fprintf(stderr, "Invalid source position...\n");
	}
	setup.PLATE_HEIGHT = WINDOW_HEIGHT;
	setup.PLATE_WIDTH = WINDOW_WIDTH;
	setup.sourceTemperature = srcTemp;
	setup.externalTemperature = extTemp;
	setup.sourceX = sourceX;
	setup.sourceY = sourceY;
}

bool margin(int x, int y) {
	return (x == 0 || y == 0 || x == setup.PLATE_WIDTH - 1 || y == setup.PLATE_HEIGHT - 1);
}

void updatePixel(float* srcBuffer, float* destBuffer, int start, int stride) {
	float sum;
	float cnt;
	for (int i = start; i < start + stride; i++) {
		int x = i % setup.PLATE_WIDTH;
		int y = i / setup.PLATE_WIDTH;
		sum = 0;
		cnt = 0;
		for (int dx = -1; dx < 2; dx++) {
			for (int dy = -1; dy < 2; dy++) {
				if (inBounds(x + dx, y + dy)) {
					if (margin(x + dx, y + dy)) {
						sum = sum + srcBuffer[(x + dx) + (y + dy) * setup.PLATE_WIDTH] * 0.00005; //coeficient de transfer al caldurii
						cnt += 0.0001;
					}
					else {
						sum = sum + srcBuffer[(x + dx) + (y + dy) * setup.PLATE_WIDTH] * 1.0; //coeficient de transfer al caldurii
						cnt+=1;
					}
					
				}
			}
		}
		sum = sum / cnt;
		destBuffer[x + y * setup.PLATE_WIDTH] = sum;
	}
}

void updateSourcePos(int x, int y) {
	setup.sourceX = x;
	setup.sourceY = y;
}

void updateBuffer(float * buffer) {
	float* tempBuffer = (float *) malloc(sizeof(float) * setup.PLATE_HEIGHT * setup.PLATE_WIDTH);

	std::vector<std::thread*> proc;
	int threadCount = 2;
	int totalSize = setup.PLATE_HEIGHT * setup.PLATE_WIDTH;
	int stride = totalSize / threadCount;
	

	for (int i = 0; i < threadCount; i++) {
		if (i == threadCount - 1) {
			proc.push_back(new std::thread(updatePixel, buffer, tempBuffer, i * stride, totalSize - stride * i));
		}
		else {
			proc.push_back(new std::thread(updatePixel, buffer, tempBuffer, i * stride, stride));
		}
	}

	for (int i = 0; i < threadCount; i++) {
		proc[i]->join();
	}

	tempBuffer[setup.sourceY * setup.PLATE_WIDTH + setup.sourceX] = setup.sourceTemperature / tempScale;

	memcpy(buffer, tempBuffer, sizeof(float) * setup.PLATE_HEIGHT * setup.PLATE_WIDTH);
	free(tempBuffer);

	buffer[setup.sourceY * setup.PLATE_WIDTH + setup.sourceX] = setup.sourceTemperature / tempScale;
	for (int i = 0; i < setup.PLATE_HEIGHT; i++) {
		buffer[i * setup.PLATE_WIDTH] = setup.externalTemperature / tempScale;
		buffer[i * setup.PLATE_WIDTH + setup.PLATE_WIDTH - 1] = setup.externalTemperature / tempScale;
	}
	for (int i = 0; i < setup.PLATE_WIDTH; i++) {
		buffer[i] = setup.externalTemperature / tempScale;
		buffer[setup.PLATE_HEIGHT * setup.PLATE_WIDTH - 1 - i] = setup.externalTemperature / tempScale;
	}
}

void fillBuffer(float* buffer, float value)
{
	for (int i = 0; i < setup.PLATE_HEIGHT * setup.PLATE_WIDTH; i++) {
		buffer[i] = setup.externalTemperature / tempScale;
		//buffer[i] = value;
	}
	buffer[setup.sourceY * setup.PLATE_WIDTH + setup.sourceX] = setup.sourceTemperature / tempScale;
}

void setupOpenCLContext(cl_device_type dev_type, GLFWwindow * window)
{
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
		context = clCreateContext(context_properties, 1, devices, &ContextCallback, NULL, &status);
		if (status == CL_SUCCESS) {
			platform_id = platforms[i];
			device_id = devices[0];
			break;
		}
	}
	free(devices);
	free(platforms);

	const cl_device_info CL_DEVICE_INFOS[] = {
		CL_DEVICE_NAME,		CL_DEVICE_VENDOR,
		CL_DEVICE_VERSION,	CL_DRIVER_VERSION,
		CL_DEVICE_PROFILE,	CL_DEVICE_EXTENSIONS };
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

	// Get additional device information (global memory size, compute unites, compiler availability)
	clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &ulong_data, NULL);
	printf("- Global memory size: %lu MB"  "\n", ulong_data >> 20);

	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &uint_data, NULL);
	printf("- Available compute units: %d"  "\n", uint_data);

	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size_data, NULL);
	printf("- Maximum device work group size: %lu"  "\n", size_data);

	clGetDeviceInfo(device_id, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &bool_data, NULL);
	printf("- Compiler available: %s" "\n", bool_data ? "Yes" : "No");

}

void prepareCLKernel(cl_GLuint sourceTexture, cl_GLuint destTexture)
{
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
		log_buffer = (char *) calloc(log_len, 1);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_len, log_buffer, NULL);
		fprintf(stderr, "Build log:\n<%s>\n", log_buffer);
		free(log_buffer);
		return;
	}

	kernel = clCreateKernel(program, "ThermalPropagation", &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Could not get specified kernel...\n");
	}

	sourceMemory = clCreateFromGLTexture(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, sourceTexture, NULL);
	destMemory = clCreateFromGLTexture(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, destTexture, NULL);
	sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, NULL);

	clSetKernelArg(kernel, 0, sizeof(sourceMemory), &sourceMemory);
	clSetKernelArg(kernel, 1, sizeof(destMemory), &destMemory);

	clSetKernelArg(kernel, 3, sizeof(sampler), &sampler);

	clFinish(command_queue);

}

void updateBufferGPU(cl_int xRes, cl_int yRes) {

	//Create memory from texture:
	
	cl_int res;
	size_t local_worksize[] = { 256, 256 };
	//balance the global worksize for OpenCL 1.2 (Intel HD Graphics 4600)
	size_t global_worksize[] = {((setup.PLATE_WIDTH / local_worksize[0]) + 1)*local_worksize[0], ((setup.PLATE_HEIGHT / local_worksize[1]) + 1) * local_worksize[1] };
	
	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { setup.PLATE_WIDTH, setup.PLATE_HEIGHT, 1 };
	cl_float data[] = { setup.PLATE_HEIGHT, setup.PLATE_WIDTH, setup.externalTemperature, setup.sourceTemperature, setup.sourceX, setup.sourceY };
	cl_mem dataBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data), data, NULL);

	//Get texture memory space
	glFinish();
	clEnqueueAcquireGLObjects(command_queue, 1, &sourceMemory, 0, 0, NULL);
	clEnqueueAcquireGLObjects(command_queue, 1, &destMemory, 0, 0, NULL);

	//set cl kernel parameters
	clSetKernelArg(kernel, 2, sizeof(dataBuffer), &dataBuffer);

	//run kernel
	res = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_worksize, local_worksize, 0, NULL, NULL);

	//copy data to the main texture memory location
	//clEnqueueCopyImage(command_queue, destMemory, sourceMemory, origin, origin, region, 0, NULL, NULL);

	//release textures
	clEnqueueReleaseGLObjects(command_queue, 1, &sourceMemory, 0, 0, NULL);
	clEnqueueReleaseGLObjects(command_queue, 1, &destMemory, 0, 0, NULL);

	clReleaseMemObject(dataBuffer);

	clFinish(command_queue);
}

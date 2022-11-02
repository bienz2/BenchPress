// Devices
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice

// Data allocation
#define gpuMallocHost hipMallocHost
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuFreeHost hipFreeHost

// Error Handling
#define gpuError hipError
#define gpuGetLastError hipGetLastError

// Memcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Streams
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy 

// Synchronization
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamSynchronize hipStreamSynchronize


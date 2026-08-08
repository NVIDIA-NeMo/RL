#include <cuda.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult result = (call);                                                   \
    if (result != CUDA_SUCCESS) {                                               \
      const char *message = NULL;                                               \
      cuGetErrorString(result, &message);                                       \
      fprintf(stderr, "%s failed: %d (%s)\n", #call, result, message);          \
      return 2;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  const size_t length = 16 * 1024 * 1024;
  CUdevice device;
  CUcontext cuda_context;
  CUdeviceptr pointer;
  CUdeviceptr allocation_base;
  size_t allocation_size;
  int dmabuf_fd = -1;

  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, device));
  CUDA_CHECK(cuCtxSetCurrent(cuda_context));
  CUDA_CHECK(cuMemAlloc(&pointer, length));
  CUDA_CHECK(cuMemGetAddressRange(&allocation_base, &allocation_size, pointer));
  CUDA_CHECK(cuMemGetHandleForAddressRange(
      &dmabuf_fd,
      allocation_base,
      allocation_size,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      0));

  int device_count = 0;
  struct ibv_device **devices = ibv_get_device_list(&device_count);
  if (devices == NULL || device_count == 0) {
    fprintf(stderr, "No RDMA devices found\n");
    return 3;
  }
  struct ibv_context *ib_context = ibv_open_device(devices[0]);
  struct ibv_pd *pd = ib_context == NULL ? NULL : ibv_alloc_pd(ib_context);
  if (pd == NULL) {
    fprintf(stderr, "Failed to create RDMA protection domain: errno=%d\n", errno);
    return 4;
  }

  unsigned int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC |
                        IBV_ACCESS_RELAXED_ORDERING;
  errno = 0;
  struct ibv_mr *mr = ibv_reg_dmabuf_mr(
      pd, 0, allocation_size, allocation_base, dmabuf_fd, access);
  int saved_errno = errno;
  printf(
      "dmabuf-probe device=%s pointer=0x%llx base=0x%llx length=%zu "
      "allocation_size=%zu fd=%d mr=%p errno=%d\n",
      ibv_get_device_name(devices[0]),
      (unsigned long long)pointer,
      (unsigned long long)allocation_base,
      length,
      allocation_size,
      dmabuf_fd,
      (void *)mr,
      saved_errno);
  fflush(stdout);

  if (mr != NULL) {
    ibv_dereg_mr(mr);
  }
  ibv_dealloc_pd(pd);
  ibv_close_device(ib_context);
  ibv_free_device_list(devices);
  close(dmabuf_fd);
  cuMemFree(pointer);
  cuDevicePrimaryCtxRelease(device);
  return mr == NULL ? 5 : 0;
}

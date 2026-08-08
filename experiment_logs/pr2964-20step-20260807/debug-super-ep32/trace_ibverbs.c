#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct ibv_mr *(*ibv_reg_mr_iova2_fn)(
    struct ibv_pd *, void *, size_t, uint64_t, unsigned int);

struct ibv_mr *ibv_reg_mr_iova2(
    struct ibv_pd *pd,
    void *addr,
    size_t length,
    uint64_t iova,
  unsigned int access) {
  static ibv_reg_mr_iova2_fn real_fn;
  if (real_fn == NULL) {
    void *handle = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
      fprintf(stderr, "ibv-trace dlopen failed: %s\n", dlerror());
      abort();
    }
    real_fn = (ibv_reg_mr_iova2_fn)dlvsym(
        handle, "ibv_reg_mr_iova2", "IBVERBS_1.8");
    if (real_fn == NULL) {
      fprintf(stderr, "ibv-trace dlvsym failed: %s\n", dlerror());
      abort();
    }
  }

  errno = 0;
  struct ibv_mr *mr = real_fn(pd, addr, length, iova, access);
  int saved_errno = errno;
  fprintf(
      stderr,
      "ibv-trace rank=%s local_rank=%s addr=%p length=%zu access=0x%x "
      "result=%p errno=%d\n",
      getenv("SLURM_PROCID") == NULL ? "?" : getenv("SLURM_PROCID"),
      getenv("SLURM_LOCALID") == NULL ? "?" : getenv("SLURM_LOCALID"),
      addr,
      length,
      access,
      (void *)mr,
      saved_errno);
  fflush(stderr);
  errno = saved_errno;
  return mr;
}

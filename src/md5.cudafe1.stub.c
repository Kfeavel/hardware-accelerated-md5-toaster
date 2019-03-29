#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "md5.fatbin.c"
extern void __device_stub__Z18md5_cuda_calculatePvP12device_statsPj(void *, struct device_stats *, unsigned *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z18md5_cuda_calculatePvP12device_statsPj(void *__par0, struct device_stats *__par1, unsigned *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(void *, struct device_stats *, unsigned *))md5_cuda_calculate)));}
# 166 "md5.cu"
void md5_cuda_calculate( void *__cuda_0,struct device_stats *__cuda_1,unsigned *__cuda_2)
# 167 "md5.cu"
{__device_stub__Z18md5_cuda_calculatePvP12device_statsPj( __cuda_0,__cuda_1,__cuda_2);
# 190 "md5.cu"
}
# 1 "md5.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T3) {  __nv_dummy_param_ref(__T3); __nv_save_fatbinhandle_for_managed_rt(__T3); __cudaRegisterEntry(__T3, ((void ( *)(void *, struct device_stats *, unsigned *))md5_cuda_calculate), _Z18md5_cuda_calculatePvP12device_statsPj, (-1)); __cudaRegisterVariable(__T3, __shadow_var(target_hash,::target_hash), 0, 16UL, 1, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop

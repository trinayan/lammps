#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#include "reaxc_types.h"

#ifdef __cplusplus
extern "C"  {
#endif

void  CudaAllocateStorageForFixQeq(int nmax, int dual_enabled, fix_qeq_gpu *qeq_gpu);
void  CudaAllocateMatrixForFixQeq(fix_qeq_gpu *qeq_gpu,int n_cap, int m_cap);
void  CudaInitStorageForFixQeq(fix_qeq_gpu *qeq_gpu, double *Hdia_inv, double *b_s,double *b_t,double *b_prc,double *b_prm,double *s,double *t, int N);


#ifdef __cplusplus
}
#endif

#endif

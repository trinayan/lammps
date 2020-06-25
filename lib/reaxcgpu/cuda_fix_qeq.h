#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#include "reaxc_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void  CudaAllocateStorageForFixQeq(int nmax, int dual_enabled, fix_qeq_gpu *qeq_gpu);
void  CudaAllocateMatrixForFixQeq(fix_qeq_gpu *qeq_gpu,int n_cap, int m_cap);
void  CudaInitStorageForFixQeq(fix_qeq_gpu *qeq_gpu, double *Hdia_inv, double *b_s,double *b_t,double *b_prc,double *b_prm,double *s,double *t, int N);
void  Cuda_Calculate_H_Matrix(reax_list **gpu_lists,  reax_system *system,fix_qeq_gpu *qeq_gpu, control_params *control, int inum);
void  Cuda_Init_Taper(fix_qeq_gpu *qeq_gpu,double *Tap, int numTap);
void  Cuda_Init_Shielding(fix_qeq_gpu *qeq_gpu,double *shld,int ntypes);
void  Cuda_Allocate_Matrix( sparse_matrix *, int, int );
void Cuda_Deallocate_Matrix( sparse_matrix *H );
void Cuda_Init_Sparse_Matrix_Indices( reax_system *system, sparse_matrix *H);
void  Cuda_Init_Fix_Atoms(reax_system *system,fix_qeq_gpu *qeq_gpu);
#ifdef __cplusplus
}
#endif

#endif

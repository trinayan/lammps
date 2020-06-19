#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_list.h"
#include "cuda_neighbors.h"
#include "cuda_utils.h"
#include "cuda_fix_qeq.h"


#include "allocate.h"
#include "index_utils.h"
#include "tool_box.h"
#include "vector.h"

extern "C"
{
void  CudaAllocateStorageForFixQeq(int nmax, int dual_enabled, fix_qeq_gpu *qeq_gpu)
{
	cuda_malloc( (void **) &qeq_gpu->s, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->t, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->Hdia_inv, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_s, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_t, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_prc, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_prm, sizeof(int) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );

	int size = nmax;
	if (dual_enabled)
	{
		size*= 2;
	}

	cuda_malloc( (void **) &qeq_gpu->p, sizeof(int) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->q, sizeof(int) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->r, sizeof(int) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->d, sizeof(int) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );

}
void  CudaAllocateMatrixForFixQeq(fix_qeq_gpu *qeq_gpu, int n, int m)
{
	qeq_gpu->H.m = m;
	qeq_gpu->H.n = n;

	cuda_malloc( (void **) &qeq_gpu->H.start, sizeof(int) * n, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->H.end, sizeof(int) * n, TRUE,
			"Cuda_Allocate_Matrix::end" );
	cuda_malloc( (void **) &qeq_gpu->H.entries, sizeof(sparse_matrix_entry) * m, TRUE,
			"Cuda_Allocate_Matrix::entries" );
}
void  CudaInitStorageForFixQeq(fix_qeq_gpu *qeq_gpu, double *Hdia_inv, double *b_s,double *b_t,double *b_prc,double *b_prm,double *s,double *t, int NN)
{
	copy_host_device( Hdia_inv, qeq_gpu->Hdia_inv, sizeof(double) * NN,
	            hipMemcpyHostToDevice, "Cuda_CG::q:get" );
	copy_host_device( b_s, qeq_gpu->b_s, sizeof(double) * NN,
		            hipMemcpyHostToDevice, "Cuda_CG::q:get" );
	copy_host_device( b_t, qeq_gpu->b_t, sizeof(double) * NN,
		            hipMemcpyHostToDevice, "Cuda_CG::q:get" );
	copy_host_device( b_prc, qeq_gpu->b_prc, sizeof(double) * NN,
		            hipMemcpyHostToDevice, "Cuda_CG::q:get" );
	copy_host_device( b_prm, qeq_gpu->b_prm, sizeof(double) * NN,
		            hipMemcpyHostToDevice, "Cuda_CG::q:get" );
	copy_host_device( s, qeq_gpu->s, sizeof(double) * NN,
			            hipMemcpyHostToDevice, "Cuda_CG::q:get" );
	copy_host_device(t, qeq_gpu->t, sizeof(double) * NN,
			            hipMemcpyHostToDevice, "Cuda_CG::q:get");


}

}

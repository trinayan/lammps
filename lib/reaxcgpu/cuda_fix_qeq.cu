#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_list.h"
#include "cuda_neighbors.h"
#include "cuda_utils.h"
#include "cuda_fix_qeq.h"
#include "cuda_reduction.h"

#include "allocate.h"
#include "index_utils.h"
#include "tool_box.h"
#include "vector.h"

extern "C"
{




/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
CUDA_GLOBAL void k_init_cm_full_fs( reax_atom *my_atoms, single_body_parameters *sbp,
		two_body_parameters *tbp,
		reax_list far_nbr_list, int num_atom_types,
		int *max_cm_entries, int *realloc_cm_entries )
{


}
/* Compute the distances and displacement vectors for entries
 * in the far neighbors list if it's a NOT re-neighboring step */
CUDA_GLOBAL void k_init_distance( reax_atom *my_atoms, reax_list far_nbrs_list, int N )
{
	int i, j, pj;
	int start_i, end_i;
	reax_atom *atom_i, *atom_j;
	far_neighbor_data *nbr_pj;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= N )
	{
		return;
	}

	atom_i = &my_atoms[i];
	start_i = Cuda_Start_Index( i, &far_nbrs_list );
	end_i =   Cuda_End_Index( i, &far_nbrs_list );

	/* update distance and displacement vector between atoms i and j (i-j) */
	for ( pj = start_i; pj < end_i; ++pj )
	{
		nbr_pj = &far_nbrs_list.select.far_nbr_list[pj];
		j = nbr_pj->nbr;
		atom_j = &my_atoms[j];

		if ( i < j )
		{
			nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
			nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
			nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
		}
		else
		{
			nbr_pj->dvec[0] = atom_i->x[0] - atom_j->x[0];
			nbr_pj->dvec[1] = atom_i->x[1] - atom_j->x[1];
			nbr_pj->dvec[2] = atom_i->x[2] - atom_j->x[2];
		}
		nbr_pj->d = rvec_Norm(nbr_pj->dvec);
	}
}

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

void  Cuda_Calculate_H_Matrix(reax_list **lists,  reax_system *system, fix_qeq_gpu *qeq_gpu,control_params *control)
{

	printf("Blocks n %d, block size %d\n", control->blocks_n, control->block_size);

	hipLaunchKernelGGL(k_init_distance, dim3(control->blocks_n), dim3(control->block_size), 0, 0,  system->d_my_atoms, *(lists[FAR_NBRS]), system->N );
	hipDeviceSynchronize();





	int blocks;
	blocks = (system->N) / DEF_BLOCK_SIZE +
			(((system->N % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

	hipLaunchKernelGGL(k_init_cm_full_fs , dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  system->d_my_atoms, system->reax_param.d_sbp,
			system->reax_param.d_tbp,
			*(lists[FAR_NBRS]),system->reax_param.num_atom_types,system->d_max_cm_entries,system->d_realloc_cm_entries);
	hipDeviceSynchronize( );

	/*int  *d_ilist, *d_jlist, *d_numneigh, *d_firstneigh;
	int *d_type;
	LAMMPS_NS::tagint *d_tag;
	double **d_x;
	int *d_mask;

	int param2 = 1000;
	int numDimensions = 3;

	cuda_malloc( (void **) &d_ilist, sizeof(int) * inum, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &d_numneigh, sizeof(int) * inum, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &d_firstneigh, sizeof(int) * inum * param2, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &d_type, sizeof(int) * param2, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &d_tag, sizeof(LAMMPS_NS::tagint) * param2, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &d_mask, sizeof(int) * inum, TRUE,
			"Cuda_Allocate_Matrix::start");
	cuda_malloc( (void **) &d_x, sizeof(double)*param2*numDimensions, TRUE,
			"Cuda_Allocate_Matrix::start");


	copy_host_device(ilist, d_ilist, sizeof(int) * inum,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	copy_host_device(numneigh, d_numneigh, sizeof(int) * inum,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	copy_host_device(firstneigh, d_firstneigh, sizeof(int) *param2*inum,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	copy_host_device(numneigh, d_numneigh, sizeof(int) * inum,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	copy_host_device(mask, d_mask, sizeof(int) * inum,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	copy_host_device(x, d_x, sizeof(double)*param2*numDimensions,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");*/



	/*copy_host_device(t, qeq_gpu->t, sizeof(double) * NN,
	  			            hipMemcpyHostToDevice, "Cuda_CG::q:get");
	  copy_host_device(t, qeq_gpu->t, sizeof(double) * NN,
	  			            hipMemcpyHostToDevice, "Cuda_CG::q:get");
	  copy_host_device(t, qeq_gpu->t, sizeof(double) * NN,
	  			            hipMemcpyHostToDevice, "Cuda_CG::q:get");*/

}

void Cuda_Init_Taper(fix_qeq_gpu *qeq_gpu,double *Tap, int numTap)
{
	cuda_malloc( (void **) &qeq_gpu->d_Tap, sizeof(double)*numTap, TRUE,
			"Cuda_Allocate_Matrix::start");
	copy_host_device(Tap, qeq_gpu->d_Tap, sizeof(double) * numTap,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
}

void  Cuda_Init_Shielding(fix_qeq_gpu *qeq_gpu,double **shld,int ntypes)
{
	cuda_malloc( (void **) &qeq_gpu->shld, sizeof(double)*ntypes*ntypes, TRUE,
			"Cuda_Allocate_Matrix::start");
	copy_host_device(shld, qeq_gpu->shld, sizeof(double) * ntypes * ntypes,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");

}

void Cuda_Allocate_Matrix( sparse_matrix *H, int n, int m )
{
	H->m = m;
	H->n = n;

	cuda_malloc( (void **) &H->start, sizeof(int) * n, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &H->end, sizeof(int) * n, TRUE,
			"Cuda_Allocate_Matrix::end" );
	cuda_malloc( (void **) &H->entries, sizeof(sparse_matrix_entry) * m, TRUE,
			"Cuda_Allocate_Matrix::entries" );
}


void Cuda_Deallocate_Matrix( sparse_matrix *H )
{
	cuda_free( H->start, "Cuda_Deallocate_Matrix::start" );
	cuda_free( H->end, "Cuda_Deallocate_Matrix::end" );
	cuda_free( H->entries, "Cuda_Deallocate_Matrix::entries" );
}

void Cuda_Init_Sparse_Matrix_Indices( reax_system *system, sparse_matrix *H )
{
	int blocks;

	/* init indices */
	Cuda_Scan_Excl_Sum( system->d_max_cm_entries, H->start, system->total_cap );

	/* init end_indices */
	blocks = system->total_cap / DEF_BLOCK_SIZE
			+ ((system->total_cap % DEF_BLOCK_SIZE == 0) ? 0 : 1);
	hipLaunchKernelGGL(k_init_end_index, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  system->d_cm_entries, H->start, H->end, system->total_cap );
	hipDeviceSynchronize( );
	cudaCheckError( );
}








}

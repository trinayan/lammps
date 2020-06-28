#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_list.h"
#include "cuda_neighbors.h"
#include "cuda_utils.h"
#include "cuda_fix_qeq.h"
#include "cuda_reduction.h"
#include "basic_comm.h"

#include "allocate.h"
#include "index_utils.h"
#include "tool_box.h"
#include "vector.h"

extern "C"
{




CUDA_DEVICE real Init_Charge_Matrix_Entry( single_body_parameters *sbp_i, real *workspace_Tap,
		int i, int j, real r_ij, real gamma)
{
	real Tap, dr3gamij_1, dr3gamij_3, ret;

	ret = 0.0;


	Tap = workspace_Tap[7] * r_ij + workspace_Tap[6];
	Tap = Tap * r_ij + workspace_Tap[5];
	Tap = Tap * r_ij + workspace_Tap[4];
	Tap = Tap * r_ij + workspace_Tap[3];
	Tap = Tap * r_ij + workspace_Tap[2];
	Tap = Tap * r_ij + workspace_Tap[1];
	Tap = Tap * r_ij + workspace_Tap[0];

	/* shielding */
	dr3gamij_1 = r_ij * r_ij * r_ij
			+ POW( gamma, -3.0 );
	dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

	/* i == j: periodic self-interaction term
	 * i != j: general interaction term */
	ret = ((i == j) ? 0.5 : 1.0) * Tap * EV_to_KCALpMOL / dr3gamij_3;
	return ret;
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
CUDA_GLOBAL void k_init_cm_full_fs( reax_atom *my_atoms, single_body_parameters *sbp,
		reax_list far_nbrs_list, int num_atom_types,
		int *max_cm_entries, int *realloc_cm_entries, sparse_matrix H, int nonb_cut, int inum, double *d_Tap,  two_body_parameters *tbp, double *gamma)
{
	int i, j, pj;
	int start_i, end_i;
	int type_i, type_j;
	int cm_top;
	int num_cm_entries;
	real r_ij;
	single_body_parameters *sbp_i;
	two_body_parameters *twbp;
	reax_atom *atom_i, *atom_j;
	far_neighbor_data *nbr_pj;
	double shld;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= inum)
	{
		return;
	}

	cm_top = H.start[i];

	atom_i = &my_atoms[i];
	type_i = atom_i->type;
	start_i = Cuda_Start_Index( i, &far_nbrs_list );
	end_i = Cuda_End_Index(i, &far_nbrs_list );
	sbp_i = &sbp[type_i];


	/* diagonal entry in the matrix */

	/*H.entries[cm_top].j = i;
	H->val[cm_top] = Init_Charge_Matrix_Entry( sbp_i, workspace.Tap, control,
	               i, i, 0.0, 0.0, DIAGONAL );
	        ++cm_top;*/

	/* update i-j distance - check if j is within cutoff */
	for ( pj = start_i; pj < end_i; ++pj )
	{
		nbr_pj = &far_nbrs_list.select.far_nbr_list[pj];
		j = nbr_pj->nbr;

		if ( nbr_pj->d  <= nonb_cut)
		{
			atom_j = &my_atoms[j];
			type_j = atom_j->type;

			twbp = &tbp[ index_tbp(type_i, type_j, num_atom_types) ];
			r_ij =  nbr_pj->d;


			H.entries[cm_top].j = j;
			shld = pow( gamma[type_i] * gamma[type_j], -1.5);


			H.entries[cm_top].val = Init_Charge_Matrix_Entry(sbp_i, d_Tap,
					i, H.entries[cm_top].j, r_ij, shld);


			++cm_top;
		}
	}


	H.end[i] = cm_top;
	num_cm_entries = cm_top - H.start[i];

	//printf("Cm top %d \n", cm_)


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
	cuda_malloc( (void **) &qeq_gpu->s, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->t, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->Hdia_inv, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_s, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_t, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_prc, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->b_prm, sizeof(double) * nmax, TRUE,
			"Cuda_Allocate_Matrix::start" );

	int size = nmax;
	if (dual_enabled)
	{
		size*= 2;
	}

	cuda_malloc( (void **) &qeq_gpu->p, sizeof(double) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->q, sizeof(double) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->r, sizeof(double) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );
	cuda_malloc( (void **) &qeq_gpu->d, sizeof(double) * size, TRUE,
			"Cuda_Allocate_Matrix::start" );

	cuda_malloc( (void **) &qeq_gpu->s_hist, sizeof(rvec4) * nmax, TRUE, "b" );
	cuda_malloc( (void **) &qeq_gpu->t_hist, sizeof(rvec4) * nmax, TRUE, "x" );


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

void  Cuda_Init_Fix_Atoms(reax_system *system,fix_qeq_gpu *qeq_gpu)

{
	cuda_malloc( (void **) &qeq_gpu->d_fix_my_atoms, sizeof(reax_atom) * system->N, TRUE,
			"Cuda_Allocate_Matrix::start" );
	copy_host_device(qeq_gpu->fix_my_atoms, qeq_gpu->d_fix_my_atoms, sizeof(reax_atom) * system->N,
			hipMemcpyHostToDevice, "Sync_Atoms::system->my_atoms");
}

void  Cuda_Calculate_H_Matrix(reax_list **lists,  reax_system *system, fix_qeq_gpu *qeq_gpu,control_params *control, int inum)
{

	printf("Blocks n %d, block size %d\n", control->blocks_n, control->block_size);

	printf("Control cut %f\n", control->nonb_cut);

	//TB:: Verify if to use inum or system->N in kernel call
	hipLaunchKernelGGL(k_init_distance, dim3(control->blocks_n), dim3(control->block_size), 0, 0,  qeq_gpu->d_fix_my_atoms, *(lists[FAR_NBRS]), inum );
	hipDeviceSynchronize();

	int blocks;
	blocks = (system->N) / DEF_BLOCK_SIZE +
			(((system->N % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

	printf("Blocks %d , blocks size %d\n", blocks, DEF_BLOCK_SIZE);

	printf("N %d, h n %d \n",system->N, qeq_gpu->H.n);

	//TB:: Verify if to use inum or system->N  or qeq_gpu->H.n in kernel call
	hipLaunchKernelGGL(k_init_cm_full_fs , dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  qeq_gpu->d_fix_my_atoms, system->reax_param.d_sbp,
			*(lists[FAR_NBRS]),system->reax_param.num_atom_types,system->d_max_cm_entries,system->d_realloc_cm_entries,qeq_gpu->H, control->nonb_cut, inum, qeq_gpu->d_Tap, system->reax_param.d_tbp,qeq_gpu->gamma);
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


CUDA_GLOBAL void k_init_matvec_fix(fix_qeq_gpu d_qeq_gpu,int nn, single_body_parameters
		*sbp,reax_atom *my_atoms)
{
	int i;
	int type_i;
	fix_qeq_gpu *qeq_gpu;
	qeq_gpu = &d_qeq_gpu;





	i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= nn)
	{
		return;
	}
	reax_atom *atom;
	atom = &my_atoms[i];
	type_i = atom->type;


	qeq_gpu->Hdia_inv[i] = 1. / qeq_gpu->eta[type_i];
	qeq_gpu->b_s[i] = -qeq_gpu->chi[type_i];
	qeq_gpu->b_t[i] = -1.0;



	qeq_gpu->t[i] = qeq_gpu->t_hist[i][2] + 3 * ( qeq_gpu->t_hist[i][0] - qeq_gpu->t_hist[i][1]);
	/* cubic extrapolation for s & t from previous solutions */
	qeq_gpu->s[i] = 4*(qeq_gpu->s_hist[i][0]+qeq_gpu->s_hist[i][2])-(6*qeq_gpu->s_hist[i][1]+qeq_gpu->s_hist[i][3]);



}

void  Cuda_Init_Matvec_Fix(int nn, fix_qeq_gpu *qeq_gpu, reax_system *system)
{
	int blocks;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	hipLaunchKernelGGL(k_init_matvec_fix, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, *(qeq_gpu),nn,system->reax_param.d_sbp,qeq_gpu->d_fix_my_atoms);
	hipDeviceSynchronize();
	cudaCheckError();
}

void  Cuda_Copy_Pertype_Parameters_To_Device(double *chi,double *eta,double *gamma,int ntypes,fix_qeq_gpu *qeq_gpu)
{
	cuda_malloc( (void **) &qeq_gpu->gamma, sizeof(double)*(ntypes+1), TRUE,
			"Cuda_Allocate_Matrix::start");
	copy_host_device(gamma, qeq_gpu->gamma, sizeof(double) * (ntypes+1),
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	cuda_malloc( (void **) &qeq_gpu->chi, sizeof(double)*(ntypes+1), TRUE,
			"Cuda_Allocate_Matrix::start");
	copy_host_device(gamma, qeq_gpu->chi, sizeof(double) * (ntypes+1),
			hipMemcpyHostToDevice, "Cuda_CG::q:get");
	cuda_malloc( (void **) &qeq_gpu->eta, sizeof(double)*(ntypes+1), TRUE,
			"Cuda_Allocate_Matrix::start");
	copy_host_device(eta, qeq_gpu->eta, sizeof(double) * (ntypes+1),
			hipMemcpyHostToDevice, "Cuda_CG::q:get");

}

void  Cuda_Copy_For_Forward_Comm_Fix(double *h_distance , double *d_distance, int nn)
{
	copy_host_device(h_distance, d_distance, sizeof(real) * nn,
			hipMemcpyDeviceToHost, "Cuda_CG::x:get" );
	printf("Copy \n");
}



CUDA_GLOBAL void k_init_b(reax_atom *my_atoms, double *b, double *x,double *eta, int nn)
{

	int i;
	int type_i;
	reax_atom *atom;


	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= nn)
	{
		return;
	}

	atom = &my_atoms[i];
	type_i = atom->type;


	b[i] = eta[type_i] * x[i];
}



void  CUDA_CG_Fix(sparse_matrix *, double *b, double *x, double *q, double *eta, reax_atom *d_fix_my_atoms, int nn, int NN)
{

	int blocks;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	printf("Blocks %d \n",blocks);


	hipLaunchKernelGGL(k_init_b, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, d_fix_my_atoms,b,x,eta,nn);
	hipDeviceSynchronize();

}





}

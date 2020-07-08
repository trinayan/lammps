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




CUDA_DEVICE real Init_Charge_Matrix_Entry(real *workspace_Tap,
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
CUDA_GLOBAL void k_init_cm_full_fs( reax_atom *my_atoms,
		reax_list far_nbrs_list, sparse_matrix H, int nonb_cut, int inum, double *d_Tap, double *gamma, int small)
{
	int i, j, pj;
	int start_i, end_i;
	int type_i, type_j;
	int cm_top;
	int num_cm_entries;
	real r_ij;
	two_body_parameters *twbp;
	reax_atom *atom_i, *atom_j;
	far_neighbor_data *nbr_pj;
	double shld;
	double dx, dy, dz;

	int flag = 0;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= inum)
	{
		return;
	}

	cm_top = H.start[i];

	//printf("I %d, CM top %d\n", i, cm_top);

	atom_i = &my_atoms[i];
	type_i = atom_i->type;
	start_i = Cuda_Start_Index( i, &far_nbrs_list );
	end_i = Cuda_End_Index(i, &far_nbrs_list );

	/* update i-j distance - check if j is within cutoff */
	for ( pj = start_i; pj < end_i; ++pj )
	{
		nbr_pj = &far_nbrs_list.select.far_nbr_list[pj];
		j = nbr_pj->nbr;
		atom_j = &my_atoms[j];

		dx = atom_j->x[0] - atom_i->x[0];
		dy = atom_j->x[1] - atom_i->x[1];
		dz = atom_j->x[2] - atom_i->x[2];

		nbr_pj->d = rvec_Norm(nbr_pj->dvec);

		flag = 0;

		if ( nbr_pj->d  <= nonb_cut)
		{
			if (j < inum)
			{
				flag = 1;
			}
			else if (atom_i->orig_id  < atom_j->orig_id)
			{
				flag = 1;
			}
			else if (atom_i->orig_id ==  atom_j->orig_id)
			{
				if (dz > small)
				{
					flag = 1;
				}
				else if (dz < small)
				{
					if (dy > small)
					{
						flag = 1;
					}
					else if (dy < small && dx > small)
					{
						flag = 1;
					}
				}
			}
		}




		if (flag == 1)
		{
			type_j = atom_j->type;
			r_ij =  nbr_pj->d;

			H.entries[cm_top].j = j;
			shld = pow( gamma[type_i] * gamma[type_j], -1.5);


			H.entries[cm_top].val = Init_Charge_Matrix_Entry(d_Tap,
					i, H.entries[cm_top].j, r_ij, shld);

			++cm_top;
		}
	}



	H.end[i] = cm_top;
	num_cm_entries = cm_top - H.start[i];

	//printf("Index : %d, H first number %d , m fill : %d, NumNbrs:%d \n",i,  H.start[i],cm_top,num_cm_entries);


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

void  Cuda_Calculate_H_Matrix(reax_list **lists,  reax_system *system, fix_qeq_gpu *qeq_gpu,control_params *control, int inum, int small)
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
	hipLaunchKernelGGL(k_init_cm_full_fs , dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  qeq_gpu->d_fix_my_atoms,
			*(lists[FAR_NBRS]),qeq_gpu->H, control->nonb_cut, inum, qeq_gpu->d_Tap,qeq_gpu->gamma,small);
	hipDeviceSynchronize();


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




CUDA_GLOBAL void k_estimate_cm_entries_storage(reax_atom *my_atoms,
		control_params *control, reax_list far_nbrs,
		int n, int total_cap,
		int *cm_entries)
{
	int i, j, pj;
	int start_i, end_i;
	int type_i, type_j;
	int local;
	int  num_cm_entries;
	real cutoff;
	far_neighbor_data *nbr_pj;
	reax_atom *atom_i, *atom_j;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= total_cap )
	{
		return;
	}


	num_cm_entries = 0;


	atom_i = &my_atoms[i];
	type_i = atom_i->type;
	start_i = Cuda_Start_Index( i, &far_nbrs );
	end_i = Cuda_End_Index( i, &far_nbrs );


	printf("Control cut %d, %d, %d, %d, %d\n", control->nonb_cut, control->bond_cut, type_i,start_i,end_i);

	if ( i < n )
	{
		local = TRUE;
		cutoff = control->nonb_cut;
		//++num_cm_entries;//TB:: Is this correct?
	}
	else
	{
		local = FALSE;
		cutoff = control->bond_cut;
	}


	for ( pj = start_i; pj < end_i; ++pj )
	{
		nbr_pj = &far_nbrs.select.far_nbr_list[pj];
		j = nbr_pj->nbr;
		atom_j = &my_atoms[j];

		if ( nbr_pj->d <= control->nonb_cut )
		{
			type_j = my_atoms[j].type;

			if ( local == TRUE )
			{
				if ( i < j && (j < n || atom_i->orig_id < atom_j->orig_id) )
				{
					++num_cm_entries;
				}
				else if ( i > j && (j < n || atom_j->orig_id > atom_i->orig_id) )
				{
					++num_cm_entries;
				}
			}
			else
			{
				if ( i > j && j < n && atom_j->orig_id < atom_i->orig_id )
				{
					++num_cm_entries;
				}
			}
		}
	}


	cm_entries[i] = num_cm_entries;

	printf("Cm etnries index %d,  %d \n", i, cm_entries[i]);
}

void Cuda_Estimate_CMEntries_Storages( reax_system *system, control_params *control, reax_list **lists, fix_qeq_gpu *qeq_gpu,int nn)
{
	int blocks;

	cuda_malloc( (void **) &qeq_gpu->d_cm_entries,
			system->total_cap * sizeof(int), TRUE, "system:d_cm_entries" );


	blocks = nn / DEF_BLOCK_SIZE +
			(((nn % DEF_BLOCK_SIZE == 0)) ? 0 : 1);

	printf("nn %d, sys n %d \n",nn,system->n);

	hipLaunchKernelGGL(k_estimate_cm_entries_storage, dim3(blocks), dim3(DEF_BLOCK_SIZE), 0, 0,  qeq_gpu->d_fix_my_atoms,
			(control_params *)control->d_control_params,
			*(lists[FAR_NBRS]),system->n, nn,
			qeq_gpu->d_cm_entries);
	hipDeviceSynchronize();
	cudaCheckError();
}

void Cuda_Init_Sparse_Matrix_Indices( reax_system *system, fix_qeq_gpu *qeq_gpu, int n)
{
	int blocks;

	/* init indices */
	Cuda_Scan_Excl_Sum(qeq_gpu->d_cm_entries, qeq_gpu->H.start, n);

	/* init end_indices */
	blocks = system->total_cap / DEF_BLOCK_SIZE
			+ ((system->total_cap % DEF_BLOCK_SIZE == 0) ? 0 : 1);
	hipLaunchKernelGGL(k_init_end_index, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  qeq_gpu->d_cm_entries, qeq_gpu->H.start, qeq_gpu->H.end, n);
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



	printf("K int matvec %f, %f \n", qeq_gpu->t[i],qeq_gpu->s[i]);
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

void  Cuda_Copy_From_Device_Comm_Fix(double *buf, double *x, int n, int offset)
{
	copy_host_device(buf, x+offset, sizeof(double) * n,
			hipMemcpyDeviceToHost, "Cuda_CG::x:get" );
	printf("Copy from device  fix \n");
}

void  Cuda_Copy_To_Device_Comm_Fix(double *buf,double *x,int n,int offset)
{
	copy_host_device(buf, x+offset, sizeof(double) * n,
			hipMemcpyHostToDevice, "Cuda_CG::x:get" );
	printf("Copy to device fix \n");
}


CUDA_GLOBAL void k_update_q(double *temp_buf, double *q, int nn)
{
	int i, c, col;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= nn )
	{
		return;
	}


	q[i] = q[i] +  temp_buf[i];

}

void  Cuda_UpdateQ_And_Copy_To_Device_Comm_Fix(double *buf,fix_qeq_gpu *qeq_gpu,int nn)
{
	double *temp_buf;
	cuda_malloc( (void **) &temp_buf, sizeof(double)*nn, TRUE,
			"Cuda_Allocate_Matrix::start");
	copy_host_device(buf, temp_buf, sizeof(double) * nn,
			hipMemcpyHostToDevice, "Cuda_CG::q:get");

	int blocks;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	printf("Blocks %d \n",blocks);


	hipLaunchKernelGGL(k_update_q, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, temp_buf,qeq_gpu->q,nn);
	hipDeviceSynchronize();

}

CUDA_GLOBAL void k_matvec_csr_fix( sparse_matrix H, real *vec, real *results,
		int num_rows )
{
	int i, c, col;
	real results_row;
	real val;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= num_rows )
	{
		return;
	}

	results_row = 0;

	for ( c = H.start[i]; c < H.end[i]; c++ )
	{
		col = H.entries [c].j;
		val = H.entries[c].val;

		results_row += val * vec[col];
	}

	results[i] = results_row;
}

CUDA_GLOBAL void k_init_q(reax_atom *my_atoms, double *q, double *x,double *eta, int nn)
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


	q[i] = eta[type_i] * x[i];

	printf("Eta:% f, x: %f, Q: %f \n", eta[type_i],x[i],q[i]);
}


void Cuda_Sparse_Matvec_Compute(sparse_matrix *H,double *x, double *q, double *eta, reax_atom *d_fix_my_atoms, int nn, int NN)
{

	int blocks;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	printf("Blocks %d \n",blocks);


	hipLaunchKernelGGL(k_init_q, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, d_fix_my_atoms,q,x,eta,nn);
	hipDeviceSynchronize();

	printf("nn%d,NN%d\n",nn,NN);


	hipLaunchKernelGGL(k_matvec_csr_fix, dim3(blocks), dim3(MATVEC_BLOCK_SIZE), sizeof(real) * MATVEC_BLOCK_SIZE , 0, *H, q, x, nn);
	hipDeviceSynchronize();
	cudaCheckError();

}

void Cuda_Vector_Sum_Fix( real *res, real a, real *x, real b, real *y, int count )
{
	//res = ax + by
	//use the cublas here
	int blocks;

	blocks = (count / DEF_BLOCK_SIZE)+ ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);

	hipLaunchKernelGGL(k_vector_sum, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  res, a, x, b, y, count );
	hipDeviceSynchronize( );
	cudaCheckError( );
}

void Cuda_CG_Preconditioner_Fix(real *res, real *a, real *b, int count)
{

	int blocks;

	blocks = (count / DEF_BLOCK_SIZE) + ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);


	hipLaunchKernelGGL(k_vector_mul, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  res, a, b, count );
	hipDeviceSynchronize( );
	cudaCheckError( );

}

void  Cuda_Copy_Vector_From_Device(real *host_vector, real *device_vector, int nn)
{
	copy_host_device( host_vector, device_vector, sizeof(real) * nn,
			hipMemcpyDeviceToHost, "Cuda_CG::b:get" );
}


int  compute_nearest_pow_2_fix( int blocks)
{

	int result = 0;
	result = (int) EXP2( CEIL( LOG2((double) blocks)));
	return result;
}



void  Cuda_Parallel_Vector_Acc(int nn,fix_qeq_gpu *qeq_gpu,int control_blocks)
{
	int blocks;
	real *output;
	//cuda malloc this
	cuda_malloc((void **) &output, sizeof(real)*(nn), TRUE,
			"Cuda_Allocate_Matrix::start");
	double my_acc, res;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	printf("Blocks %d, control blocks %d \n", blocks, control_blocks);


	int blocks_pow_2 = compute_nearest_pow_2_fix(control_blocks);


	printf("Blocks pow 2 %d \n", blocks_pow_2);

	hipLaunchKernelGGL(k_reduction, dim3(blocks), dim3(DEF_BLOCK_SIZE), sizeof(real) * DEF_BLOCK_SIZE , 0,   qeq_gpu->s, output, nn );
	hipDeviceSynchronize();
	cudaCheckError( );


	hipLaunchKernelGGL(k_reduction, dim3(1), dim3(blocks_pow_2), sizeof(real) * DEF_BLOCK_SIZE , 0,  qeq_gpu->s, output+nn, blocks);
	hipDeviceSynchronize();
	cudaCheckError( );

	copy_host_device( &my_acc, output + nn,
			sizeof(real), hipMemcpyDeviceToHost, "charges:x" );

	my_acc = 0.0;
	res = 0.0;

	double s_sum = MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


	my_acc = res = 0.0;
	cuda_memset( output, 0, sizeof(real) * nn, "cuda_charges_x:q" );

	hipLaunchKernelGGL(k_reduction, dim3(blocks), dim3(DEF_BLOCK_SIZE), sizeof(real) * DEF_BLOCK_SIZE , 0,   qeq_gpu->t, output, nn );
	hipDeviceSynchronize();
	cudaCheckError( );

	hipLaunchKernelGGL(k_reduction, dim3(1), dim3(blocks_pow_2), sizeof(real) * DEF_BLOCK_SIZE , 0,  qeq_gpu->t, output+nn, blocks);
	hipDeviceSynchronize();
	cudaCheckError( );

	copy_host_device( &my_acc, output + nn,
			sizeof(real), hipMemcpyDeviceToHost, "charges:x" );

	double t_sum = MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


	printf("T sum %f, S Sum %f \n", t_sum, s_sum);

	exit(0);
}


void  Cuda_Calculate_Q(int nn,fix_qeq_gpu *qeq_gpu, int charges,int control_blocks)
{

	int i, k;
	double u, s_sum, t_sum;


	Cuda_Parallel_Vector_Acc(nn,qeq_gpu,control_blocks);




	/*real u;//, s_sum, t_sum;
	    rvec2 my_sum, all_sum;
	    real *q;

	    my_sum[0] = 0.0;
	    my_sum[1] = 0.0;
	    q = (real *) workspace->host_scratch;
	    memset( q, 0, system->N * sizeof(real) );

	    cuda_charges_x( system, control, workspace, my_sum );

	#if defined(DEBUG_FOCUS)
	    fprintf( stderr, "Device: my_sum[0]: %f, my_sum[1]: %f\n",
	            my_sum[0], my_sum[1] );
	#endif

	    MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );

	    u = all_sum[0] / all_sum[1];

	#if defined(DEBUG_FOCUS)
	    fprintf( stderr, "Device: u: %f \n", u );
	#endif

	    cuda_charges_st( system, workspace, q, u );

	    Dist( system, mpi_data, q, REAL_PTR_TYPE, MPI_DOUBLE );

	    cuda_charges_updateq( system, workspace, q );*/

}

}

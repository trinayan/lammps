

#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_list.h"
#include "cuda_neighbors.h"
#include "cuda_utils.h"

#include "allocate.h"
#include "index_utils.h"
#include "tool_box.h"
#include "vector.h"

extern "C"
{


void Cuda_Allocate_Control( control_params *control )
{
	cuda_malloc( (void **)&control->d_control_params,
			sizeof(control_params), TRUE, "control_params" );
	copy_host_device( control, control->d_control_params,
			sizeof(control_params), hipMemcpyHostToDevice, "control_params" );
}


CUDA_GLOBAL void Init_Nbrs( ivec *nbrs, int N )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if ( index >= N )
	{
		return;
	}

	nbrs[index][0] = -1;
	nbrs[index][1] = -1;
	nbrs[index][2] = -1;
}


void Cuda_Allocate_Grid( reax_system *system )
{
	int total;
	//    grid_cell local_cell;
	grid *host = &system->my_grid;
	grid *device = &system->d_my_grid;
	//    ivec *nbrs_x = (ivec *) workspace->scratch;

	total = host->ncells[0] * host->ncells[1] * host->ncells[2];
	ivec_Copy( device->ncells, host->ncells );
	rvec_Copy( device->cell_len, host->cell_len );
	rvec_Copy( device->inv_len, host->inv_len );

	ivec_Copy( device->bond_span, host->bond_span );
	ivec_Copy( device->nonb_span, host->nonb_span );
	ivec_Copy( device->vlist_span, host->vlist_span );

	ivec_Copy( device->native_cells, host->native_cells );
	ivec_Copy( device->native_str, host->native_str );
	ivec_Copy( device->native_end, host->native_end );

	device->ghost_cut = host->ghost_cut;
	ivec_Copy( device->ghost_span, host->ghost_span );
	ivec_Copy( device->ghost_nonb_span, host->ghost_nonb_span );
	ivec_Copy( device->ghost_hbond_span, host->ghost_hbond_span );
	ivec_Copy( device->ghost_bond_span, host->ghost_bond_span );

	cuda_malloc( (void **) &device->str, sizeof(int) * total, TRUE,
			"Cuda_Allocate_Grid::grid->str" );
	cuda_malloc( (void **) &device->end, sizeof(int) * total, TRUE,
			"Cuda_Allocate_Grid::grid->end" );
	cuda_malloc( (void **) &device->cutoff, sizeof(real) * total, TRUE,
			"Cuda_Allocate_Grid::grid->cutoff" );

	cuda_malloc( (void **) &device->nbrs_x, sizeof(ivec) * total * host->max_nbrs,
			TRUE, "Cuda_Allocate_Grid::grid->nbrs_x" );
	cuda_malloc( (void **) &device->nbrs_cp, sizeof(rvec) * total * host->max_nbrs,
			TRUE, "Cuda_Allocate_Grid::grid->nbrs_cp" );
	cuda_malloc( (void **) &device->rel_box, sizeof(ivec) * total,
			TRUE, "Cuda_Allocate_Grid::grid->rel_box" );

	//    int block_size = 512;
	//    int blocks = (host->max_nbrs) / block_size + ((host->max_nbrs) % block_size == 0 ? 0 : 1);
	//
	//    Init_Nbrs <<< blocks, block_size >>>
	//        ( nbrs_x, host->max_nbrs );
	//    hipDeviceSynchronize( );
	//    cudaCheckError( );
	//
	//    cuda_malloc( (void **)& device->cells, sizeof(grid_cell) * total,
	//            TRUE, "grid:cells");
	//    fprintf( stderr, " Device cells address --> %ld \n", device->cells );
	//    cuda_malloc( (void **) &device->order,
	//            sizeof(ivec) * (host->total + 1), TRUE, "grid:order" );
	//
	//    local_cell.top = local_cell.mark = local_cell.str = local_cell.end = 0;
	//    fprintf( stderr, "Total cells to be allocated -- > %d \n", total );
	//    for (int i = 0; i < total; i++)
	//    {
	//        //fprintf( stderr, "Address of the local atom -> %ld  \n", &local_cell );
	//
	//        cuda_malloc( (void **) &local_cell.atoms, sizeof(int) * host->max_atoms,
	//                TRUE, "alloc:grid:cells:atoms" );
	//        //fprintf( stderr, "Allocated address of the atoms --> %ld  (%d)\n", local_cell.atoms, host->max_atoms );
	//
	//        cuda_malloc( (void **) &local_cell.nbrs_x, sizeof(ivec) * host->max_nbrs,
	//                TRUE, "alloc:grid:cells:nbrs_x" );
	//        copy_device( local_cell.nbrs_x, nbrs_x, host->max_nbrs * sizeof(ivec), "grid:nbrs_x" );
	//        //fprintf( stderr, "Allocated address of the nbrs_x--> %ld \n", local_cell.nbrs_x );
	//
	//        cuda_malloc( (void **) &local_cell.nbrs_cp, sizeof(rvec) * host->max_nbrs,
	//                TRUE, "alloc:grid:cells:nbrs_cp" );
	//        //fprintf( stderr, "Allocated address of the nbrs_cp--> %ld \n", local_cell.nbrs_cp );
	//
	//        //cuda_malloc( (void **) &local_cell.nbrs, sizeof(grid_cell *) * host->max_nbrs,
	//        //                TRUE, "alloc:grid:cells:nbrs" );
	//        //fprintf( stderr, "Allocated address of the nbrs--> %ld \n", local_cell.nbrs );
	//
	//        copy_host_device( &local_cell, &device->cells[i], sizeof(grid_cell),
	//                hipMemcpyHostToDevice, "grid:cell-alloc" );
	//    }
}


void Cuda_Deallocate_Grid_Cell_Atoms( reax_system *system )
{
	int total;
	grid_cell local_cell;
	grid *host = &system->my_grid;
	grid *device = &system->d_my_grid;

	total = host->ncells[0] * host->ncells[1] * host->ncells[2];

	for (int i = 0; i < total; i++)
	{
		copy_host_device( &local_cell, &device->cells[i],
				sizeof(grid_cell), hipMemcpyDeviceToHost,
				"Cuda_Deallocate_Grid_Cell_Atoms::grid" );
		cuda_free( local_cell.atoms,
				"Cuda_Deallocate_Grid_Cell_Atoms::grid_cell.atoms" );
	}
}


void Cuda_Allocate_Grid_Cell_Atoms( reax_system *system, int cap )
{
	int i, total;
	grid_cell local_cell;
	grid *host = &system->my_grid;
	grid *device = &system->d_my_grid;

	total = host->ncells[0] * host->ncells[1] * host->ncells[2];

	for (i = 0; i < total; i++)
	{
		copy_host_device( &local_cell, &device->cells[i],
				sizeof(grid_cell), hipMemcpyDeviceToHost, "grid:cell-dealloc" );
		cuda_malloc( (void **)&local_cell.atoms, sizeof(int) * cap,
				TRUE, "realloc:grid:cells:atoms" );
		copy_host_device( &local_cell, &device->cells[i],
				sizeof(grid_cell), hipMemcpyHostToDevice, "grid:cell-realloc" );
	}
}


void Cuda_Allocate_System( reax_system *system )
{
	/* atoms */
	printf("Total cap %d \n", system->total_cap);
	cuda_malloc( (void **) &system->d_my_atoms,
			system->total_cap * sizeof(reax_atom),
			TRUE, "system:d_my_atoms" );
	cuda_malloc( (void **) &system->d_numH, sizeof(int), TRUE, "system:d_numH" );

	/* list management */
	cuda_malloc( (void **) &system->d_far_nbrs,
			system->total_cap * sizeof(int), TRUE, "system:d_far_nbrs" );
	cuda_malloc( (void **) &system->d_max_far_nbrs,
			system->total_cap * sizeof(int), TRUE, "system:d_max_far_nbrs" );
	cuda_malloc( (void **) &system->d_total_far_nbrs,
			sizeof(int), TRUE, "system:d_total_far_nbrs" );
	cuda_malloc( (void **) &system->d_realloc_far_nbrs,
			sizeof(int), TRUE, "system:d_realloc_far_nbrs" );

	cuda_malloc( (void **) &system->d_bonds,
			system->total_cap * sizeof(int), TRUE, "system:d_bonds" );
	cuda_malloc( (void **) &system->d_max_bonds,
			system->total_cap * sizeof(int), TRUE, "system:d_max_bonds" );
	cuda_malloc( (void **) &system->d_total_bonds,
			sizeof(int), TRUE, "system:d_total_bonds" );
	cuda_malloc( (void **) &system->d_realloc_bonds,
			sizeof(int), TRUE, "system:d_realloc_bonds" );

	cuda_malloc( (void **) &system->d_hbonds,
			system->total_cap * sizeof(int), TRUE, "system:d_hbonds" );
	cuda_malloc( (void **) &system->d_max_hbonds,
			system->total_cap * sizeof(int), TRUE, "system:d_max_hbonds" );
	cuda_malloc( (void **) &system->d_total_hbonds,
			sizeof(int), TRUE, "system:d_total_hbonds" );
	cuda_malloc( (void **) &system->d_realloc_hbonds,
			sizeof(int), TRUE, "system:d_realloc_hbonds" );

	cuda_malloc( (void **) &system->d_cm_entries,
			system->total_cap * sizeof(int), TRUE, "system:d_cm_entries" );
	cuda_malloc( (void **) &system->d_max_cm_entries,
			system->total_cap * sizeof(int), TRUE, "system:d_max_cm_entries" );
	cuda_malloc( (void **) &system->d_total_cm_entries,
			sizeof(int), TRUE, "system:d_total_cm_entries" );
	cuda_malloc( (void **) &system->d_realloc_cm_entries,
			sizeof(int), TRUE, "system:d_realloc_cm_entries" );

	cuda_malloc( (void **) &system->d_total_thbodies,
			sizeof(int), TRUE, "system:d_total_thbodies" );

	/* simulation boxes */
	cuda_malloc( (void **) &system->d_big_box,
			sizeof(simulation_box), TRUE, "system:d_big_box" );
	cuda_malloc( (void **) &system->d_my_box,
			sizeof(simulation_box), TRUE, "system:d_my_box" );
	cuda_malloc( (void **) &system->d_my_ext_box,
			sizeof(simulation_box), TRUE, "d_my_ext_box" );

	/* interaction parameters */
	cuda_malloc( (void **) &system->reax_param.d_sbp,
			system->reax_param.num_atom_types * sizeof(single_body_parameters),
			TRUE, "system:d_sbp" );

	cuda_malloc( (void **) &system->reax_param.d_tbp,
			POW( system->reax_param.num_atom_types, 2.0 ) * sizeof(two_body_parameters),
			TRUE, "system:d_tbp" );

	cuda_malloc( (void **) &system->reax_param.d_thbp,
			POW( system->reax_param.num_atom_types, 3.0 ) * sizeof(three_body_header),
			TRUE, "system:d_thbp" );

	cuda_malloc( (void **) &system->reax_param.d_hbp,
			POW( system->reax_param.num_atom_types, 3.0 ) * sizeof(hbond_parameters),
			TRUE, "system:d_hbp" );

	cuda_malloc( (void **) &system->reax_param.d_fbp,
			POW( system->reax_param.num_atom_types, 4.0 ) * sizeof(four_body_header),
			TRUE, "system:d_fbp" );

	cuda_malloc( (void **) &system->reax_param.d_gp.l,
			system->reax_param.gp.n_global * sizeof(real), TRUE, "system:d_gp.l" );

	system->reax_param.d_gp.n_global = 0;
	system->reax_param.d_gp.vdw_type = 0;
}


static int Cuda_Reallocate_System( reax_system *system, storage *workspace,
		int old_total_cap, int total_cap, char *msg )
{
	int *temp;
	reax_atom *temp_atom;

	temp = (int *) workspace->scratch;
	temp_atom = (reax_atom*) workspace->scratch;

	/* free the existing storage for atoms, leave other info allocated */
	copy_device( temp_atom, system->d_my_atoms, old_total_cap * sizeof(reax_atom),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_my_atoms, "system::d_my_atoms" );
	cuda_malloc( (void **) &system->d_my_atoms, sizeof(reax_atom) * total_cap,
			TRUE, "system::d_my_atoms" );
	copy_device( system->d_my_atoms, temp, old_total_cap * sizeof(reax_atom),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_far_nbrs, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_far_nbrs, "system::d_far_nbrs" );
	cuda_malloc( (void **) &system->d_far_nbrs,
			system->total_cap * sizeof(int), TRUE, "system::d_far_nbrs" );
	copy_device( system->d_far_nbrs, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_max_far_nbrs, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_max_far_nbrs, "system::d_max_far_nbrs" );
	cuda_malloc( (void **) &system->d_max_far_nbrs,
			system->total_cap * sizeof(int), TRUE, "system::d_max_far_nbrs" );
	copy_device( system->d_max_far_nbrs, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_bonds, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_bonds, "system::d_bonds" );
	cuda_malloc( (void **) &system->d_bonds,
			system->total_cap * sizeof(int), TRUE, "system::d_bonds" );
	copy_device( system->d_bonds, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_max_bonds, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_max_bonds, "system::d_max_bonds" );
	cuda_malloc( (void **) &system->d_max_bonds,
			system->total_cap * sizeof(int), TRUE, "system::d_max_bonds" );
	copy_device( system->d_max_bonds, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_hbonds, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_hbonds, "system::d_hbonds" );
	cuda_malloc( (void **) &system->d_hbonds,
			system->total_cap * sizeof(int), TRUE, "system::d_hbonds" );
	copy_device( system->d_hbonds, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_max_hbonds, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_max_hbonds, "system::d_max_hbonds" );
	cuda_malloc( (void **) &system->d_max_hbonds,
			system->total_cap * sizeof(int), TRUE, "system::d_max_hbonds" );
	copy_device( system->d_max_hbonds, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_cm_entries, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_cm_entries, "system::d_cm_entries" );
	cuda_malloc( (void **) &system->d_cm_entries,
			system->total_cap * sizeof(int), TRUE, "system::d_cm_entries" );
	copy_device( system->d_cm_entries, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	copy_device( temp, system->d_max_cm_entries, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );
	cuda_free( system->d_max_cm_entries, "system::d_max_cm_entries" );
	cuda_malloc( (void **) &system->d_max_cm_entries,
			system->total_cap * sizeof(int), TRUE, "system::d_max_cm_entries" );
	copy_device( system->d_max_cm_entries, temp, old_total_cap * sizeof(int),
			"Cuda_Reallocate_System::temp" );

	return 1;
}


void Cuda_Allocate_Simulation_Data( simulation_data *data )
{
	cuda_malloc( (void **) &data->d_simulation_data,
			sizeof(simulation_data), TRUE, "simulation_data" );
}


void Cuda_Allocate_Workspace( reax_system *system, control_params *control, 
		storage *workspace, int local_cap, int total_cap )
{
	int total_real, total_rvec, local_rvec;

	workspace->allocated = TRUE;

	total_real = total_cap * sizeof(real);
	total_rvec = total_cap * sizeof(rvec);
	local_rvec = local_cap * sizeof(rvec);

	/* communication storage */
	/*
       workspace->tmp_dbl = NULL;
       workspace->tmp_rvec = NULL;
       workspace->tmp_rvec2 = NULL;
	 */

	/* bond order related storage  */
	cuda_malloc( (void **) &workspace->total_bond_order, total_real, TRUE, "total_bo" );
	cuda_malloc( (void **) &workspace->Deltap, total_real, TRUE, "Deltap" );
	cuda_malloc( (void **) &workspace->Deltap_boc, total_real, TRUE, "Deltap_boc" );
	cuda_malloc( (void **) &workspace->dDeltap_self, total_rvec, TRUE, "dDeltap_self" );
	cuda_malloc( (void **) &workspace->Delta, total_real, TRUE, "Delta" );
	cuda_malloc( (void **) &workspace->Delta_lp, total_real, TRUE, "Delta_lp" );
	cuda_malloc( (void **) &workspace->Delta_lp_temp, total_real, TRUE, "Delta_lp_temp" );
	cuda_malloc( (void **) &workspace->dDelta_lp, total_real, TRUE, "Delta_lp_temp" );
	cuda_malloc( (void **) &workspace->dDelta_lp_temp, total_real, TRUE, "dDelta_lp_temp" );
	cuda_malloc( (void **) &workspace->Delta_e, total_real, TRUE, "Delta_e" );
	cuda_malloc( (void **) &workspace->Delta_boc, total_real, TRUE, "Delta_boc" );
	cuda_malloc( (void **) &workspace->nlp, total_real, TRUE, "nlp" );
	cuda_malloc( (void **) &workspace->nlp_temp, total_real, TRUE, "nlp_temp" );
	cuda_malloc( (void **) &workspace->Clp, total_real, TRUE, "Clp" );
	cuda_malloc( (void **) &workspace->vlpex, total_real, TRUE, "vlpex" );
	cuda_malloc( (void **) &workspace->bond_mark, total_real, TRUE, "bond_mark" );

	/* charge matrix storage */
	//TB:: Moved to CudAllocateStorageForFixQEq
	/*if ( control->cm_solver_pre_comp_type == DIAG_PC )
	{
		cuda_malloc( (void **) &workspace->Hdia_inv, total_cap * sizeof(real), TRUE, "Hdia_inv" );
	}
	cuda_malloc( (void **) &workspace->b_s, total_cap * sizeof(real), TRUE, "b_s" );
	cuda_malloc( (void **) &workspace->b_t, total_cap * sizeof(real), TRUE, "b_t" );
	cuda_malloc( (void **) &workspace->b_prc, total_cap * sizeof(real), TRUE, "b_prc" );
	cuda_malloc( (void **) &workspace->b_prm, total_cap * sizeof(real), TRUE, "b_prm" );
	cuda_malloc( (void **) &workspace->s, total_cap * sizeof(real), TRUE, "s" );
	cuda_malloc( (void **) &workspace->t, total_cap * sizeof(real), TRUE, "t" );*/
	if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
			control->cm_solver_pre_comp_type == ILUT_PAR_PC )
	{
		cuda_malloc( (void **) &workspace->droptol, total_cap * sizeof(real), TRUE, "droptol" );
	}
	cuda_malloc( (void **) &workspace->b, total_cap * sizeof(rvec2), TRUE, "b" );
	cuda_malloc( (void **) &workspace->x, total_cap * sizeof(rvec2), TRUE, "x" );

	switch ( control->cm_solver_type )
	{
	case GMRES_S:
	case GMRES_H_S:
		cuda_malloc( (void **) &workspace->y, (RESTART+1)*sizeof(real), TRUE, "y" );
		cuda_malloc( (void **) &workspace->z, (RESTART+1)*sizeof(real), TRUE, "z" );
		cuda_malloc( (void **) &workspace->g, (RESTART+1)*sizeof(real), TRUE, "g" );
		cuda_malloc( (void **) &workspace->h, (RESTART+1)*(RESTART+1)*sizeof(real), TRUE, "h" );
		cuda_malloc( (void **) &workspace->hs, (RESTART+1)*sizeof(real), TRUE, "hs" );
		cuda_malloc( (void **) &workspace->hc, (RESTART+1)*sizeof(real), TRUE, "hc" );
		cuda_malloc( (void **) &workspace->v, (RESTART+1)*(RESTART+1)*sizeof(real), TRUE, "v" );
		break;

	case SDM_S:
		break;

	case CG_S:
		//TB:: Moved to CudaAllocateStorageForFixQEq
		/*cuda_malloc( (void **) &workspace->r, total_cap * sizeof(real), TRUE, "r" );
		cuda_malloc( (void **) &workspace->d, total_cap * sizeof(real), TRUE, "d" );
		cuda_malloc( (void **) &workspace->q, total_cap * sizeof(real), TRUE, "q" );
		cuda_malloc( (void **) &workspace->p, total_cap * sizeof(real), TRUE, "p" );*/
		cuda_malloc( (void **) &workspace->r2, total_cap * sizeof(rvec2), TRUE, "r2" );
		cuda_malloc( (void **) &workspace->d2, total_cap * sizeof(rvec2), TRUE, "d2" );
		cuda_malloc( (void **) &workspace->q2, total_cap * sizeof(rvec2), TRUE, "q2" );
		cuda_malloc( (void **) &workspace->p2, total_cap * sizeof(rvec2), TRUE, "p2" );
		break;

	default:
		fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
		exit( INVALID_INPUT );
		break;
	}

	/* integrator storage */
	cuda_malloc( (void **) &workspace->v_const, local_rvec, TRUE, "v_const" );

	/* storage for analysis */
	if( control->molecular_analysis || control->diffusion_coef )
	{
		cuda_malloc( (void **) &workspace->mark, local_cap * sizeof(int), TRUE, "mark" );
		cuda_malloc( (void **) &workspace->old_mark, local_cap * sizeof(int), TRUE, "old_mark" );
	}
	else
	{
		workspace->mark = workspace->old_mark = NULL;
	}

	if( control->diffusion_coef )
	{
		cuda_malloc( (void **) &workspace->x_old, local_cap * sizeof(rvec), TRUE, "x_old" );
	}
	else
	{
		workspace->x_old = NULL;
	}

	/* force related storage */
	cuda_malloc( (void **) &workspace->f, total_cap * sizeof(rvec), TRUE, "f" );
	cuda_malloc( (void **) &workspace->CdDelta, total_cap * sizeof(rvec), TRUE, "CdDelta" );

	/* Taper params */
	cuda_malloc( (void **) &workspace->Tap, 8 * sizeof(real), TRUE, "Tap" );


}


void Cuda_Deallocate_Workspace( control_params *control, storage *workspace )
{
	if ( workspace->allocated == FALSE )
	{
		return;
	}

	workspace->allocated = FALSE;

	/* communication storage */
	/*
       workspace->tmp_dbl = NULL;
       workspace->tmp_rvec = NULL;
       workspace->tmp_rvec2 = NULL;
	 */

	/* bond order related storage  */
	cuda_free( workspace->total_bond_order, "total_bo" );
	cuda_free( workspace->Deltap, "Deltap" );
	cuda_free( workspace->Deltap_boc, "Deltap_boc" );
	cuda_free( workspace->dDeltap_self, "dDeltap_self" );
	cuda_free( workspace->Delta, "Delta" );
	cuda_free( workspace->Delta_lp, "Delta_lp" );
	cuda_free( workspace->Delta_lp_temp, "Delta_lp_temp" );
	cuda_free( workspace->dDelta_lp, "Delta_lp_temp" );
	cuda_free( workspace->dDelta_lp_temp, "dDelta_lp_temp" );
	cuda_free( workspace->Delta_e, "Delta_e" );
	cuda_free( workspace->Delta_boc, "Delta_boc" );
	cuda_free( workspace->nlp, "nlp" );
	cuda_free( workspace->nlp_temp, "nlp_temp" );
	cuda_free( workspace->Clp, "Clp" );
	cuda_free( workspace->vlpex, "vlpex" );
	cuda_free( workspace->bond_mark, "bond_mark" );

	/* charge matrix storage */
	if ( control->cm_solver_pre_comp_type == DIAG_PC )
	{
		cuda_free( workspace->Hdia_inv, "Hdia_inv" );
	}
	cuda_free( workspace->b_s, "b_s" );
	cuda_free( workspace->b_t, "b_t" );
	cuda_free( workspace->b_prc, "b_prc" );
	cuda_free( workspace->b_prm, "b_prm" );
	cuda_free( workspace->s, "s" );
	cuda_free( workspace->t, "t" );
	if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
			control->cm_solver_pre_comp_type == ILUT_PAR_PC )
	{
		cuda_free( workspace->droptol, "droptol" );
	}
	cuda_free( workspace->b, "b" );
	cuda_free( workspace->x, "x" );

	switch ( control->cm_solver_type )
	{
	case GMRES_S:
	case GMRES_H_S:
		cuda_free( workspace->y, "y" );
		cuda_free( workspace->z, "z" );
		cuda_free( workspace->g, "g" );
		cuda_free( workspace->h, "h" );
		cuda_free( workspace->hs, "hs" );
		cuda_free( workspace->hc, "hc" );
		cuda_free( workspace->v, "v" );
		break;

	case SDM_S:
		break;

	case CG_S:
		cuda_free( workspace->r, "r" );
		cuda_free( workspace->d, "d" );
		cuda_free( workspace->q, "q" );
		cuda_free( workspace->p, "p" );
		cuda_free( workspace->r2, "r2" );
		cuda_free( workspace->d2, "d2" );
		cuda_free( workspace->q2, "q2" );
		cuda_free( workspace->p2, "p2" );
		break;

	default:
		fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
		exit( INVALID_INPUT );
		break;
	}

	/* integrator storage */
	cuda_free( workspace->v_const, "v_const" );

	/* storage for analysis */
	if( control->molecular_analysis || control->diffusion_coef )
	{
		cuda_free( workspace->mark, "mark" );
		cuda_free( workspace->old_mark, "old_mark" );
	}
	else
	{
		workspace->mark = workspace->old_mark = NULL;
	}

	if( control->diffusion_coef )
	{
		cuda_free( workspace->x_old, "x_old" );
	}
	else
	{
		workspace->x_old = NULL;
	}

	/* force related storage */
	cuda_free( workspace->f, "f" );
	cuda_free( workspace->CdDelta, "CdDelta" );

	/* Taper params */
	cuda_free( workspace->Tap, "Tap" );
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


void Cuda_Reallocate_Neighbor_List( reax_list *far_nbrs, size_t n, size_t num_intrs )
{
	Cuda_Delete_List( far_nbrs );
	Cuda_Make_List( n, num_intrs, TYP_FAR_NEIGHBOR, far_nbrs );
}


void Cuda_Reallocate_HBonds_List( reax_list *hbonds, size_t n, size_t num_intrs )
{
	Cuda_Delete_List( hbonds );
	Cuda_Make_List( n, num_intrs, TYP_HBOND, hbonds );
}


void Cuda_Reallocate_Bonds_List( reax_list *bonds, size_t n, size_t num_intrs )
{
	Cuda_Delete_List( bonds );
	Cuda_Make_List( n, num_intrs, TYP_BOND, bonds );
}


void Cuda_Reallocate_Thbodies_List( reax_list *thbodies, size_t n, size_t num_intrs )
{
	Cuda_Delete_List( thbodies );
	Cuda_Make_List( n, num_intrs, TYP_THREE_BODY, thbodies );

}


void Cuda_ReAllocate( reax_system *system, control_params *control,
		simulation_data *data, storage *workspace, reax_list **lists)
{

	//TB:: device workspace is passed by host. Verify later
	int num_bonds, est_3body, Hflag, ret;
	int renbr, newsize;
	reallocate_data *realloc;
	reax_list *far_nbrs;
	char msg[200];

	int old_total_cap = 0;

	int mincap = system->mincap;
	double safezone = system->safezone;
	double saferzone = system->saferzone;

	realloc = &(workspace->realloc);

	if( system->n >= DANGER_ZONE * system->local_cap ||
			(0 && system->n <= LOOSE_ZONE * system->local_cap) ) {
		system->local_cap = MAX( (int)(system->n * safezone), mincap );
	}

	int Nflag = 0;
	if( system->N >= DANGER_ZONE * system->total_cap ||
			(0 && system->N <= LOOSE_ZONE * system->total_cap) ) {
		Nflag = 1;
		old_total_cap = system->total_cap;
		system->total_cap = MAX( (int)(system->N * safezone), mincap );
	}

	if (Nflag) {
		/* system */
		ret = Cuda_Reallocate_System( system,workspace,  old_total_cap, system->total_cap, msg );
		if (ret != SUCCESS) {
			printf("Not enough space for list \n");
			exit(0);
		}
		Cuda_Deallocate_Workspace( control, workspace );
		Cuda_Allocate_Workspace( system, control, workspace, system->local_cap,
				system->total_cap );

	}

	renbr = (data->step - data->prev_steps) % control->reneighbor == 0;

	if (renbr) {
		far_nbrs = lists[FAR_NBRS];

		if (Nflag || realloc->num_far >= far_nbrs->num_intrs * DANGER_ZONE) {
			if (realloc->num_far > far_nbrs->num_intrs) {
				printf("Num far %d , far nbrs %d\n ", realloc->num_far, far_nbrs->num_intrs);
				printf("Ran out of space \n");
				exit(0);
			}

			newsize = static_cast<int>
			(MAX( realloc->num_far*safezone, mincap*MIN_NBRS ));
			Cuda_Reallocate_Neighbor_List( far_nbrs, system->total_cap, system->total_far_nbrs );
			Cuda_Init_Neighbor_Indices( system, lists );


			//TB:: Verify if both are same
			realloc->num_far = 0;
			realloc->far_nbrs = FALSE;

		}
	}

	/* hydrogen bonds list */
	if ( control->hbond_cut > 0.0 && system->numH > 0 )
	{

		if ( Nflag == TRUE || realloc->hbonds == TRUE )
		{
#if defined(DEBUG_FOCUS)
			fprintf( stderr, "p%d: reallocating hbonds: total_hbonds=%d space=%dMB\n",
					system->my_rank, system->total_hbonds,
					(int)(system->total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif

			Cuda_Reallocate_HBonds_List( lists[HBONDS], system->total_cap, system->total_hbonds );

			Cuda_Init_HBond_Indices( system, workspace, lists );

			realloc->hbonds = FALSE;
		}
	}

	/* bonds list */
	if ( Nflag == TRUE || realloc->bonds == TRUE )
	{
#if defined(DEBUG_FOCUS)
		fprintf( stderr, "p%d: reallocating bonds: total_bonds=%d, space=%dMB\n",
				system->my_rank, system->total_bonds,
				(int)(system->total_bonds * sizeof(bond_data) / (1024 * 1024)) );
#endif

		Cuda_Reallocate_Bonds_List( lists[BONDS], system->total_cap, system->total_bonds );

		Cuda_Init_Bond_Indices( system, lists );

		realloc->bonds = FALSE;
	}


	/* 3-body list */
	if ( Nflag == TRUE || realloc->thbody == TRUE )
	{
#if defined(DEBUG_FOCUS)
		fprintf( stderr, "p%d: reallocating thbody list: num_thbody=%d, space=%dMB\n",
				system->my_rank, system->total_thbodies,
				(int)(system->total_thbodies * sizeof(three_body_interaction_data) /
						(1024*1024)) );
#endif

		Cuda_Reallocate_Thbodies_List( lists[THREE_BODIES],
				system->total_thbodies_indices, system->total_thbodies );

		realloc->thbody = FALSE;
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
}


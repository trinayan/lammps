
#include "cuda_init_md.h"

#include "cuda_allocate.h"
#include "cuda_list.h"
#include "cuda_copy.h"
#include "cuda_forces.h"
#include "cuda_integrate.h"
#include "cuda_neighbors.h"
#include "cuda_reset_tools.h"
#include "cuda_system_props.h"
#include "cuda_utils.h"

#if defined(PURE_REAX)
  #include "box.h"
  #include "comm_tools.h"
  #include "grid.h"
  #include "init_md.h"
  #include "io_tools.h"
#ifdef __cplusplus
extern "C" {
#endif
  #include "lookup.h"
#ifdef __cplusplus
}
#endif
  #include "random.h"
  #include "reset_tools.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_box.h"
  #include "reax_comm_tools.h"
  #include "reax_grid.h"
  #include "reax_init_md.h"
  #include "reax_io_tools.h"
  #include "reax_list.h"
  #include "reax_lookup.h"
  #include "reax_random.h"
  #include "reax_reset_tools.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif


static void Cuda_Init_Scratch_Space( storage *workspace )
{
    cuda_malloc( (void **)&workspace->scratch, DEVICE_SCRATCH_SIZE, TRUE,
            "Cuda_Init_Scratch_Space::workspace->scratch" );

    workspace->host_scratch = (void *) smalloc( HOST_SCRATCH_SIZE,
            "Cuda_Init_Scratch_Space::workspace->host_scratch" );
}


int Cuda_Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data, char *msg )
{
    //TB:: Modified entire function with code from cpu version	
    int i;
    reax_atom *atom;

    /* determine the local and total capacity */
    system->local_cap = MAX( (int)(system->n * SAFE_ZONE), MIN_CAP );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0 )
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);
            if ( system->reax_param.sbp[ atom->type ].p_hbond == 1 )
                atom->Hindex = system->numH++;
            else atom->Hindex = -1;
        }
    system->Hcap = (int)(MAX( system->numH * SAFER_ZONE, MIN_CAP ));

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
            system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
            system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
            system->my_rank, system->numH, system->Hcap );
#endif

    return SUCCESS;

}


void Cuda_Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data )
{
    Cuda_Allocate_Simulation_Data( data );

    Reset_Simulation_Data( data );

    if ( !control->restart )
    {
        data->step = data->prev_steps = 0;
    }

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->bigN;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein;
        control->virial = 0;
        if ( !control->restart || (control->restart && control->random_vel) )
        {
            data->therm.G_xi = control->Tau_T *
                               (2.0 * data->sys_en.e_kin - data->N_f * K_B * control->T );
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->therm.v_xi_old = 0;
            data->therm.xi = 0;
        }
        break;

    /* Semi-Isotropic NPT */
    case sNPT:
        data->N_f = 3 * system->bigN + 4;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
        data->N_f = 3 * system->bigN + 9;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;

        fprintf( stderr, "p%d: init_simulation_data: option not yet implemented\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        break;

    default:
        fprintf( stderr, "p%d: init_simulation_data: ensemble not recognized\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }

    MPI_Barrier( MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.start = Get_Time( );

#if defined(LOG_PERFORMANCE)
        Reset_Timing( &data->timing );
#endif
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "data->N_f: %8.3f\n", data->N_f );
#endif
}


void Cuda_Init_Workspace( reax_system *system, control_params *control,
        gpu_storage *workspace )
{

    Cuda_Allocate_Workspace( system, control, workspace,
            system->local_cap, system->total_cap );

    memset( &workspace->realloc, 0, sizeof(reallocate_data) );
    //TB::Commented out for now
    Cuda_Reset_Workspace( system, workspace );

    /*Init_Taper( control, workspace );*/
}


void Cuda_Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    Cuda_Estimate_Neighbors( system );

    Cuda_Make_List( system->total_cap, system->total_far_nbrs,
            TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated far_nbrs: num_far=%d, space=%dMB\n",
            system->my_rank, system->total_far_nbrs,
            (int)(system->total_far_nbrs * sizeof(far_neighbor_data) / (1024 * 1024)) );
    fprintf( stderr, "N: %d and total_cap: %d \n", system->N, system->total_cap );
#endif

    Cuda_Init_Neighbor_Indices( system, lists );

    Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

    /* estimate storage for bonds, hbonds, and sparse matrix */
    Cuda_Estimate_Storages( system, control, lists,
            TRUE, TRUE, TRUE, data->step );

    Cuda_Allocate_Matrix( &workspace->d_workspace->H, system->total_cap, system->total_cm_entries );
    Cuda_Init_Sparse_Matrix_Indices( system, &workspace->d_workspace->H );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p:%d - allocated H matrix: max_entries: %d, space=%dMB\n",
            system->my_rank, system->total_cm_entries,
            (int)(system->total_cm_entries * sizeof(sparse_matrix_entry) / (1024 * 1024)) );
#endif

    if ( control->hbond_cut > 0.0 && system->numH > 0 )
    {
        Cuda_Make_List( system->total_cap, system->total_hbonds, TYP_HBOND, lists[HBONDS] );
        Cuda_Init_HBond_Indices( system, workspace, lists );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: allocated hbonds: total_hbonds=%d, space=%dMB\n",
                system->my_rank, system->total_hbonds,
                (int)(system->total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif
    }

    /* bonds list */
    Cuda_Make_List( system->total_cap, system->total_bonds, TYP_BOND, lists[BONDS] );
    Cuda_Init_Bond_Indices( system, lists );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated bonds: total_bonds=%d, space=%dMB\n",
            system->my_rank, total_bonds,
            (int)(total_bonds * sizeof(bond_data) / (1024 * 1024)) );
#endif

    /* 3bodies list: since a more accurate estimate of the num.
     * three body interactions requires that bond orders have
     * been computed, delay estimation until computation */
}


void Cuda_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
	gpu_storage *d_workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];

    Cuda_Init_Scratch_Space( workspace );

    //TB:: Check init MPI code in original vs lammps uSER-REAXC. Why are they different?
    Init_MPI_Datatypes( system, workspace, mpi_data );

    if ( Cuda_Init_System( system, control, data, workspace, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "[ERROR] p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "[ERROR] p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

 

    //TB::Commented out grid functionality
    /*Cuda_Allocate_Grid( system );
    Sync_Grid( &system->my_grid, &system->d_my_grid );
*/
    //validate_grid( system );

    Cuda_Init_Simulation_Data( system, control, data );

    Cuda_Init_Workspace( system, control, d_workspace);

    Cuda_Allocate_Control( control );

    Cuda_Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    /* Lookup Tables */
    if ( control->tabulate )
    {
      printf("Unable to link exiting \n");
      exit(0);
        //Init_Lookup_Tables( system, control, workspace->d_workspace, mpi_data );
    }

    Cuda_Init_Block_Sizes( system, control );

#if defined(DEBUG_FOCUS)
    Cuda_Print_Mem_Usage( );
#endif
}

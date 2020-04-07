/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#include "reaxc_types.h"

#include <stddef.h>

#include "init_md.h"
#include "allocate.h"
#include "box.h"
#include "comm_tools.h"
#include "forces.h"
#include "grid.h"
#include "integrate.h"
#include "io_tools.h"
#include "list.h"
#include "lookup.h"
#include "neighbors.h"
#include "random.h"
#include "reset_tools.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"

//#if defined(PURE_REAX)
/************************ initialize system ************************/
static int Reposition_Atoms( reax_system * const system, control_params * const control,
        simulation_data * const data, mpi_datatypes * const mpi_data, char * const msg )
{
    int i;
    rvec dx;

    /* reposition atoms */
    /* fit atoms to periodic box */
    if ( control->reposition_atoms == 0 )
    {
        rvec_MakeZero( dx );
    }
    /* put center of mass to center */
    else if ( control->reposition_atoms == 1 )
    {
        rvec_Scale( dx, 0.5, system->big_box.box_norms );
        rvec_ScaledAdd( dx, -1., data->xcm );
    }
    /* put center of mass to origin */
    else if ( control->reposition_atoms == 2 )
    {
        rvec_Scale( dx, -1., data->xcm );
    }
    else
    {
        strcpy( msg, "[ERROR] reposition_atoms: invalid option" );
        return FAILURE;
    }

    for ( i = 0; i < system->n; ++i )
    {
        // Inc_on_T3_Gen( system->my_atoms[i].x, dx, &(system->big_box) );
        rvec_Add( system->my_atoms[i].x, dx );
    }

    return SUCCESS;
}



void Generate_Initial_Velocities( reax_system * const system, real T )
{
    int i;
    real m, scale, norm;

    if ( T <= 0.1 )
    {
        for ( i = 0; i < system->n; i++ )
        {
            rvec_MakeZero( system->my_atoms[i].v );
        }
    }
    else
    {
        Randomize( );

        for ( i = 0; i < system->n; i++ )
        {
            rvec_Random( system->my_atoms[i].v );

            norm = rvec_Norm_Sqr( system->my_atoms[i].v );
            m = system->reax_param.sbp[ system->my_atoms[i].type ].mass;
            scale = SQRT( m * norm / (3.0 * K_B * T) );

            rvec_Scale( system->my_atoms[i].v, 1.0 / scale, system->my_atoms[i].v );
        }
    }
}


void Init_System( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data )
{
    int i;
    reax_atom *atom;
    int nrecv[MAX_NBRS];

    Setup_New_Grid( system, control, MPI_COMM_WORLD );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d GRID:\n", system->my_rank );
    Print_Grid( &system->my_grid, stderr );
#endif

    Bin_My_Atoms( system, workspace );
    Reorder_My_Atoms( system, workspace );

    /* estimate N and total capacity */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        nrecv[i] = 0;
    }
    MPI_Barrier( MPI_COMM_WORLD );
    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
            Estimate_Boundary_Atoms, Unpack_Estimate_Message, TRUE );

    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0.0 )
    {
        for ( i = 0; i < system->N; ++i )
        {
            atom = &system->my_atoms[i];

            if ( system->reax_param.sbp[ atom->type ].p_hbond == H_ATOM )
            {
                atom->Hindex = system->numH++;
            }
            else
            {
                atom->Hindex = -1;
            }
        }
    }
    //Tried fix
    //system->Hcap = MAX( system->numH * SAFER_ZONE, MIN_CAP );
    system->Hcap = MAX( system->n * SAFER_ZONE, MIN_CAP );

    /* list management */
    system->far_nbrs = smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->far_nbrs" );
    
    printf("Allocating max nbrs in init \n");
    system->max_far_nbrs = smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->max_far_nbrs" );

   
    system->bonds = smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->bonds" );
    system->max_bonds = smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->max_bonds" );

    system->hbonds = smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->hbonds" );
    system->max_hbonds = smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->max_hbonds" );

    system->cm_entries = smalloc( sizeof(int) * system->local_cap,
            "ReAllocate_System::system->cm_entries" );
    system->max_cm_entries = smalloc( sizeof(int) * system->local_cap,
            "ReAllocate_System::max_cm_entries->max_hbonds" );
    
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
             system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
             system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
             system->my_rank, system->numH, system->Hcap );
#endif

    Compute_Total_Mass( system, data, mpi_data->comm_mesh3D );

    Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

//    if( Reposition_Atoms( system, control, data, mpi_data ) == FAILURE )
//    {
//        return FAILURE;
//    }

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Generate_Initial_Velocities( system, control->T_init );
    }

    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
}


/************************ initialize simulation data ************************/
void Init_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data )
{
    Reset_Simulation_Data( data );

    if ( !control->restart )
    {
        data->step = data->prev_steps = 0;
    }

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->bigN;
        control->Evolve = Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        control->Evolve = Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        control->Evolve = Velocity_Verlet_Nose_Hoover_NVT_Klein;
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
        control->Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        control->Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
        fprintf( stderr, "[ERROR] p%d: init_simulation_data: option not yet implemented\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );

        data->N_f = 3 * system->bigN + 9;
        control->Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        /*if( !control->restart ) {
          data->therm.G_xi = control->Tau_T *
          (2.0 * data->my_en.e_Kin - data->N_f * K_B * control->T );
          data->therm.v_xi = data->therm.G_xi * control->dt;
          data->iso_bar.eps = (1.0 / 3.0) * LOG(system->box.volume);
          data->inv_W = 1.0 /
          ( data->N_f * K_B * control->T * SQR(control->Tau_P) );
          Compute_Pressure( system, control, data, out_control );
          }*/
        break;

    default:
        fprintf( stderr, "[ERROR] p%d: init_simulation_data: ensemble not recognized\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }

    /* initialize the timer(s) */
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


/*void Init_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data )
{
    Reset_Simulation_Data( data );

#if defined(LOG_PERFORMANCE)
    Reset_Timing( &data->timing );
#endif

    //if( !control->restart )
    data->step = data->prev_steps = 0;
}*/



/************************ initialize workspace ************************/
/* Initialize Taper params */
void Init_Taper( control_params * const control,  storage * const workspace )
{
    real d1, d7;
    const real swa = control->nonb_low;
    const real swb = control->nonb_cut;
    real swa2, swa3;
    real swb2, swb3;

    if ( FABS( swa ) > 0.01 )
    {
        fprintf( stderr, "[WARNING] non-zero lower Taper-radius cutoff in force field parameters\n" );
    }

    if ( swb < 0.0 )
    {
        fprintf( stderr, "[ERROR] negative upper Taper-radius cutoff in force field parameters\n" );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }
    else if ( swb < 5.0 )
    {
        fprintf( stderr, "[WARNING] very low Taper-radius cutoff in force field parameters (%f)\n", swb );
    }

    d1 = swb - swa;
    d7 = POW( d1, 7.0 );
    swa2 = SQR( swa );
    swa3 = CUBE( swa );
    swb2 = SQR( swb );
    swb3 = CUBE( swb );

    workspace->Tap[7] =  20.0 / d7;
    workspace->Tap[6] = -70.0 * (swa + swb) / d7;
    workspace->Tap[5] =  84.0 * (swa2 + 3.0 * swa * swb + swb2) / d7;
    workspace->Tap[4] = -35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3 ) / d7;
    workspace->Tap[3] = 140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3 ) / d7;
    workspace->Tap[2] = -210.0 * (swa3 * swb2 + swa2 * swb3) / d7;
    workspace->Tap[1] = 140.0 * swa3 * swb3 / d7;
    workspace->Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2
            + 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb ) / d7;
}


void Init_Workspace( reax_system * const system, control_params * const control,
        storage * const workspace )
{
    Allocate_Workspace( system, control, workspace, system->local_cap,
            system->total_cap );

    workspace->realloc.far_nbrs = FALSE;
    workspace->realloc.cm = FALSE;
    workspace->realloc.hbonds = FALSE;
    workspace->realloc.bonds = FALSE;
    workspace->realloc.thbody = FALSE;
    workspace->realloc.gcell_atoms = 0;

    Reset_Workspace( system, workspace );

    Init_Taper( control, workspace );
}


/* Setup communication data structures
 * */
void Init_MPI_Datatypes( reax_system * const system, storage * const workspace,
        mpi_datatypes * const mpi_data )
{
    int block[11];
    int i;
    MPI_Aint disp[11];
    MPI_Aint base, size_entry;
    MPI_Datatype type[11], temp_type;
    mpi_atom sample[2];
    boundary_atom b_sample[2];
    restart_atom r_sample[2];
    rvec rvec_sample[2];
    rvec2 rvec2_sample[2];

    mpi_data->world = MPI_COMM_WORLD;

    /* mpi_atom */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = MAX_ATOM_NAME_LEN;
    block[6] = 3;
    block[7] = 3;
    block[8] = 3;
    block[9] = 4;
    block[10] = 4;

    MPI_Get_address( sample, disp );
    MPI_Get_address( &sample[0].imprt_id, disp + 1 );
    MPI_Get_address( &sample[0].type, disp + 2 );
    MPI_Get_address( &sample[0].num_bonds, disp + 3 );
    MPI_Get_address( &sample[0].num_hbonds, disp + 4 );
    MPI_Get_address( sample[0].name, disp + 5 );
    MPI_Get_address( sample[0].x, disp + 6 );
    MPI_Get_address( sample[0].v, disp + 7 );
    MPI_Get_address( sample[0].f_old, disp + 8 );
    MPI_Get_address( sample[0].s, disp + 9 );
    MPI_Get_address( sample[0].t, disp + 10 );
    base = disp[0];
    for ( i = 0; i < 11; ++i )
    {
        disp[i] -= base;
    }

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_INT;
    type[3] = MPI_INT;
    type[4] = MPI_INT;
    type[5] = MPI_CHAR;
    type[6] = MPI_DOUBLE;
    type[7] = MPI_DOUBLE;
    type[8] = MPI_DOUBLE;
    type[9] = MPI_DOUBLE;
    type[10] = MPI_DOUBLE;

    MPI_Type_create_struct( 11, block, disp, type, &temp_type );
    /* in case of compiler padding, compute difference
     * between 2 consecutive struct elements */
    MPI_Get_address( sample + 1, &size_entry );
    size_entry = MPI_Aint_diff( size_entry, base );
    MPI_Type_create_resized( temp_type, 0, size_entry, &mpi_data->mpi_atom_type );
    MPI_Type_commit( &mpi_data->mpi_atom_type );

    /* boundary_atom */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = 3;

    MPI_Get_address( b_sample, disp );
    MPI_Get_address( &b_sample[0].imprt_id, disp + 1 );
    MPI_Get_address( &b_sample[0].type, disp + 2 );
    MPI_Get_address( &b_sample[0].num_bonds, disp + 3 );
    MPI_Get_address( &b_sample[0].num_hbonds, disp + 4 );
    MPI_Get_address( b_sample[0].x, disp + 5 );
    base = disp[0];
    for ( i = 0; i < 6; ++i )
    {
        disp[i] -= base;
    }

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_INT;
    type[3] = MPI_INT;
    type[4] = MPI_INT;
    type[5] = MPI_DOUBLE;

    MPI_Type_create_struct( 6, block, disp, type, &temp_type );
    /* in case of compiler padding, compute difference
     * between 2 consecutive struct elements */
    MPI_Get_address( b_sample + 1, &size_entry );
    size_entry = MPI_Aint_diff( size_entry, base );
    MPI_Type_create_resized( temp_type, 0, size_entry, &mpi_data->boundary_atom_type );
    MPI_Type_commit( &mpi_data->boundary_atom_type );

    /* mpi_rvec */
    block[0] = 3;

    MPI_Get_address( &rvec_sample, disp );
    base = disp[0];
    for ( i = 0; i < 1; ++i )
    {
        disp[i] -= base;
    }

    type[0] = MPI_DOUBLE;

    MPI_Type_create_struct( 1, block, disp, type, &temp_type );
    /* in case of compiler padding, compute difference
     * between 2 consecutive struct elements */
    MPI_Get_address( rvec_sample + 1, &size_entry );
    size_entry = MPI_Aint_diff( size_entry, base );
    MPI_Type_create_resized( temp_type, 0, size_entry, &mpi_data->mpi_rvec );
    MPI_Type_commit( &mpi_data->mpi_rvec );

    /* mpi_rvec2 */
    block[0] = 2;

    MPI_Get_address( &rvec2_sample, disp );
    base = disp[0];
    for ( i = 0; i < 1; ++i )
    {
        disp[i] -= base;
    }

    type[0] = MPI_DOUBLE;

    MPI_Type_create_struct( 1, block, disp, type, &temp_type );
    /* in case of compiler padding, compute difference
     * between 2 consecutive struct elements */
    MPI_Get_address( rvec2_sample + 1, &size_entry );
    size_entry = MPI_Aint_diff( size_entry, base );
    MPI_Type_create_resized( temp_type, 0, size_entry, &mpi_data->mpi_rvec2 );
    MPI_Type_commit( &mpi_data->mpi_rvec2 );

    /* restart_atom */
    block[0] = 1;
    block[1] = 1 ;
    block[2] = MAX_ATOM_NAME_LEN;
    block[3] = 3;
    block[4] = 3;

    MPI_Get_address( &r_sample, disp );
    MPI_Get_address( &r_sample[0].type, disp + 1 );
    MPI_Get_address( r_sample[0].name, disp + 2 );
    MPI_Get_address( r_sample[0].x, disp + 3 );
    MPI_Get_address( r_sample[0].v, disp + 4 );
    base = disp[0];
    for ( i = 0; i < 5; ++i )
    {
        disp[i] -= base;
    }

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_CHAR;
    type[3] = MPI_DOUBLE;
    type[4] = MPI_DOUBLE;

    MPI_Type_create_struct( 5, block, disp, type, &temp_type );
    /* in case of compiler padding, compute difference
     * between 2 consecutive struct elements */
    MPI_Get_address( r_sample + 1, &size_entry );
    size_entry = MPI_Aint_diff( size_entry, base );
    MPI_Type_create_resized( temp_type, 0, size_entry, &mpi_data->restart_atom_type );
    MPI_Type_commit( &mpi_data->restart_atom_type );

    mpi_data->in1_buffer = NULL;
    mpi_data->in2_buffer = NULL;
}


/* Allocate and initialize lists
 * */
void Init_Lists( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    int ret;

    Estimate_Num_Neighbors( system );

    Make_List( system->total_cap, system->total_far_nbrs, TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
   printf("Init list max nbrs \n"); 
   Init_List_Indices( lists[FAR_NBRS], system->max_far_nbrs );

    ret = Generate_Neighbor_Lists( system, data, workspace, lists );
    if ( ret != SUCCESS )
    {
        fprintf( stderr, "[ERROR] p%d: failed to generate neighbor lists. Terminating...\n", system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    
    Estimate_Storages( system, control, lists );
    
    Allocate_Matrix( &workspace->H, system->n, system->local_cap, system->total_cm_entries );
    Init_Matrix_Row_Indices( &workspace->H, system->max_cm_entries );

    if ( control->hbond_cut > 0.0 )
    {
        Make_List( system->total_cap, system->total_hbonds, TYP_HBOND, lists[HBONDS] );
        Init_List_Indices( lists[HBONDS], system->max_hbonds );
    }

    Make_List( system->total_cap, system->total_bonds, TYP_BOND, lists[BONDS] );
    Init_List_Indices( lists[BONDS], system->max_bonds );

    Make_List( system->total_bonds, system->total_thbodies, TYP_THREE_BODY, lists[THREE_BODIES] );

#if defined(TEST_FORCES)
    Make_List( system->total_cap, system->total_bonds * 8, TYP_DDELTA, lists[DDELTAS] );
    Make_List( system->total_bonds, system->total_bonds * 50, TYP_DBO, lists[DBOS] );
#endif
}


void Initialize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    Init_MPI_Datatypes( system, workspace, mpi_data );

    exit(0);
    Init_System( system, control, data, workspace, mpi_data );

    Init_Simulation_Data( system, control, data );

    Init_Workspace( system, control, workspace );

    Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    if ( control->tabulate )
    {
        Init_Lookup_Tables( system, control, workspace, mpi_data );
    }

    Init_Force_Functions( control );

#ifdef TEST_FORCES
//    Init_Force_Test_Functions( );
//    fprintf( stderr, "p%d: initialized force test functions\n", system->my_rank );
#endif
}


void Pure_Initialize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    Init_Simulation_Data( system, control, data );

    Init_Workspace( system, control, workspace );

    Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Force_Functions( control );
}



static void Finalize_System( reax_system * const system, control_params * const control,
        simulation_data * const data )
{
    reax_interaction * const reax = &system->reax_param;

    Deallocate_Grid( &system->my_grid );

    sfree( reax->gp.l, "Finalize_System::reax->gp.l" );

    sfree( reax->sbp, "Finalize_System::reax->sbp" );
    sfree( reax->tbp, "Finalize_System::reax->tbp" );
    sfree( reax->thbp, "Finalize_System::reax->thbp" );
    sfree( reax->hbp, "Finalize_System::reax->hbp" );
    sfree( reax->fbp, "Finalize_System::reax->fbp" );

    /*sfree( system->far_nbrs, "Finalize_System::system->far_nbrs" );
    sfree( system->max_far_nbrs, "Finalize_System::system->max_far_nbrs" );
    sfree( system->bonds, "Finalize_System::system->bonds" );
    sfree( system->max_bonds, "Finalize_System::system->max_bonds" );
    sfree( system->hbonds, "Finalize_System::system->hbonds" );
    sfree( system->max_hbonds, "Finalize_System::system->max_hbonds" );
    sfree( system->cm_entries, "Finalize_System::system->cm_entries" );
    sfree( system->max_cm_entries, "Finalize_System::system->max_cm_entries" );
*/
    sfree( system->my_atoms, "Finalize_System::system->atoms" );
}


static void Finalize_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data, output_controls * const out_control )
{
}


static void Finalize_Workspace( reax_system * const system, control_params * const control,
        storage * const workspace )
{
    int i;

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        sfree( workspace->tmp_dbl[i], "Finalize_Workspace::tmp_dbl[i]" );
        sfree( workspace->tmp_rvec[i], "Finalize_Workspace::tmp_rvec[i]" );
        sfree( workspace->tmp_rvec2[i], "Finalize_Workspace::tmp_rvec2[i]" );
    }

    sfree( workspace->within_bond_box, "Finalize_Workspace::skin" );
    sfree( workspace->total_bond_order, "Finalize_Workspace::workspace->total_bond_order" );
    sfree( workspace->Deltap, "Finalize_Workspace::workspace->Deltap" );
    sfree( workspace->Deltap_boc, "Finalize_Workspace::workspace->Deltap_boc" );
    sfree( workspace->dDeltap_self, "Finalize_Workspace::workspace->dDeltap_self" );
    sfree( workspace->Delta, "Finalize_Workspace::workspace->Delta" );
    sfree( workspace->Delta_lp, "Finalize_Workspace::workspace->Delta_lp" );
    sfree( workspace->Delta_lp_temp, "Finalize_Workspace::workspace->Delta_lp_temp" );
    sfree( workspace->dDelta_lp, "Finalize_Workspace::workspace->dDelta_lp" );
    sfree( workspace->dDelta_lp_temp, "Finalize_Workspace::workspace->dDelta_lp_temp" );
    sfree( workspace->Delta_e, "Finalize_Workspace::workspace->Delta_e" );
    sfree( workspace->Delta_boc, "Finalize_Workspace::workspace->Delta_boc" );
    sfree( workspace->nlp, "Finalize_Workspace::workspace->nlp" );
    sfree( workspace->nlp_temp, "Finalize_Workspace::workspace->nlp_temp" );
    sfree( workspace->Clp, "Finalize_Workspace::workspace->Clp" );
    sfree( workspace->CdDelta, "Finalize_Workspace::workspace->CdDelta" );
    sfree( workspace->vlpex, "Finalize_Workspace::workspace->vlpex" );
    sfree( workspace->bond_mark, "Finalize_Workspace::bond_mark" );

    Deallocate_Matrix( &workspace->H );
    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
//        Deallocate_Matrix( &workspace->H_full );
//        Deallocate_Matrix( &workspace->H_spar_patt );
//        Deallocate_Matrix( &workspace->H_spar_patt_full );
//        Deallocate_Matrix( &workspace->H_app_inv );
    }

    if ( control->cm_solver_pre_comp_type == DIAG_PC )
    {
        sfree( workspace->Hdia_inv, "Finalize_Workspace::workspace->Hdia_inv" );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        sfree( workspace->droptol, "Finalize_Workspace::workspace->droptol" );
    }
    sfree( workspace->b_s, "Finalize_Workspace::workspace->b_s" );
    sfree( workspace->b_t, "Finalize_Workspace::workspace->b_t" );
    sfree( workspace->b_prc, "Finalize_Workspace::workspace->b_prc" );
    sfree( workspace->b_prm, "Finalize_Workspace::workspace->b_prm" );
    sfree( workspace->s, "Finalize_Workspace::workspace->s" );
    sfree( workspace->t, "Finalize_Workspace::workspace->t" );

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            sfree( workspace->y, "Finalize_Workspace::workspace->y" );
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->g, "Finalize_Workspace::workspace->g" );
            sfree( workspace->h, "Finalize_Workspace::workspace->h" );
            sfree( workspace->hs, "Finalize_Workspace::workspace->hs" );
            sfree( workspace->hc, "Finalize_Workspace::workspace->hc" );
            sfree( workspace->v, "Finalize_Workspace::workspace->v" );
            break;

        case CG_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->r2, "Finalize_Workspace::workspace->r2" );
            sfree( workspace->d2, "Finalize_Workspace::workspace->d2" );
            sfree( workspace->q2, "Finalize_Workspace::workspace->q2" );
            sfree( workspace->p2, "Finalize_Workspace::workspace->p2" );
            break;

        case SDM_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->r2, "Finalize_Workspace::workspace->r2" );
            sfree( workspace->d2, "Finalize_Workspace::workspace->d2" );
            sfree( workspace->q2, "Finalize_Workspace::workspace->q2" );
            sfree( workspace->p2, "Finalize_Workspace::workspace->p2" );
            break;

        case BiCGStab_S:
        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    /* integrator storage */
    sfree( workspace->v_const, "Finalize_Workspace::workspace->v_const" );

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sfree( workspace->mark, "Finalize_Workspace::workspace->mark" );
        sfree( workspace->old_mark, "Finalize_Workspace::workspace->old_mark" );
    }

    if ( control->diffusion_coef )
    {
        sfree( workspace->x_old, "Finalize_Workspace::workspace->x_old" );
    }

    /* force-related storage */
    sfree( workspace->f, "Finalize_Workspace::workspace->f" );

    /* space for keeping restriction info, if any */
    if ( control->restrict_bonds )
    {
        sfree( workspace->restricted, "Finalize_Workspace::workspace->restricted" );
        sfree( workspace->restricted_list, "Finalize_Workspace::workspace->restricted_list" );
    }

#ifdef TEST_FORCES
    sfree( workspace->dDelta, "Finalize_Workspace::workspace->dDelta" );
    sfree( workspace->f_ele, "Finalize_Workspace::workspace->f_ele" );
    sfree( workspace->f_vdw, "Finalize_Workspace::workspace->f_vdw" );
    sfree( workspace->f_bo, "Finalize_Workspace::workspace->f_bo" );
    sfree( workspace->f_be, "Finalize_Workspace::workspace->f_be" );
    sfree( workspace->f_lp, "Finalize_Workspace::workspace->f_lp" );
    sfree( workspace->f_ov, "Finalize_Workspace::workspace->f_ov" );
    sfree( workspace->f_un, "Finalize_Workspace::workspace->f_un" );
    sfree( workspace->f_ang, "Finalize_Workspace::workspace->f_ang" );
    sfree( workspace->f_coa, "Finalize_Workspace::workspace->f_coa" );
    sfree( workspace->f_pen, "Finalize_Workspace::workspace->f_pen" );
    sfree( workspace->f_hb, "Finalize_Workspace::workspace->f_hb" );
    sfree( workspace->f_tor, "Finalize_Workspace::workspace->f_tor" );
    sfree( workspace->f_con, "Finalize_Workspace::workspace->f_con" );
    sfree( workspace->f_tot, "Finalize_Workspace::workspace->f_tot" );

    sfree( workspace->rcounts, "Finalize_Workspace::workspace->rcounts" );
    sfree( workspace->displs, "Finalize_Workspace::workspace->displs" );
    sfree( workspace->id_all, "Finalize_Workspace::workspace->id_all" );
    sfree( workspace->f_all, "Finalize_Workspace::workspace->f_all" );
#endif
}


static void Finalize_Lists( control_params * const control, reax_list ** const lists )
{
    int i;

    for ( i = 0; i < LIST_N; ++i )
    {
        sfree( lists[i], "Finalize_Lists::lists[i]" );
    }
}


static void Finalize_MPI_Datatypes( mpi_datatypes * const mpi_data )
{
    int ret;

    Deallocate_MPI_Buffers( mpi_data );

    ret = MPI_Type_free( &mpi_data->mpi_atom_type );
    Check_MPI_Error( ret, "Finalize_MPI_Datatypes::mpi_data->mpi_atom_type" );
    ret = MPI_Type_free( &mpi_data->boundary_atom_type );
    Check_MPI_Error( ret, "Finalize_MPI_Datatypes::mpi_data->boundary_atom_type" );
    ret = MPI_Type_free( &mpi_data->mpi_rvec );
    Check_MPI_Error( ret, "Finalize_MPI_Datatypes::mpi_data->mpi_rvec" );
    ret = MPI_Type_free( &mpi_data->mpi_rvec2 );
    Check_MPI_Error( ret, "Finalize_MPI_Datatypes::mpi_data->mpi_rvec2" );
    ret = MPI_Type_free( &mpi_data->restart_atom_type );
    Check_MPI_Error( ret, "Finalize_MPI_Datatypes::mpi_data->restart_atom_type" );
}


/* Deallocate top-level data structures, close file handles, etc.
 *
 */
void Finalize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data,
        const int output_enabled )
{
    if ( control->tabulate )
    {
        Finalize_LR_Lookup_Table( system, control, workspace, mpi_data );
    }

    if ( output_enabled == TRUE )
    {
        Finalize_Output_Files( system, control, out_control );
    }

    Finalize_Lists( control, lists );

//    Finalize_Workspace( system, control, workspace );

    Finalize_Simulation_Data( system, control, data, out_control );

    Finalize_System( system, control, data );

    Finalize_MPI_Datatypes( mpi_data );
}

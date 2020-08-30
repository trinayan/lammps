/*----------------------------------------------------------------------
xfI  PuReMD - Purdue ReaxFF Molecular Dynamics Program

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




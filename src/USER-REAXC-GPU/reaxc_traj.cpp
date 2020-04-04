/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, hmaktulga@lbl.gov
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  Please cite the related publication:
  H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
  "Parallel Reactive Molecular Dynamics: Numerical Methods and
  Algorithmic Techniques", Parallel Computing, in press.

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

#include "reaxc_traj.h"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "reaxc_defs.h"
#include "reaxc_list.h"

#include "error.h"

int Reallocate_Output_Buffer( LAMMPS_NS::Error *error_ptr, output_controls *out_control, int req_space )
{
  if (out_control->buffer_len > 0)
    free( out_control->buffer );

  out_control->buffer_len = (int)(req_space*SAFE_ZONE);
  out_control->buffer = (char*) malloc(out_control->buffer_len*sizeof(char));
  if (out_control->buffer == NULL) {
    char errmsg[256];
    snprintf(errmsg, 256, "Insufficient memory for required buffer size %d", (int) (req_space*SAFE_ZONE));
    error_ptr->one(FLERR,errmsg);
  }

  return SUCCESS;
}


void Write_Skip_Line( output_controls *out_control, mpi_datatypes * /*mpi_data*/,
                      int my_rank, int skip, int num_section )
{
  if (my_rank == MASTER_NODE)
    fprintf( out_control->strj, INT2_LINE,
             "chars_to_skip_section:", skip, num_section );
}


int Write_Header( reax_system *system, control_params *control,
                  output_controls *out_control, mpi_datatypes * /*mpi_data*/ )
{
printf("Write header not impl because of select list\n");	
exit(0);
}


int Write_Init_Desc( reax_system *system, control_params * /*control*/,
                     output_controls *out_control, mpi_datatypes *mpi_data )
{
  int i, me, np, cnt, buffer_len, buffer_req;
  reax_atom *p_atom;
  MPI_Status status;

  me = system->my_rank;
  np = system->wsize;

  /* skip info */
  Write_Skip_Line( out_control, mpi_data, me,
                   system->bigN * INIT_DESC_LEN, system->bigN );

  if (out_control->traj_method == REG_TRAJ && me == MASTER_NODE)
    buffer_req = system->bigN * INIT_DESC_LEN + 1;
  else buffer_req = system->n * INIT_DESC_LEN + 1;

  if (buffer_req > out_control->buffer_len * DANGER_ZONE)
    Reallocate_Output_Buffer( system->error_ptr, out_control, buffer_req );

  out_control->line[0] = 0;
  out_control->buffer[0] = 0;
  for( i = 0; i < system->n; ++i ) {
    p_atom = &( system->my_atoms[i] );
    sprintf( out_control->line, INIT_DESC,
             p_atom->orig_id, p_atom->type, p_atom->name,
             system->reax_param.sbp[ p_atom->type ].mass );
    strncpy( out_control->buffer + i*INIT_DESC_LEN,
             out_control->line, INIT_DESC_LEN+1 );
  }

  if (me != MASTER_NODE) {
    MPI_Send( out_control->buffer, buffer_req-1, MPI_CHAR, MASTER_NODE,
              np * INIT_DESCS + me, mpi_data->world );
  } else {
    buffer_len = system->n * INIT_DESC_LEN;
    for( i = 0; i < np; ++i )
      if (i != MASTER_NODE) {
        MPI_Recv( out_control->buffer + buffer_len, buffer_req - buffer_len,
                  MPI_CHAR, i, np*INIT_DESCS+i, mpi_data->world, &status );
        MPI_Get_count( &status, MPI_CHAR, &cnt );
        buffer_len += cnt;
      }
    out_control->buffer[buffer_len] = 0;
    fprintf( out_control->strj, "%s", out_control->buffer );
  }

  return SUCCESS;
}


int Init_Traj( reax_system *system, control_params *control,
               output_controls *out_control, mpi_datatypes *mpi_data,
               char *msg )
{
  char fname[MAX_STR+8];
  int  atom_line_len[ NR_OPT_ATOM ] = { 0, 0, 0, 0,
                                        ATOM_BASIC_LEN, ATOM_wV_LEN,
                                        ATOM_wF_LEN, ATOM_FULL_LEN };
  int  bond_line_len[ NR_OPT_BOND ] = { 0, BOND_BASIC_LEN, BOND_FULL_LEN };
  int  angle_line_len[ NR_OPT_ANGLE ] = { 0, ANGLE_BASIC_LEN };

  /* generate trajectory name */
  sprintf( fname, "%s.trj", control->sim_name );

  /* how should I write atoms? */
  out_control->atom_line_len = atom_line_len[ out_control->atom_info ];
  out_control->write_atoms = ( out_control->atom_line_len ? 1 : 0 );
  /* bonds? */
  out_control->bond_line_len = bond_line_len[ out_control->bond_info ];
  out_control->write_bonds = ( out_control->bond_line_len ? 1 : 0 );
  /* angles? */
  out_control->angle_line_len = angle_line_len[ out_control->angle_info ];
  out_control->write_angles = ( out_control->angle_line_len ? 1 : 0 );

  /* allocate line & buffer space */
  out_control->line = (char*) calloc( MAX_TRJ_LINE_LEN + 1, sizeof(char) );
  out_control->buffer_len = 0;
  out_control->buffer = NULL;

  /* write trajectory header and atom info, if applicable */
  if (out_control->traj_method == REG_TRAJ) {
    if (system->my_rank == MASTER_NODE)
      out_control->strj = fopen( fname, "w" );
  } else {
    strcpy( msg, "init_traj: unknown trajectory option" );
    return FAILURE;
  }
  Write_Header( system, control, out_control, mpi_data );
  Write_Init_Desc( system, control, out_control, mpi_data );

  return SUCCESS;
}


int Write_Frame_Header( reax_system *system, control_params *control,
                        simulation_data *data, output_controls *out_control,
                        mpi_datatypes * /*mpi_data*/ )
{
  int me, num_frm_hdr_lines, my_frm_hdr_lines, buffer_req;

  me = system->my_rank;
  /* frame header lengths */
  num_frm_hdr_lines = 22;
  my_frm_hdr_lines = num_frm_hdr_lines * ( me == MASTER_NODE );
  buffer_req = my_frm_hdr_lines * HEADER_LINE_LEN;
  if (buffer_req > out_control->buffer_len * DANGER_ZONE)
    Reallocate_Output_Buffer( control->error_ptr, out_control, buffer_req );

  /* only the master node writes into trajectory header */
  if (me == MASTER_NODE) {
    /* clear the contents of line & buffer */
    out_control->line[0] = 0;
    out_control->buffer[0] = 0;

    /* skip info */
    sprintf( out_control->line, INT_LINE, "chars_to_skip_frame_header:",
             (num_frm_hdr_lines - 1) * HEADER_LINE_LEN );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    /* step & time */
    sprintf( out_control->line, INT_LINE, "step:", data->step );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "time_in_ps:",
             data->step * control->dt );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );


    /* box info */
    sprintf( out_control->line, REAL_LINE, "volume:", system->big_box.V );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL3_LINE, "box_dimensions:",
             system->big_box.box_norms[0],
             system->big_box.box_norms[1],
             system->big_box.box_norms[2] );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL3_LINE,
             "coordinate_angles:", 90.0, 90.0, 90.0 );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );


    /* system T and P */
    sprintf( out_control->line, REAL_LINE, "temperature:", data->therm.T );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "pressure:",
             (control->ensemble==iNPT) ?
             data->iso_bar.P : data->flex_bar.P_scalar );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );


    /* energies */
    sprintf( out_control->line, REAL_LINE, "total_energy:",
             data->sys_en.e_tot );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "total_kinetic:",
             data->sys_en.e_kin );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "total_potential:",
             data->sys_en.e_pot );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "bond_energy:",
             data->sys_en.e_bond );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "atom_energy:",
             data->sys_en.e_ov + data->sys_en.e_un  );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "lone_pair_energy:",
             data->sys_en.e_lp  );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "valence_angle_energy:",
             data->sys_en.e_ang + data->sys_en.e_pen );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "3-body_conjugation:",
             data->sys_en.e_coa );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "hydrogen_bond_energy:",
             data->sys_en.e_hb );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "torsion_angle_energy:",
             data->sys_en.e_tor );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "4-body_conjugation:",
             data->sys_en.e_con );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "vdWaals_energy:",
             data->sys_en.e_vdW );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "electrostatics_energy:",
             data->sys_en.e_ele );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );

    sprintf( out_control->line, REAL_LINE, "polarization_energy:",
             data->sys_en.e_pol );
    strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN+1 );
  }

  /* dump out the buffer */
  if (system->my_rank == MASTER_NODE)
    fprintf( out_control->strj, "%s", out_control->buffer );

  return SUCCESS;
}



int Write_Atoms( reax_system *system, control_params * /*control*/,
                 output_controls *out_control, mpi_datatypes *mpi_data )
{
  int i, me, np, line_len, buffer_len, buffer_req, cnt;
  MPI_Status  status;
  reax_atom  *p_atom;

  me = system->my_rank;
  np = system->wsize;
  line_len = out_control->atom_line_len;

  Write_Skip_Line( out_control, mpi_data, me,
                   system->bigN*line_len, system->bigN );

  if (out_control->traj_method == REG_TRAJ && me == MASTER_NODE)
    buffer_req = system->bigN * line_len + 1;
  else buffer_req = system->n * line_len + 1;

  if (buffer_req > out_control->buffer_len * DANGER_ZONE)
    Reallocate_Output_Buffer( system->error_ptr, out_control, buffer_req );

  /* fill in buffer */
  out_control->line[0] = 0;
  out_control->buffer[0] = 0;
  for( i = 0; i < system->n; ++i ) {
    p_atom = &( system->my_atoms[i] );

    switch( out_control->atom_info ) {
    case OPT_ATOM_BASIC:
      sprintf( out_control->line, ATOM_BASIC,
               p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
               p_atom->q );
      break;
    case OPT_ATOM_wF:
      sprintf( out_control->line, ATOM_wF,
               p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
               p_atom->f[0], p_atom->f[1], p_atom->f[2], p_atom->q );
      break;
    case OPT_ATOM_wV:
      sprintf( out_control->line, ATOM_wV,
               p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
               p_atom->v[0], p_atom->v[1], p_atom->v[2], p_atom->q );
      break;
    case OPT_ATOM_FULL:
      sprintf( out_control->line, ATOM_FULL,
               p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
               p_atom->v[0], p_atom->v[1], p_atom->v[2],
               p_atom->f[0], p_atom->f[1], p_atom->f[2], p_atom->q );
      break;
    default:
      system->error_ptr->one(FLERR,"Write_traj_atoms: unknown atom trajectory format");
    }

    strncpy( out_control->buffer + i*line_len, out_control->line, line_len+1 );
  }

  if (me != MASTER_NODE) {
    MPI_Send( out_control->buffer, buffer_req-1, MPI_CHAR, MASTER_NODE,
              np*ATOM_LINES+me, mpi_data->world );
  } else {
    buffer_len = system->n * line_len;
    for( i = 0; i < np; ++i )
      if (i != MASTER_NODE) {
        MPI_Recv( out_control->buffer + buffer_len, buffer_req - buffer_len,
                  MPI_CHAR, i, np*ATOM_LINES+i, mpi_data->world, &status );
        MPI_Get_count( &status, MPI_CHAR, &cnt );
        buffer_len += cnt;
      }
    out_control->buffer[buffer_len] = 0;
    fprintf( out_control->strj, "%s", out_control->buffer );
  }

  return SUCCESS;
}


int Write_Bonds(reax_system *system, control_params *control, reax_list *bonds,
                output_controls *out_control, mpi_datatypes *mpi_data)
{
 printf("Write bonds not impl because of select list");	
 exit(0); 
}


int Write_Angles( reax_system *system, control_params *control,
                  reax_list *bonds, reax_list *thb_intrs,
                  output_controls *out_control, mpi_datatypes *mpi_data )
{
printf("Write angles not impl because of select list");
exit(0);
}


int Append_Frame( reax_system *system, control_params *control,
                  simulation_data *data, reax_list **lists,
                  output_controls *out_control, mpi_datatypes *mpi_data )
{
  Write_Frame_Header( system, control, data, out_control, mpi_data );

  if (out_control->write_atoms)
    Write_Atoms( system, control, out_control, mpi_data );

  if (out_control->write_bonds)
    Write_Bonds( system, control, (*lists + BONDS), out_control, mpi_data );

  if (out_control->write_angles)
    Write_Angles( system, control, (*lists + BONDS), (*lists + THREE_BODIES),
                  out_control, mpi_data );

  return SUCCESS;
}


int End_Traj( int my_rank, output_controls *out_control )
{
  if (my_rank == MASTER_NODE)
    fclose( out_control->strj );

  free( out_control->buffer );
  free( out_control->line );

  return SUCCESS;
}

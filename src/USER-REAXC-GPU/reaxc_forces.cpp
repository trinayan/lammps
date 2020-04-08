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

#include "reaxc_forces.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "reaxc_bond_orders.h"
#include "reaxc_bonds.h"
#include "reaxc_hydrogen_bonds.h"
#include "reaxc_list.h"
#include "reaxc_multi_body.h"
#include "reaxc_nonbonded.h"
#include "reaxc_torsion_angles.h"
#include "reaxc_valence_angles.h"
#include "reaxc_vector.h"

#include "error.h"

interaction_function Interaction_Functions[NUM_INTRS];

void Dummy_Interaction( reax_system * /*system*/, control_params * /*control*/,
                        simulation_data * /*data*/, storage * /*workspace*/,
                        reax_list ** /*lists*/, output_controls * /*out_control*/ )
{
}


void Init_Force_Functions( control_params *control )
{
  Interaction_Functions[0] = BO;
  Interaction_Functions[1] = Bonds; //Dummy_Interaction;
  Interaction_Functions[2] = Atom_Energy; //Dummy_Interaction;
  Interaction_Functions[3] = Valence_Angles; //Dummy_Interaction;
  Interaction_Functions[4] = Torsion_Angles; //Dummy_Interaction;
  if (control->hbond_cut > 0)
    Interaction_Functions[5] = Hydrogen_Bonds;
  else Interaction_Functions[5] = Dummy_Interaction;
  Interaction_Functions[6] = Dummy_Interaction; //empty
  Interaction_Functions[7] = Dummy_Interaction; //empty
  Interaction_Functions[8] = Dummy_Interaction; //empty
  Interaction_Functions[9] = Dummy_Interaction; //empty
}


void Compute_Bonded_Forces( reax_system *system, control_params *control,
                            simulation_data *data, storage *workspace,
                            reax_list **lists, output_controls *out_control,
                            MPI_Comm /*comm*/ )
{
  int i;

  /* Implement all force calls as function pointers */
  for( i = 0; i < NUM_INTRS; i++ ) {
    (Interaction_Functions[i])( system, control, data, workspace,
                                lists, out_control );
  }
}


void Compute_NonBonded_Forces( reax_system *system, control_params *control,
                               simulation_data *data, storage *workspace,
                               reax_list **lists, output_controls *out_control,
                               MPI_Comm /*comm*/ )
{

  /* van der Waals and Coulomb interactions */
  if (control->tabulate == 0)
    vdW_Coulomb_Energy( system, control, data, workspace,
                        lists, out_control );
  else
    Tabulated_vdW_Coulomb_Energy( system, control, data, workspace,
                                  lists, out_control );
}


void Compute_Total_Force( reax_system *system, control_params *control,
                          simulation_data *data, storage *workspace,
                          reax_list **lists, mpi_datatypes * /*mpi_data*/ )
{
int i, pj;
  reax_list *bonds = (*lists) + BONDS;

  for( i = 0; i < system->N; ++i )
    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
      if (i < bonds->select.bond_list[pj].nbr) {
        if (control->virial == 0)
          Add_dBond_to_Forces( system, i, pj, workspace, lists );
        else
          Add_dBond_to_Forces_NPT( i, pj, data, workspace, lists );
      }
 
}

void Validate_Lists( reax_system *system, storage * /*workspace*/, reax_list **lists,
                     int step, int /*n*/, int N, int numH )
{
  int i, comp, Hindex;
  reax_list *bonds, *hbonds;

  double saferzone = system->saferzone;

  /* bond list */
  if (N > 0) {
    bonds = *lists + BONDS;

    for( i = 0; i < N; ++i ) {
      system->my_atoms[i].num_bonds = MAX(Num_Entries(i,bonds)*2, MIN_BONDS);

      if (i < N-1)
        comp = Start_Index(i+1, bonds);
      else comp = bonds->num_intrs;

      if (End_Index(i, bonds) > comp) {
        char errmsg[256];
        snprintf(errmsg, 256, "step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                 step, i, End_Index(i,bonds), comp );
        system->error_ptr->one(FLERR,errmsg);
      }
    }
  }


  /* hbonds list */
  if (numH > 0) {
    hbonds = *lists + HBONDS;

    for( i = 0; i < N; ++i ) {
      Hindex = system->my_atoms[i].Hindex;
      if (Hindex > -1) {
        system->my_atoms[i].num_hbonds =
          (int)(MAX( Num_Entries(Hindex, hbonds)*saferzone, MIN_HBONDS ));

        //if( Num_Entries(i, hbonds) >=
        //(Start_Index(i+1,hbonds)-Start_Index(i,hbonds))*0.90/*DANGER_ZONE*/){
        //  workspace->realloc.hbonds = 1;

        if (Hindex < numH-1)
          comp = Start_Index(Hindex+1, hbonds);
        else comp = hbonds->num_intrs;

        if (End_Index(Hindex, hbonds) > comp) {
          char errmsg[256];
          snprintf(errmsg, 256, "step%d-hbondchk failed: H=%d end(H)=%d str(H+1)=%d\n",
                  step, Hindex, End_Index(Hindex,hbonds), comp );
          system->error_ptr->one(FLERR, errmsg);
        }
      }
    }
  }
}


void Init_Forces_noQEq( reax_system *system, control_params *control,
                        simulation_data *data, storage *workspace,
                        reax_list **lists, output_controls * /*out_control*/ ) 
{
  printf("Not implemented because of twbp\n");
  exit(0);  
}


void Estimate_Storages( reax_system *system, control_params *control,
                        reax_list **lists, int *Htop, int *hb_top,
                        int *bond_top, int *num_3body )
{
 printf("Not implemented because of twbp\n");
 exit(0);
}


void Compute_Forces( reax_system *system, control_params *control,
                     simulation_data *data, storage *workspace,
                     reax_list **lists, output_controls *out_control,
                     mpi_datatypes *mpi_data )
{

  Init_Forces_noQEq( system, control, data, workspace,
                       lists, out_control);

  /********* bonded interactions ************/
  Compute_Bonded_Forces( system, control, data, workspace,
                         lists, out_control, mpi_data->world );

  /********* nonbonded interactions ************/
  Compute_NonBonded_Forces( system, control, data, workspace,
                            lists, out_control, mpi_data->world );

  /*********** total force ***************/
  Compute_Total_Force( system, control, data, workspace, lists, mpi_data );

}

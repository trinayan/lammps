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

#ifndef __NONBONDED_H_
#define __NONBONDED_H_

#include "reaxc_types.h"


#ifdef _cplusplus
extern "C" {
#endif

void vdW_Coulomb_Energy( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls* );

void Tabulated_vdW_Coulomb_Energy( reax_system*, control_params*,
        simulation_data*, storage*, reax_list**, output_controls* );

void LR_vdW_Coulomb( reax_system*, real *, int, int, real, LR_data* );

#ifdef _cplusplus
}
#endif


#endif

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

#include "reaxc_torsion_angles.h"
#include <cmath>
#include "pair.h"
#include "reaxc_defs.h"
#include "reaxc_list.h"
#include "reaxc_vector.h"

#define MIN_SINE 1e-10

double Calculate_Omega( rvec dvec_ij, double r_ij,
                      rvec dvec_jk, double r_jk,
                      rvec dvec_kl, double r_kl,
                      rvec dvec_li, double r_li,
                      three_body_interaction_data *p_ijk,
                      three_body_interaction_data *p_jkl,
                      rvec dcos_omega_di, rvec dcos_omega_dj,
                      rvec dcos_omega_dk, rvec dcos_omega_dl,
                      output_controls * /*out_control*/ )
{
  double unnorm_cos_omega, unnorm_sin_omega, omega;
  double sin_ijk, cos_ijk, sin_jkl, cos_jkl;
  double htra, htrb, htrc, hthd, hthe, hnra, hnrc, hnhd, hnhe;
  double arg, poem, tel;
  rvec cross_jk_kl;

  sin_ijk = sin( p_ijk->theta );
  cos_ijk = cos( p_ijk->theta );
  sin_jkl = sin( p_jkl->theta );
  cos_jkl = cos( p_jkl->theta );

  /* omega */
  unnorm_cos_omega = -rvec_Dot(dvec_ij, dvec_jk) * rvec_Dot(dvec_jk, dvec_kl) +
    SQR( r_jk ) *  rvec_Dot( dvec_ij, dvec_kl );

  rvec_Cross( cross_jk_kl, dvec_jk, dvec_kl );
  unnorm_sin_omega = -r_jk * rvec_Dot( dvec_ij, cross_jk_kl );

  omega = atan2( unnorm_sin_omega, unnorm_cos_omega );

  htra = r_ij + cos_ijk * ( r_kl * cos_jkl - r_jk );
  htrb = r_jk - r_ij * cos_ijk - r_kl * cos_jkl;
  htrc = r_kl + cos_jkl * ( r_ij * cos_ijk - r_jk );
  hthd = r_ij * sin_ijk * ( r_jk - r_kl * cos_jkl );
  hthe = r_kl * sin_jkl * ( r_jk - r_ij * cos_ijk );
  hnra = r_kl * sin_ijk * sin_jkl;
  hnrc = r_ij * sin_ijk * sin_jkl;
  hnhd = r_ij * r_kl * cos_ijk * sin_jkl;
  hnhe = r_ij * r_kl * sin_ijk * cos_jkl;

  poem = 2.0 * r_ij * r_kl * sin_ijk * sin_jkl;
  if (poem < 1e-20) poem = 1e-20;

  tel  = SQR( r_ij ) + SQR( r_jk ) + SQR( r_kl ) - SQR( r_li ) -
    2.0 * ( r_ij * r_jk * cos_ijk - r_ij * r_kl * cos_ijk * cos_jkl +
            r_jk * r_kl * cos_jkl );

  arg  = tel / poem;
  if (arg >  1.0) arg =  1.0;
  if (arg < -1.0) arg = -1.0;

  if (sin_ijk >= 0 && sin_ijk <= MIN_SINE) sin_ijk = MIN_SINE;
  else if( sin_ijk <= 0 && sin_ijk >= -MIN_SINE ) sin_ijk = -MIN_SINE;
  if (sin_jkl >= 0 && sin_jkl <= MIN_SINE) sin_jkl = MIN_SINE;
  else if( sin_jkl <= 0 && sin_jkl >= -MIN_SINE ) sin_jkl = -MIN_SINE;

  // dcos_omega_di
  rvec_ScaledSum( dcos_omega_di, (htra-arg*hnra)/r_ij, dvec_ij, -1., dvec_li );
  rvec_ScaledAdd( dcos_omega_di,-(hthd-arg*hnhd)/sin_ijk, p_ijk->dcos_dk );
  rvec_Scale( dcos_omega_di, 2.0 / poem, dcos_omega_di );

  // dcos_omega_dj
  rvec_ScaledSum( dcos_omega_dj,-(htra-arg*hnra)/r_ij, dvec_ij,
                  -htrb / r_jk, dvec_jk );
  rvec_ScaledAdd( dcos_omega_dj,-(hthd-arg*hnhd)/sin_ijk, p_ijk->dcos_dj );
  rvec_ScaledAdd( dcos_omega_dj,-(hthe-arg*hnhe)/sin_jkl, p_jkl->dcos_di );
  rvec_Scale( dcos_omega_dj, 2.0 / poem, dcos_omega_dj );

  // dcos_omega_dk
  rvec_ScaledSum( dcos_omega_dk,-(htrc-arg*hnrc)/r_kl, dvec_kl,
                  htrb / r_jk, dvec_jk );
  rvec_ScaledAdd( dcos_omega_dk,-(hthd-arg*hnhd)/sin_ijk, p_ijk->dcos_di );
  rvec_ScaledAdd( dcos_omega_dk,-(hthe-arg*hnhe)/sin_jkl, p_jkl->dcos_dj );
  rvec_Scale( dcos_omega_dk, 2.0 / poem, dcos_omega_dk );

  // dcos_omega_dl
  rvec_ScaledSum( dcos_omega_dl, (htrc-arg*hnrc)/r_kl, dvec_kl, 1., dvec_li );
  rvec_ScaledAdd( dcos_omega_dl,-(hthe-arg*hnhe)/sin_jkl, p_jkl->dcos_dk );
  rvec_Scale( dcos_omega_dl, 2.0 / poem, dcos_omega_dl );

  return omega;
}



void Torsion_Angles( reax_system *system, control_params *control,
                     simulation_data *data, storage *workspace,
                     reax_list **lists, output_controls *out_control )
{
 printf("Not impl because of twbp\n");
 exit(0);
  // j loop
}

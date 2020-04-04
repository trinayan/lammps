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

#include "reaxc_valence_angles.h"
#include <cmath>
#include "pair.h"
#include "reaxc_defs.h"
#include "reaxc_list.h"
#include "reaxc_vector.h"

#include "error.h"

static double Dot( double* v1, double* v2, int k )
{
  double ret = 0.0;

  for( int i=0; i < k; ++i )
    ret +=  v1[i] * v2[i];

  return ret;
}

void Calculate_Theta( rvec dvec_ji, double d_ji, rvec dvec_jk, double d_jk,
                      double *theta, double *cos_theta )
{
  (*cos_theta) = Dot( dvec_ji, dvec_jk, 3 ) / ( d_ji * d_jk );
  if (*cos_theta > 1.) *cos_theta  = 1.0;
  if (*cos_theta < -1.) *cos_theta  = -1.0;

  (*theta) = acos( *cos_theta );
}

void Calculate_dCos_Theta( rvec dvec_ji, double d_ji, rvec dvec_jk, double d_jk,
                           rvec* dcos_theta_di,
                           rvec* dcos_theta_dj,
                           rvec* dcos_theta_dk )
{
  int t;
  double sqr_d_ji = SQR(d_ji);
  double sqr_d_jk = SQR(d_jk);
  double inv_dists = 1.0 / (d_ji * d_jk);
  double inv_dists3 = pow( inv_dists, 3.0 );
  double dot_dvecs = Dot( dvec_ji, dvec_jk, 3 );
  double Cdot_inv3 = dot_dvecs * inv_dists3;

  for( t = 0; t < 3; ++t ) {
    (*dcos_theta_di)[t] = dvec_jk[t] * inv_dists -
      Cdot_inv3 * sqr_d_jk * dvec_ji[t];
    (*dcos_theta_dj)[t] = -(dvec_jk[t] + dvec_ji[t]) * inv_dists +
      Cdot_inv3 * ( sqr_d_jk * dvec_ji[t] + sqr_d_ji * dvec_jk[t] );
    (*dcos_theta_dk)[t] = dvec_ji[t] * inv_dists -
      Cdot_inv3 * sqr_d_ji * dvec_jk[t];
  }
}


void Valence_Angles( reax_system *system, control_params *control,
                     simulation_data *data, storage *workspace,
                     reax_list **lists, output_controls * /*out_control*/ )
{
printf("Not impl because of twbp\n");
exit(0);
}

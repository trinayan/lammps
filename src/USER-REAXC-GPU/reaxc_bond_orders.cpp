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

#include "reaxc_bond_orders.h"
#include <cmath>
#include "pair.h"
#include "reaxc_defs.h"
#include "reaxc_types.h"
#include "reaxc_list.h"
#include "reaxc_vector.h"

void Add_dBond_to_Forces_NPT( int i, int pj, simulation_data *data,
                              storage *workspace, reax_list **lists )
{
  reax_list *bonds = (*lists) + BONDS;
  bond_data *nbr_j, *nbr_k;
  bond_order_data *bo_ij, *bo_ji;
  dbond_coefficients coef;
  rvec temp, ext_press;
  ivec rel_box;
  int pk, k, j;

  /* Initializations */
  nbr_j = &(bonds->select.bond_list[pj]);
  j = nbr_j->nbr;
  bo_ij = &(nbr_j->bo_data);
  bo_ji = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);

  coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
  coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
  coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

  coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
  coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
  coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
  coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

  coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
  coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
  coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
  coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

  coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
  coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
  coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);

  for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
    nbr_k = &(bonds->select.bond_list[pk]);
    k = nbr_k->nbr;

    rvec_Scale(temp, -coef.C2dbo, nbr_k->bo_data.dBOp);       /*2nd, dBO*/
    rvec_ScaledAdd(temp, -coef.C2dDelta, nbr_k->bo_data.dBOp);/*dDelta*/
    rvec_ScaledAdd(temp, -coef.C3dbopi, nbr_k->bo_data.dBOp); /*3rd, dBOpi*/
    rvec_ScaledAdd(temp, -coef.C3dbopi2, nbr_k->bo_data.dBOp);/*3rd, dBOpi2*/

    /* force */
    rvec_Add( workspace->f[k], temp );
    /* pressure */
    rvec_iMultiply( ext_press, nbr_k->rel_box, temp );
    rvec_Add( data->my_ext_press, ext_press );

  }

  /* then atom i itself  */
  rvec_Scale( temp, coef.C1dbo, bo_ij->dBOp );                      /*1st,dBO*/
  rvec_ScaledAdd( temp, coef.C2dbo, workspace->dDeltap_self[i] );   /*2nd,dBO*/
  rvec_ScaledAdd( temp, coef.C1dDelta, bo_ij->dBOp );               /*1st,dBO*/
  rvec_ScaledAdd( temp, coef.C2dDelta, workspace->dDeltap_self[i] );/*2nd,dBO*/
  rvec_ScaledAdd( temp, coef.C1dbopi, bo_ij->dln_BOp_pi );        /*1st,dBOpi*/
  rvec_ScaledAdd( temp, coef.C2dbopi, bo_ij->dBOp );              /*2nd,dBOpi*/
  rvec_ScaledAdd( temp, coef.C3dbopi, workspace->dDeltap_self[i]);/*3rd,dBOpi*/

  rvec_ScaledAdd( temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2 );  /*1st,dBO_pi2*/
  rvec_ScaledAdd( temp, coef.C2dbopi2, bo_ij->dBOp );         /*2nd,dBO_pi2*/
  rvec_ScaledAdd( temp, coef.C3dbopi2, workspace->dDeltap_self[i] );/*3rd*/

  /* force */
  rvec_Add( workspace->f[i], temp );

  for( pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk ) {
    nbr_k = &(bonds->select.bond_list[pk]);
    k = nbr_k->nbr;

    rvec_Scale( temp, -coef.C3dbo, nbr_k->bo_data.dBOp );      /*3rd,dBO*/
    rvec_ScaledAdd( temp, -coef.C3dDelta, nbr_k->bo_data.dBOp);/*dDelta*/
    rvec_ScaledAdd( temp, -coef.C4dbopi, nbr_k->bo_data.dBOp); /*4th,dBOpi*/
    rvec_ScaledAdd( temp, -coef.C4dbopi2, nbr_k->bo_data.dBOp);/*4th,dBOpi2*/

    /* force */
    rvec_Add( workspace->f[k], temp );
    /* pressure */
    if (k != i) {
      ivec_Sum( rel_box, nbr_k->rel_box, nbr_j->rel_box ); //rel_box(k, i)
      rvec_iMultiply( ext_press, rel_box, temp );
      rvec_Add( data->my_ext_press, ext_press );

    }
  }

  /* then atom j itself */
  rvec_Scale( temp, -coef.C1dbo, bo_ij->dBOp );                    /*1st, dBO*/
  rvec_ScaledAdd( temp, coef.C3dbo, workspace->dDeltap_self[j] );  /*2nd, dBO*/
  rvec_ScaledAdd( temp, -coef.C1dDelta, bo_ij->dBOp );             /*1st, dBO*/
  rvec_ScaledAdd( temp, coef.C3dDelta, workspace->dDeltap_self[j]);/*2nd, dBO*/

  rvec_ScaledAdd( temp, -coef.C1dbopi, bo_ij->dln_BOp_pi );       /*1st,dBOpi*/
  rvec_ScaledAdd( temp, -coef.C2dbopi, bo_ij->dBOp );             /*2nd,dBOpi*/
  rvec_ScaledAdd( temp, coef.C4dbopi, workspace->dDeltap_self[j]);/*3rd,dBOpi*/

  rvec_ScaledAdd( temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );    /*1st,dBOpi2*/
  rvec_ScaledAdd( temp, -coef.C2dbopi2, bo_ij->dBOp );           /*2nd,dBOpi2*/
  rvec_ScaledAdd( temp,coef.C4dbopi2,workspace->dDeltap_self[j]);/*3rd,dBOpi2*/

  /* force */
  rvec_Add( workspace->f[j], temp );
  /* pressure */
  rvec_iMultiply( ext_press, nbr_j->rel_box, temp );
  rvec_Add( data->my_ext_press, ext_press );

}

void Add_dBond_to_Forces( reax_system *system, int i, int pj,
                          storage *workspace, reax_list **lists )
{
  reax_list *bonds = (*lists) + BONDS;
  bond_data *nbr_j, *nbr_k;
  bond_order_data *bo_ij, *bo_ji;
  dbond_coefficients coef;
  int pk, k, j;

  /* Virial Tallying variables */
  rvec fi_tmp, fj_tmp, fk_tmp, delij, delji, delki, delkj, temp;

  /* Initializations */
  nbr_j = &(bonds->select.bond_list[pj]);
  j = nbr_j->nbr;
  bo_ij = &(nbr_j->bo_data);
  bo_ji = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);

  coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
  coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
  coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

  coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
  coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
  coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
  coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

  coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
  coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
  coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
  coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

  coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
  coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
  coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);

  // forces on i
  rvec_Scale(           temp, coef.C1dbo,    bo_ij->dBOp );
  rvec_ScaledAdd( temp, coef.C2dbo,    workspace->dDeltap_self[i] );
  rvec_ScaledAdd( temp, coef.C1dDelta, bo_ij->dBOp );
  rvec_ScaledAdd( temp, coef.C2dDelta, workspace->dDeltap_self[i] );
  rvec_ScaledAdd( temp, coef.C1dbopi,  bo_ij->dln_BOp_pi );
  rvec_ScaledAdd( temp, coef.C2dbopi,  bo_ij->dBOp );
  rvec_ScaledAdd( temp, coef.C3dbopi,  workspace->dDeltap_self[i]);
  rvec_ScaledAdd( temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
  rvec_ScaledAdd( temp, coef.C2dbopi2, bo_ij->dBOp );
  rvec_ScaledAdd( temp, coef.C3dbopi2, workspace->dDeltap_self[i] );
  rvec_Add( workspace->f[i], temp );

  if (system->pair_ptr->vflag_atom) {
    rvec_Scale(fi_tmp, -1.0, temp);
    rvec_ScaledSum( delij, 1., system->my_atoms[i].x,-1., system->my_atoms[j].x );
    system->pair_ptr->v_tally(i,fi_tmp,delij);
  }

  // forces on j
  rvec_Scale(           temp, -coef.C1dbo,    bo_ij->dBOp );
  rvec_ScaledAdd( temp,  coef.C3dbo,    workspace->dDeltap_self[j] );
  rvec_ScaledAdd( temp, -coef.C1dDelta, bo_ij->dBOp );
  rvec_ScaledAdd( temp,  coef.C3dDelta, workspace->dDeltap_self[j]);
  rvec_ScaledAdd( temp, -coef.C1dbopi,  bo_ij->dln_BOp_pi );
  rvec_ScaledAdd( temp, -coef.C2dbopi,  bo_ij->dBOp );
  rvec_ScaledAdd( temp,  coef.C4dbopi,  workspace->dDeltap_self[j]);
  rvec_ScaledAdd( temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
  rvec_ScaledAdd( temp, -coef.C2dbopi2, bo_ij->dBOp );
  rvec_ScaledAdd( temp,  coef.C4dbopi2, workspace->dDeltap_self[j]);
  rvec_Add( workspace->f[j], temp );

  if (system->pair_ptr->vflag_atom) {
    rvec_Scale(fj_tmp, -1.0, temp);
    rvec_ScaledSum( delji, 1., system->my_atoms[j].x,-1., system->my_atoms[i].x );
    system->pair_ptr->v_tally(j,fj_tmp,delji);
  }

  // forces on k: i neighbor
  for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
    nbr_k = &(bonds->select.bond_list[pk]);
    k = nbr_k->nbr;

    rvec_Scale(     temp, -coef.C2dbo,    nbr_k->bo_data.dBOp);
    rvec_ScaledAdd( temp, -coef.C2dDelta, nbr_k->bo_data.dBOp);
    rvec_ScaledAdd( temp, -coef.C3dbopi,  nbr_k->bo_data.dBOp);
    rvec_ScaledAdd( temp, -coef.C3dbopi2, nbr_k->bo_data.dBOp);
    rvec_Add( workspace->f[k], temp );

    if (system->pair_ptr->vflag_atom) {
      rvec_Scale(fk_tmp, -1.0, temp);
      rvec_ScaledSum(delki,1.,system->my_atoms[k].x,-1.,system->my_atoms[i].x);
      system->pair_ptr->v_tally(k,fk_tmp,delki);
      rvec_ScaledSum(delkj,1.,system->my_atoms[k].x,-1.,system->my_atoms[j].x);
      system->pair_ptr->v_tally(k,fk_tmp,delkj);
    }
  }

  // forces on k: j neighbor
  for( pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk ) {
    nbr_k = &(bonds->select.bond_list[pk]);
    k = nbr_k->nbr;

    rvec_Scale(     temp, -coef.C3dbo,    nbr_k->bo_data.dBOp );
    rvec_ScaledAdd( temp, -coef.C3dDelta, nbr_k->bo_data.dBOp);
    rvec_ScaledAdd( temp, -coef.C4dbopi,  nbr_k->bo_data.dBOp);
    rvec_ScaledAdd( temp, -coef.C4dbopi2, nbr_k->bo_data.dBOp);
    rvec_Add( workspace->f[k], temp );

    if (system->pair_ptr->vflag_atom) {
      rvec_Scale(fk_tmp, -1.0, temp);
      rvec_ScaledSum(delki,1.,system->my_atoms[k].x,-1.,system->my_atoms[i].x);
      system->pair_ptr->v_tally(k,fk_tmp,delki);
      rvec_ScaledSum(delkj,1.,system->my_atoms[k].x,-1.,system->my_atoms[j].x);
      system->pair_ptr->v_tally(k,fk_tmp,delkj);
    }
  }

}


int BOp( storage *workspace, reax_list *bonds, double bo_cut,
         int i, int btop_i, far_neighbor_data *nbr_pj,
         single_body_parameters *sbp_i, single_body_parameters *sbp_j,
         two_body_parameters *twbp ) {
  int j, btop_j;
  double r2, C12, C34, C56;
  double Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
  double BO, BO_s, BO_pi, BO_pi2;
  bond_data *ibond, *jbond;
  bond_order_data *bo_ij, *bo_ji;

  j = nbr_pj->nbr;
  r2 = SQR(nbr_pj->d);

  if (sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
    C12 = twbp->p_bo1 * pow( nbr_pj->d / twbp->r_s, twbp->p_bo2 );
    BO_s = (1.0 + bo_cut) * exp( C12 );
  }
  else BO_s = C12 = 0.0;

  if (sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
    C34 = twbp->p_bo3 * pow( nbr_pj->d / twbp->r_p, twbp->p_bo4 );
    BO_pi = exp( C34 );
  }
  else BO_pi = C34 = 0.0;

  if (sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
    C56 = twbp->p_bo5 * pow( nbr_pj->d / twbp->r_pp, twbp->p_bo6 );
    BO_pi2= exp( C56 );
  }
  else BO_pi2 = C56 = 0.0;

  /* Initially BO values are the uncorrected ones, page 1 */
  BO = BO_s + BO_pi + BO_pi2;

  if (BO >= bo_cut) {
    /****** bonds i-j and j-i ******/
    ibond = &( bonds->select.bond_list[btop_i] );
    btop_j = End_Index( j, bonds );
    jbond = &(bonds->select.bond_list[btop_j]);

    ibond->nbr = j;
    jbond->nbr = i;
    ibond->d = nbr_pj->d;
    jbond->d = nbr_pj->d;
    rvec_Copy( ibond->dvec, nbr_pj->dvec );
    rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
    ivec_Copy( ibond->rel_box, nbr_pj->rel_box );
    ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );
    ibond->dbond_index = btop_i;
    jbond->dbond_index = btop_i;
    ibond->sym_index = btop_j;
    jbond->sym_index = btop_i;
    Set_End_Index( j, btop_j+1, bonds );

    bo_ij = &( ibond->bo_data );
    bo_ji = &( jbond->bo_data );
    bo_ji->BO = bo_ij->BO = BO;
    bo_ji->BO_s = bo_ij->BO_s = BO_s;
    bo_ji->BO_pi = bo_ij->BO_pi = BO_pi;
    bo_ji->BO_pi2 = bo_ij->BO_pi2 = BO_pi2;

    /* Bond Order page2-3, derivative of total bond order prime */
    Cln_BOp_s = twbp->p_bo2 * C12 / r2;
    Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
    Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

    /* Only dln_BOp_xx wrt. dr_i is stored here, note that
       dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
    rvec_Scale(bo_ij->dln_BOp_s,-bo_ij->BO_s*Cln_BOp_s,ibond->dvec);
    rvec_Scale(bo_ij->dln_BOp_pi,-bo_ij->BO_pi*Cln_BOp_pi,ibond->dvec);
    rvec_Scale(bo_ij->dln_BOp_pi2,
               -bo_ij->BO_pi2*Cln_BOp_pi2,ibond->dvec);
    rvec_Scale(bo_ji->dln_BOp_s, -1., bo_ij->dln_BOp_s);
    rvec_Scale(bo_ji->dln_BOp_pi, -1., bo_ij->dln_BOp_pi );
    rvec_Scale(bo_ji->dln_BOp_pi2, -1., bo_ij->dln_BOp_pi2 );

    rvec_Scale( bo_ij->dBOp,
                -(bo_ij->BO_s * Cln_BOp_s +
                  bo_ij->BO_pi * Cln_BOp_pi +
                  bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );
    rvec_Scale( bo_ji->dBOp, -1., bo_ij->dBOp );

    rvec_Add( workspace->dDeltap_self[i], bo_ij->dBOp );
    rvec_Add( workspace->dDeltap_self[j], bo_ji->dBOp );

    bo_ij->BO_s -= bo_cut;
    bo_ij->BO -= bo_cut;
    bo_ji->BO_s -= bo_cut;
    bo_ji->BO -= bo_cut;
    workspace->total_bond_order[i] += bo_ij->BO; //currently total_BOp
    workspace->total_bond_order[j] += bo_ji->BO; //currently total_BOp
    bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;
    bo_ji->Cdbo = bo_ji->Cdbopi = bo_ji->Cdbopi2 = 0.0;

    return 1;
  }

  return 0;
}


void BO( reax_system *system, control_params * /*control*/, simulation_data * /*data*/,
         storage *workspace, reax_list **lists, output_controls * /*out_control*/ )
{
	printf("Bond orders \n");
	exit(0);
}

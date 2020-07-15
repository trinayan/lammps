/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)

     Hybrid and sub-group capabilities: Ray Shan (Sandia)
------------------------------------------------------------------------- */

#include "fix_qeq_reax.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "pair_reaxc_gpu.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "pair.h"
#include "respa.h"
#include "memory.h"
#include "citeme.h"
#include "error.h"
#include "reaxc_defs.h"
#include "reaxc_types.h"


extern "C" void  CudaAllocateStorageForFixQeq(int nmax, int dual_enabled, fix_qeq_gpu *qeq_gpu);
extern "C" void  CudaAllocateMatrixForFixQeq(fix_qeq_gpu *qeq_gpu,int n_cap, int m_cap);
extern "C" void  CudaInitStorageForFixQeq(fix_qeq_gpu *qeq_gpu,double *Hdia_inv, double *b_s,double *b_t,double *b_prc,double *b_prm,double *s,double *t, int NN);
extern "C" void  Cuda_Calculate_H_Matrix(reax_list **gpu_lists,  reax_system *system,fix_qeq_gpu *qeq_gpu, control_params *control, int inum, int SMALL);
extern "C" void  Cuda_Init_Taper(fix_qeq_gpu *qeq_gpu, double *Tap, int numTap);
extern "C" void  Cuda_Allocate_Matrix( sparse_matrix *, int, int );
extern "C" void  Cuda_Init_Sparse_Matrix_Indices( reax_system *system,fix_qeq_gpu *qeq_gpu, int n);
extern "C" void  Cuda_Init_Fix_Atoms(reax_system *system,fix_qeq_gpu *qeq_gpu);
extern "C" void  Cuda_Init_Matvec_Fix(int nn, fix_qeq_gpu *qeq_gpu, reax_system *system);
extern "C" void  Cuda_Copy_Pertype_Parameters_To_Device(double *chi,double *eta,double *gamma,int ntypes,fix_qeq_gpu *qeq_gpu);
extern "C" void Cuda_Copy_From_Device_Comm_Fix(double *buf, double *x, int n, int offset);
extern "C" void  Cuda_Copy_To_Device_Comm_Fix(double *buf,double *x,int n,int offset);
extern "C" void  Cuda_Sparse_Matvec_Compute(sparse_matrix *H,double *x, double *q, double *eta, reax_atom *d_fix_my_atoms, int nn, int NN);
extern "C" void  Cuda_Vector_Sum_Fix( real *res, real a, real *x, real b, real *y, int count);
extern "C" void  Cuda_CG_Preconditioner_Fix( real *, real *, real *, int );
extern "C" void  Cuda_Copy_Vector_From_Device(real *host_vector, real *device_vector, int nn);
extern "C" float  Cuda_Calculate_Local_S_Sum(int nn,fix_qeq_gpu *qeq_gpu);
extern "C" float  Cuda_Calculate_Local_T_Sum(int nn,fix_qeq_gpu *qeq_gpu);
extern "C" void  Cuda_UpdateQ_And_Copy_To_Device_Comm_Fix(double *buf,fix_qeq_gpu *qeq_gpu,int n);
extern "C" void Cuda_Estimate_CMEntries_Storages( reax_system *system, control_params *control, reax_list **lists, fix_qeq_gpu *qeq_gpu,int nn);
extern "C" void  Cuda_Update_Q_And_Backup_ST(int nn, fix_qeq_gpu *qeq_gpu, double u);
extern "C" void  CudaFreeFixQeqParams(fix_qeq_gpu *qeq_gpu);
extern "C" void  CudaFreeHMatrix(fix_qeq_gpu *qeq_gpu);



using namespace LAMMPS_NS;
using namespace FixConst;

#define EV_TO_KCAL_PER_MOL 14.4
//#define DANGER_ZONE     0.95
//#define LOOSE_ZONE      0.7
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
//#define MIN_NBRS 100

static const char cite_fix_qeq_reax[] =
		"fix qeq/reax command:\n\n"
		"@Article{Aktulga12,\n"
		" author = {H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama},\n"
		" title = {Parallel reactive molecular dynamics: Numerical methods and algorithmic techniques},\n"
		" journal = {Parallel Computing},\n"
		" year =    2012,\n"
		" volume =  38,\n"
		" pages =   {245--259}\n"
		"}\n\n";

/* ---------------------------------------------------------------------- */

FixQEqReax::FixQEqReax(LAMMPS *lmp, int narg, char **arg) :
																																																																												  Fix(lmp, narg, arg), pertype_option(NULL)
{
	if (lmp->citeme) lmp->citeme->add(cite_fix_qeq_reax);

	if (narg<8 || narg>9) error->all(FLERR,"Illegal fix qeq/reax command");

	nevery = force->inumeric(FLERR,arg[3]);
	if (nevery <= 0) error->all(FLERR,"Illegal fix qeq/reax command");

	swa = force->numeric(FLERR,arg[4]);
	swb = force->numeric(FLERR,arg[5]);
	tolerance = force->numeric(FLERR,arg[6]);
	int len = strlen(arg[7]) + 1;
	pertype_option = new char[len];
	strcpy(pertype_option,arg[7]);

	// dual CG support only available for USER-OMP variant
	// check for compatibility is in Fix::post_constructor()
	dual_enabled = 0;
	if (narg == 9) {
		if (strcmp(arg[8],"dual") == 0) dual_enabled = 1;
		else error->all(FLERR,"Illegal fix qeq/reax command");
	}
	shld = NULL;

	n = n_cap = 0;
	N = nmax = 0;
	m_fill = m_cap = 0;
	pack_flag = 0;
	s = NULL;
	t = NULL;
	nprev = 4;

	Hdia_inv = NULL;
	b_s = NULL;
	b_t = NULL;
	b_prc = NULL;
	b_prm = NULL;

	// CG
	p = NULL;
	q = NULL;
	r = NULL;
	d = NULL;

	// H matrix
	H.firstnbr = NULL;
	H.numnbrs = NULL;
	H.jlist = NULL;
	H.val = NULL;

	// dual CG support
	// Update comm sizes for this fix
	if (dual_enabled) comm_forward = comm_reverse = 2;
	else comm_forward = comm_reverse = 1;

	// perform initial allocation of atom-based arrays
	// register with Atom class

	reaxc = NULL;
	reaxc = (PairReaxCGPU *) force->pair_match("^reax/c/gpu",0);

	s_hist = t_hist = NULL;
	grow_arrays(atom->nmax);
	atom->add_callback(0);
	for (int i = 0; i < atom->nmax; i++)
		for (int j = 0; j < nprev; ++j)
			s_hist[i][j] = t_hist[i][j] = 0;


	qeq_gpu = (fix_qeq_gpu *)memory->smalloc(sizeof(fix_qeq_gpu),"reax:storage");


}

/* ---------------------------------------------------------------------- */

FixQEqReax::~FixQEqReax()
{
	if (copymode) return;

	delete[] pertype_option;

	// unregister callbacks to this fix from Atom class

	atom->delete_callback(id,0);

	memory->destroy(s_hist);
	memory->destroy(t_hist);

	deallocate_storage();
	deallocate_matrix();

	memory->destroy(shld);

	if (!reaxflag) {
		memory->destroy(chi);
		memory->destroy(eta);
		memory->destroy(gamma);
	}
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::post_constructor()
{
	pertype_parameters(pertype_option);
	if (dual_enabled)
		error->all(FLERR,"Dual keyword only supported with fix qeq/reax/omp");
}

/* ---------------------------------------------------------------------- */

int FixQEqReax::setmask()
{
	//TB:: Understand what this does
	int mask = 0;
	mask |= PRE_FORCE;
	mask |= PRE_FORCE_RESPA;
	mask |= MIN_PRE_FORCE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::pertype_parameters(char *arg)
{
	if (strcmp(arg,"reax/c/gpu") == 0) {
		reaxflag = 1;
		Pair *pair = force->pair_match("reax/c/gpu",0);
		if (pair == NULL) error->all(FLERR,"No pair reax/c/gpu for fix qeq/reax");

		int tmp;
		chi = (double *) pair->extract("chi",tmp);
		eta = (double *) pair->extract("eta",tmp);
		gamma = (double *) pair->extract("gamma",tmp);
		if (chi == NULL || eta == NULL || gamma == NULL)
			error->all(FLERR,
					"Fix qeq/reax could not extract params from pair reax/c/gpu");

		Cuda_Copy_Pertype_Parameters_To_Device(chi,eta,gamma,atom->ntypes,qeq_gpu);
		//TB: SBP parameters updated in Sync_System for GPU
		return;
	}

	//TB:: Not sure why the below part exists and if it will ever be called
	int i,itype,ntypes,rv;
	double v1,v2,v3;
	FILE *pf;

	reaxflag = 0;
	ntypes = atom->ntypes;

	printf("Pertype parameters\n");
	memory->create(chi,ntypes+1,"qeq/reax:chi");
	memory->create(eta,ntypes+1,"qeq/reax:eta");
	memory->create(gamma,ntypes+1,"qeq/reax:gamma");

	if (comm->me == 0) {
		if ((pf = fopen(arg,"r")) == NULL)
			error->one(FLERR,"Fix qeq/reax parameter file could not be found");

		for (i = 1; i <= ntypes && !feof(pf); i++) {
			rv = fscanf(pf,"%d %lg %lg %lg",&itype,&v1,&v2,&v3);
			if (rv != 4)
				error->one(FLERR,"Fix qeq/reax: Incorrect format of param file");
			if (itype < 1 || itype > ntypes)
				error->one(FLERR,"Fix qeq/reax: invalid atom type in param file");
			chi[itype] = v1;
			eta[itype] = v2;
			gamma[itype] = v3;
		}
		if (i <= ntypes) error->one(FLERR,"Invalid param file for fix qeq/reax");
		fclose(pf);
	}

	MPI_Bcast(&chi[1],ntypes,MPI_DOUBLE,0,world);
	MPI_Bcast(&eta[1],ntypes,MPI_DOUBLE,0,world);
	MPI_Bcast(&gamma[1],ntypes,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::allocate_storage()
{
	nmax = atom->nmax;

	memory->create(s,nmax,"qeq:s");
	memory->create(t,nmax,"qeq:t");

	memory->create(Hdia_inv,nmax,"qeq:Hdia_inv");
	memory->create(b_s,nmax,"qeq:b_s");
	memory->create(b_t,nmax,"qeq:b_t");
	memory->create(b_prc,nmax,"qeq:b_prc");
	memory->create(b_prm,nmax,"qeq:b_prm");

	// dual CG support
	int size = nmax;
	if (dual_enabled) size*= 2;


	memory->create(p,size,"qeq:p");
	memory->create(q,size,"qeq:q");
	memory->create(r,size,"qeq:r");
	memory->create(d,size,"qeq:d");

	CudaAllocateStorageForFixQeq(nmax, dual_enabled, qeq_gpu);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::deallocate_storage()
{
	memory->destroy(s);
	memory->destroy(t);

	memory->destroy( Hdia_inv );
	memory->destroy( b_s );
	memory->destroy( b_t );
	memory->destroy( b_prc );
	memory->destroy( b_prm );

	memory->destroy( p );
	memory->destroy( q );
	memory->destroy( r );
	memory->destroy( d );

	CudaFreeFixQeqParams(qeq_gpu);

}

/* ---------------------------------------------------------------------- */

void FixQEqReax::reallocate_storage()
{
	deallocate_storage();
	allocate_storage();
	init_storage();
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::allocate_matrix()
{
	int i,ii,inum,m;
	int *ilist, *numneigh;

	int mincap;
	double safezone;

	if (reaxflag) {
		mincap = reaxc->system->mincap;
		safezone = reaxc->system->safezone;
	} else {
		mincap = MIN_CAP;
		safezone = SAFE_ZONE;
	}

	n = atom->nlocal;
	n_cap = MAX( (int)(n * safezone), mincap);

	// determine the total space for the H matrix

	if (reaxc) {
		inum = reaxc->list->inum;
		ilist = reaxc->list->ilist;
		numneigh = reaxc->list->numneigh;
	} else {
		inum = list->inum;
		ilist = list->ilist;
		numneigh = list->numneigh;
	}

	m = 0;
	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		m += numneigh[i];
	}
	m_cap = MAX( (int)(m * safezone), mincap * MIN_NBRS);

	H.n = n_cap;
	H.m = m_cap;
	memory->create(H.firstnbr,n_cap,"qeq:H.firstnbr");
	memory->create(H.numnbrs,n_cap,"qeq:H.numnbrs");
	memory->create(H.jlist,m_cap,"qeq:H.jlist");
	memory->create(H.val,m_cap,"qeq:H.val");


	CudaAllocateMatrixForFixQeq(qeq_gpu, n_cap, m_cap);

}

/* ---------------------------------------------------------------------- */

void FixQEqReax::deallocate_matrix()
{
	memory->destroy( H.firstnbr );
	memory->destroy( H.numnbrs );
	memory->destroy( H.jlist );
	memory->destroy( H.val );

	CudaFreeHMatrix(qeq_gpu);

	//TB:: Implement deallocation of device side H matrix
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::reallocate_matrix()
{
	deallocate_matrix();
	allocate_matrix();
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::init()
{
	if (!atom->q_flag)
		error->all(FLERR,"Fix qeq/reax requires atom attribute q");

	ngroup = group->count(igroup);
	if (ngroup == 0) error->all(FLERR,"Fix qeq/reax group has no atoms");

	// need a half neighbor list w/ Newton off and ghost neighbors
	// built whenever re-neighboring occurs

	int irequest = neighbor->request(this,instance_me);
	neighbor->requests[irequest]->pair = 0;
	neighbor->requests[irequest]->fix = 1;
	neighbor->requests[irequest]->newton = 2;
	neighbor->requests[irequest]->ghost = 1;

	init_shielding();
	init_taper();

	if (strstr(update->integrate_style,"respa"))
		nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::init_list(int /*id*/, NeighList *ptr)
{
	list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::init_shielding()
{
	int i,j;
	int ntypes;


	ntypes = atom->ntypes;
	if (shld == NULL)
		memory->create(shld,ntypes+1,ntypes+1,"qeq:shielding");


	for (i = 1; i <= ntypes; ++i)
	{
		for (j = 1; j <= ntypes; ++j)
		{
			shld[i][j] = pow( gamma[i] * gamma[j], -1.5);
		}
	}
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::init_taper()
{
	double d7, swa2, swa3, swb2, swb3;

	if (fabs(swa) > 0.01 && comm->me == 0)
		error->warning(FLERR,"Fix qeq/reax has non-zero lower Taper radius cutoff");
	if (swb < 0)
		error->all(FLERR, "Fix qeq/reax has negative upper Taper radius cutoff");
	else if (swb < 5 && comm->me == 0)
		error->warning(FLERR,"Fix qeq/reax has very low Taper radius cutoff");

	d7 = pow( swb - swa, 7);
	swa2 = SQR( swa);
	swa3 = CUBE( swa);
	swb2 = SQR( swb);
	swb3 = CUBE( swb);


	Tap[7] =  20.0 / d7;
	Tap[6] = -70.0 * (swa + swb) / d7;
	Tap[5] =  84.0 * (swa2 + 3.0*swa*swb + swb2) / d7;
	Tap[4] = -35.0 * (swa3 + 9.0*swa2*swb + 9.0*swa*swb2 + swb3) / d7;
	Tap[3] = 140.0 * (swa3*swb + 3.0*swa2*swb2 + swa*swb3) / d7;
	Tap[2] =-210.0 * (swa3*swb2 + swa2*swb3) / d7;
	Tap[1] = 140.0 * swa3 * swb3 / d7;
	Tap[0] = (-35.0*swa3*swb2*swb2 + 21.0*swa2*swb3*swb2 -
			7.0*swa*swb3*swb3 + swb3*swb3*swb) / d7;


	Cuda_Init_Taper(qeq_gpu,Tap,8);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::setup_pre_force(int vflag)
{
	deallocate_storage();
	allocate_storage();

	init_storage();

	deallocate_matrix();
	allocate_matrix();

	pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::setup_pre_force_respa(int vflag, int ilevel)
{
	if (ilevel < nlevels_respa-1) return;
	setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::min_setup_pre_force(int vflag)
{
	setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::init_storage()
{
	int NN;

	if (reaxc)
		NN = reaxc->list->inum + reaxc->list->gnum;
	else
		NN = list->inum + list->gnum;


	for (int i = 0; i < NN; i++) {
		Hdia_inv[i] = 1. / eta[atom->type[i]];
		b_s[i] = -chi[atom->type[i]];
		b_t[i] = -1.0;
		b_prc[i] = 0;
		b_prm[i] = 0;
		s[i] = t[i] = 0;
	}


	CudaInitStorageForFixQeq(qeq_gpu, Hdia_inv, b_s, b_t, b_prc, b_prm, s, t,NN);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::pre_force(int /*vflag*/)
{
	double t_start, t_end;

	if (update->ntimestep % nevery) return;
	if (comm->me == 0) t_start = MPI_Wtime();

	n = atom->nlocal;
	N = atom->nlocal + atom->nghost;

	// grow arrays if necessary
	// need to be atom->nmax in length

	if (atom->nmax > nmax) reallocate_storage();
	if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
		reallocate_matrix();

	init_matvec();


	matvecs_s = Cuda_CG(qeq_gpu->b_s, qeq_gpu->s);       // CG on s - parallel
	matvecs_t = Cuda_CG(qeq_gpu->b_t, qeq_gpu->t);       // CG on t - parallel

	matvecs = matvecs_s + matvecs_t;

	printf("Matvecs %d \n", matvecs);

	cuda_calculate_Q();


	if (comm->me == 0) {
		t_end = MPI_Wtime();
		qeq_time = t_end - t_start;
	}
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::pre_force_respa(int vflag, int ilevel, int /*iloop*/)
{
	if (ilevel == nlevels_respa-1) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::min_pre_force(int vflag)
{
	pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::init_matvec()
{
	/* fill-in H matrix */
	compute_H();

	int nn, ii, i;
	int *ilist;

	if (reaxc) {
		nn = reaxc->list->inum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		ilist = list->ilist;
	}


	Cuda_Init_Matvec_Fix(nn, qeq_gpu,reaxc->system);

	for (ii = 0; ii < nn; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit) {

			/* init pre-conditioner for H and init solution vectors */
			Hdia_inv[i] = 1. / eta[ atom->type[i]];
			b_s[i]      = -chi[ atom->type[i] ];
			b_t[i]      = -1.0;

			/* quadratic extrapolation for s & t from previous solutions */
			t[i] = t_hist[i][2] + 3 * ( t_hist[i][0] - t_hist[i][1]);

			/* cubic extrapolation for s & t from previous solutions */
			s[i] = 4*(s_hist[i][0]+s_hist[i][2])-(6*s_hist[i][1]+s_hist[i][3]);
		}
	}

	pack_flag = 2;
	comm->forward_comm_fix(this); //Dist_vector( s );
	pack_flag = 3;
	comm->forward_comm_fix(this); //Dist_vector( t );


}

/*-----------------------------------------------------------------------*/
void FixQEqReax::intializeAtomsAndCopyToDevice()
{
	//int *num_bonds = fix_reax->num_bonds;
	//int *num_hbonds = fix_reax->num_hbonds;
	qeq_gpu->fix_my_atoms = (reax_atom*) malloc(reaxc->system->N * sizeof(reax_atom));
	for( int i = 0; i < reaxc->system->N; ++i ){
		qeq_gpu->fix_my_atoms[i].orig_id = atom->tag[i];
		qeq_gpu->fix_my_atoms[i].type = atom->type[i];
		qeq_gpu->fix_my_atoms[i].x[0] = atom->x[i][0];
		qeq_gpu->fix_my_atoms[i].x[1] = atom->x[i][1];
		qeq_gpu->fix_my_atoms[i].x[2] = atom->x[i][2];
		qeq_gpu->fix_my_atoms[i].q = atom->q[i];
		// qeq_gpu->fix_my_atoms[i].num_bonds = num_bonds[i];
		//qeq_gpu->fix_my_atoms[i].num_hbonds = num_hbonds[i];
	}
	Cuda_Init_Fix_Atoms(reaxc->system, qeq_gpu);

}

/* ---------------------------------------------------------------------- */

void FixQEqReax::compute_H()
{
	int inum, jnum, *ilist, *jlist, *numneigh, **firstneigh;
	int i, j, ii, jj, flag;
	double dx, dy, dz, r_sqr;
	const double SMALL = 0.0001;

	int *type = atom->type;
	tagint *tag = atom->tag;
	double **x = atom->x;
	int *mask = atom->mask;

	if (reaxc) {
		inum = reaxc->list->inum;
		ilist = reaxc->list->ilist;
		numneigh = reaxc->list->numneigh;
		firstneigh = reaxc->list->firstneigh;
	} else {
		inum = list->inum;
		ilist = list->ilist;
		numneigh = list->numneigh;
		firstneigh = list->firstneigh;
	}


	printf("Ttoal cap, total cm , n %d,%d,%d\n", reaxc->system->total_cap, reaxc->system->total_cm_entries,reaxc->system->N);

	intializeAtomsAndCopyToDevice();
	Cuda_Allocate_Matrix(&qeq_gpu->H, reaxc->system->total_cap, reaxc->system->total_cm_entries);
	Cuda_Estimate_CMEntries_Storages(reaxc->system, reaxc->control,reaxc->gpu_lists, qeq_gpu, inum);
	Cuda_Init_Sparse_Matrix_Indices( reaxc->system, qeq_gpu, inum);
	Cuda_Calculate_H_Matrix(reaxc->gpu_lists, reaxc->system,qeq_gpu,reaxc->control,inum,SMALL);



	printf("System N %d, m %d \n", reaxc->system->N,n);



	// fill in the H matrix

	printf("I num %d \n", inum);

	printf("Atom nmax %d, n local %d  \n", atom->nmax,atom->nlocal);

	printf("Swb %f, SMALL %f\n", swb, SMALL);


	m_fill = 0;
	r_sqr = 0;
	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		if (mask[i] & groupbit) {
			jlist = firstneigh[i];
			jnum = numneigh[i];
			H.firstnbr[i] = m_fill;
			for (jj = 0; jj < jnum; jj++) {
				j = jlist[jj];


				j &= NEIGHMASK;
				dx = x[j][0] - x[i][0];
				dy = x[j][1] - x[i][1];
				dz = x[j][2] - x[i][2];
				r_sqr = SQR(dx) + SQR(dy) + SQR(dz);


				flag = 0;
				if (r_sqr <= SQR(swb)) {
					if (j < n)
					{
						flag = 1;

					}
					else if (tag[i] < tag[j])
					{
						flag = 1;
					}
					else if (tag[i] == tag[j]) {
						if (dz > SMALL)
						{
							flag = 1;
						}
						else if (fabs(dz) < SMALL) {
							if (dy > SMALL)
							{
								flag = 1;

							}
							else if (fabs(dy) < SMALL && dx > SMALL)
							{
								flag = 1;
							}
						}
					}
				}

				if (flag) {
					H.jlist[m_fill] = j;
					H.val[m_fill] = calculate_H( sqrt(r_sqr), shld[type[i]][type[j]]);
					m_fill++;
				}
			}
			H.numnbrs[i] = m_fill - H.firstnbr[i];
		}

		//printf("Index : %d, H first number %d , m fill : %d, NumNbrs:%d,  \n",i,  H.firstnbr[i],m_fill,H.numnbrs[i]);

	}

	if (m_fill >= H.m) {
		char str[128];
		sprintf(str,"H matrix size has been exceeded: m_fill=%d H.m=%d\n",
				m_fill, H.m);
		error->warning(FLERR,str);
		error->all(FLERR,"Fix qeq/reax has insufficient QEq matrix size");
	}

}

/* ---------------------------------------------------------------------- */

double FixQEqReax::calculate_H( double r, double gamma)
{
	double Taper, denom;

	Taper = Tap[7] * r + Tap[6];
	Taper = Taper * r + Tap[5];
	Taper = Taper * r + Tap[4];
	Taper = Taper * r + Tap[3];
	Taper = Taper * r + Tap[2];
	Taper = Taper * r + Tap[1];
	Taper = Taper * r + Tap[0];

	denom = r * r * r + gamma;
	denom = pow(denom,0.3333333333333);

	return Taper * EV_TO_KCAL_PER_MOL / denom;
}

/* ---------------------------------------------------------------------- */


int FixQEqReax::Cuda_CG( double *device_b, double *device_x)
{
	int  i, j, imax;
	double tmp, alpha, beta, b_norm;
	double sig_old, sig_new;

	int nn, jj;
	int *ilist;
	if (reaxc) {
		nn = reaxc->list->inum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		ilist = list->ilist;
	}

	imax = 200;

	cuda_sparse_matvec(device_x, qeq_gpu->q);


	pack_flag = 1;
	comm->reverse_comm_fix(this); //Coll_Vector( q );


	Cuda_Vector_Sum_Fix(qeq_gpu->r , 1.0,  device_b, -1.0,
			qeq_gpu->q, nn);

	printf("\n\n");


	Cuda_CG_Preconditioner_Fix(qeq_gpu->d, qeq_gpu->r,
			qeq_gpu->Hdia_inv,  nn);


	real *b,*r,*d,*q,*p;
	memory->create(b,nn,"b_temp");
	memory->create(r,nn,"r_temp");
	memory->create(d,nn,"d_temp");
	memory->create(q,nn,"d_temp");
	memory->create(p,nn,"d_temp");



	Cuda_Copy_Vector_From_Device(b,device_b,nn);
	b_norm = parallel_norm( b, nn);

	Cuda_Copy_Vector_From_Device(r,qeq_gpu->r,nn);
	Cuda_Copy_Vector_From_Device(d,qeq_gpu->d,nn);




	sig_new = parallel_dot(r,d,nn);

	printf("b NORM %f, sig new %f \n", b_norm, sig_new);


	for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
		printf("\n\n");
		printf("Packing forward comm on d\n");
		comm->forward_comm_fix(this); //Dist_vector( d );
		printf("\n\n");
		cuda_sparse_matvec(qeq_gpu->d, qeq_gpu->q);
		comm->reverse_comm_fix(this); //Coll_vector( q );

		Cuda_Copy_Vector_From_Device(d,qeq_gpu->d,nn);

		printf("\n\n");
		for(int i = 0; i < nn; i++)
		{
			printf("D[%d]=%f\n", i,d[i]);
		}

		printf("\n\n");

		Cuda_Copy_Vector_From_Device(q,qeq_gpu->q,nn);

		for(int i = 0; i < nn; i++)
		{
			printf("Q[%d]=%f\n", i,q[i]);
		}


		tmp = parallel_dot( d, q, nn);

		printf("Tmp %f \n", tmp);
		alpha = sig_new / tmp;

		printf("Device x vector sum \n");
		Cuda_Vector_Sum_Fix(device_x , alpha,  qeq_gpu->d, 1.0, device_x, nn);
		printf("\n\n");


		printf("Device r vector sum \n");
		Cuda_Vector_Sum_Fix(qeq_gpu->r, -alpha, qeq_gpu->q, 1.0,qeq_gpu->r,nn);
		printf("\n\n");


		Cuda_CG_Preconditioner_Fix(qeq_gpu->p,qeq_gpu->r,qeq_gpu->Hdia_inv,nn);

		sig_old = sig_new;

		Cuda_Copy_Vector_From_Device(r,qeq_gpu->r,nn);
		Cuda_Copy_Vector_From_Device(p,qeq_gpu->p,nn);

		sig_new = parallel_dot( r, p, nn);

		beta = sig_new / sig_old;

		printf("Device d vector sum \n");
		Cuda_Vector_Sum_Fix(qeq_gpu->d, 1.,qeq_gpu->p,beta,qeq_gpu->d,nn);
		printf("\n\n");


		printf("i:%d,beta %f, sig new %f, alpha %f\n",i, beta, sig_new, alpha);


	}



	if (i >= imax && comm->me == 0) {
		char str[128];
		sprintf(str,"Fix qeq/reax CG convergence failed after %d iterations "
				"at " BIGINT_FORMAT " step",i,update->ntimestep);
		error->warning(FLERR,str);
	}

	printf("I is %d \n", i);
	return i;
}



int FixQEqReax::CG( double *b, double *x)
{
	int  i, j, imax;
	double tmp, alpha, beta, b_norm;
	double sig_old, sig_new;

	int nn, jj;
	int *ilist;
	if (reaxc) {
		nn = reaxc->list->inum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		ilist = list->ilist;
	}

	imax = 200;

	pack_flag = 1;
	sparse_matvec( &H, x, q);




	//Cuda_Matvec(qeq_gpu->H, qeq_gpu->s, workspace->d_workspace->q, system->N, system->total_cap );

	comm->reverse_comm_fix(this); //Coll_Vector( q );

	vector_sum( r , 1.,  b, -1., q, nn);

	for (jj = 0; jj < nn; ++jj) {
		j = ilist[jj];
		if (atom->mask[j] & groupbit)
			d[j] = r[j] * Hdia_inv[j]; //pre-condition
	}

	b_norm = parallel_norm( b, nn);
	sig_new = parallel_dot( r, d, nn);

	for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
		comm->forward_comm_fix(this); //Dist_vector( d );
		sparse_matvec( &H, d, q );
		comm->reverse_comm_fix(this); //Coll_vector( q );

		tmp = parallel_dot( d, q, nn);
		alpha = sig_new / tmp;

		vector_add( x, alpha, d, nn );
		vector_add( r, -alpha, q, nn );

		// pre-conditioning
		for (jj = 0; jj < nn; ++jj) {
			j = ilist[jj];
			if (atom->mask[j] & groupbit)
				p[j] = r[j] * Hdia_inv[j];
		}

		sig_old = sig_new;
		sig_new = parallel_dot( r, p, nn);

		beta = sig_new / sig_old;
		vector_sum( d, 1., p, beta, d, nn );
	}

	if (i >= imax && comm->me == 0) {
		char str[128];
		sprintf(str,"Fix qeq/reax CG convergence failed after %d iterations "
				"at " BIGINT_FORMAT " step",i,update->ntimestep);
		error->warning(FLERR,str);
	}

	return i;
}


/* ---------------------------------------------------------------------- */
void FixQEqReax::cuda_sparse_matvec(double *x, double *q)
{
	int i, j, itr_j;
	int nn, NN, ii;
	int *ilist;

	if (reaxc) {
		nn = reaxc->list->inum;
		NN = reaxc->list->inum + reaxc->list->gnum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		NN = list->inum + list->gnum;
		ilist = list->ilist;
	}

	Cuda_Sparse_Matvec_Compute(&qeq_gpu->H, x, q ,qeq_gpu->eta, qeq_gpu->d_fix_my_atoms, nn, NN);
}




void FixQEqReax::sparse_matvec( sparse_matrix *A, double *x, double *b)
{
	int i, j, itr_j;
	int nn, NN, ii;
	int *ilist;

	if (reaxc) {
		nn = reaxc->list->inum;
		NN = reaxc->list->inum + reaxc->list->gnum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		NN = list->inum + list->gnum;
		ilist = list->ilist;
	}

	for (ii = 0; ii < nn; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit)
			b[i] = eta[ atom->type[i] ] * x[i];
	}

	for (ii = nn; ii < NN; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit)
			b[i] = 0;
	}

	for (ii = 0; ii < nn; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit) {
			for (itr_j=A->firstnbr[i]; itr_j<A->firstnbr[i]+A->numnbrs[i]; itr_j++) {
				j = A->jlist[itr_j];
				b[i] += A->val[itr_j] * x[j];
				b[j] += A->val[itr_j] * x[i];
			}
		}
	}

}

/* ---------------------------------------------------------------------- */

void FixQEqReax::cuda_calculate_Q()
{
	int i, k;
	double u;
	double *q = atom->q;

	int nn, ii;
	int *ilist;

	if (reaxc) {
		nn = reaxc->list->inum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		ilist = list->ilist;
	}


	double s_sum = Cuda_Calculate_Local_S_Sum(nn,qeq_gpu);
	double t_sum = Cuda_Calculate_Local_T_Sum(nn,qeq_gpu);

	u = s_sum / t_sum;


    Cuda_Update_Q_And_Backup_ST(nn,qeq_gpu,u);


	pack_flag = 4;
	comm->forward_comm_fix(this); //Dist_vector( atom->q );

}




void FixQEqReax::calculate_Q()
{
	int i, k;
	double u, s_sum, t_sum;
	double *q = atom->q;

	int nn, ii;
	int *ilist;

	if (reaxc) {
		nn = reaxc->list->inum;
		ilist = reaxc->list->ilist;
	} else {
		nn = list->inum;
		ilist = list->ilist;
	}

	s_sum = parallel_vector_acc( s, nn);
	t_sum = parallel_vector_acc( t, nn);
	u = s_sum / t_sum;

	for (ii = 0; ii < nn; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit) {
			q[i] = s[i] - u * t[i];

			/* backup s & t */
			for (k = nprev-1; k > 0; --k) {
				s_hist[i][k] = s_hist[i][k-1];
				t_hist[i][k] = t_hist[i][k-1];
			}
			s_hist[i][0] = s[i];
			t_hist[i][0] = t[i];
		}
	}

	pack_flag = 4;
	comm->forward_comm_fix(this); //Dist_vector( atom->q );
}

/* ---------------------------------------------------------------------- */

int FixQEqReax::pack_forward_comm(int n, int *list, double *buf,
		int /*pbc_flag*/, int * /*pbc*/)
{
	int m;

	if (pack_flag == 1)
	{
		Cuda_Copy_From_Device_Comm_Fix(buf,qeq_gpu->d,n,0);

		/*printf("Copying from device \n");
		for(m = 0; m < n; m++)
		{
			printf("%f\n", buf[m]);
		}

		printf("\n\n");*/

		/*for(m = 0; m < n; m++)
		{
			buf[m] = d[list[m]];
		}*/
	}
	else if (pack_flag == 2)
	{
		//TB:: Ask if list[m] is always linearly increasing
		Cuda_Copy_From_Device_Comm_Fix(buf,qeq_gpu->s,n,0);
		/*for(m = 0; m < n; m++)
		{
			printf("List %d \n", list[m]);
			buf[m] = s[list[m]];
		}*/
	}
	else if (pack_flag == 3)
	{
		Cuda_Copy_From_Device_Comm_Fix(buf,qeq_gpu->t,n,0);
		/*for(m = 0; m < n; m++)
		{
			buf[m] = t[list[m]];
		}*/
	}
	else if (pack_flag == 4)
	{
		Cuda_Copy_From_Device_Comm_Fix(buf,qeq_gpu->q,n,0);

		//TB:: Find out where atom->q is used
		/*for(m = 0; m < n; m++)
		{
			buf[m] = atom->q[list[m]];
		}*/
	}
	else if (pack_flag == 5) {
		printf("Not used \n");
		m = 0;
		exit(0);
		for(int i = 0; i < n; i++) {
			int j = 2 * list[i];
			buf[m++] = d[j];
			buf[m++] = d[j+1];
		}
		return m;
	}
	return n;
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::unpack_forward_comm(int n, int first, double *buf)
{
	int i, m;

	if (pack_flag == 1)
	{
		Cuda_Copy_To_Device_Comm_Fix(buf,qeq_gpu->d,n,first);

		/*printf("Copying to device \n");

		for(m = 0; m < n; m++)
		{
			printf("%f\n", buf[m]);
		}

		printf("\n\n");*/

		/*for(m = 0, i = first; m < n; m++, i++)
		{
			d[i] = buf[m];
		}*/

	}
	else if (pack_flag == 2)
	{
		Cuda_Copy_To_Device_Comm_Fix(buf,qeq_gpu->s,n,first);

		/*for(m = 0, i = first; m < n; m++, i++)
		{
			printf("I %d, m%d \n", i, m);
			s[i] = buf[m];
		}*/
	}
	else if (pack_flag == 3)
	{
		Cuda_Copy_To_Device_Comm_Fix(buf,qeq_gpu->t,n,first);

		/*for(m = 0, i = first; m < n; m++, i++)
		{
			t[i] = buf[m];
		}*/
	}
	else if (pack_flag == 4)
	{
		Cuda_Copy_To_Device_Comm_Fix(buf,qeq_gpu->q,n,first);

		/*for(m = 0, i = first; m < n; m++, i++)
		{
			atom->q[i] = buf[m];
		}*/
	}
	else if (pack_flag == 5) {
		printf("Not used \n");
		exit(0);
		int last = first + n;
		m = 0;
		for(i = first; i < last; i++) {
			int j = 2 * i;
			d[j  ] = buf[m++];
			d[j+1] = buf[m++];
		}
	}
}

/* ---------------------------------------------------------------------- */

int FixQEqReax::pack_reverse_comm(int n, int first, double *buf)
{
	int i, m;
	if (pack_flag == 5) {
		m = 0;
		int last = first + n;
		for(i = first; i < last; i++) {
			int indxI = 2 * i;
			buf[m++] = q[indxI  ];
			buf[m++] = q[indxI+1];
		}
		printf("Not used \n");
		exit(0);
		return m;
	}
	else
	{
		//printf("Copying q from device \n");
		Cuda_Copy_From_Device_Comm_Fix(buf,qeq_gpu->q,n,first);
		for (m = 0, i = first; m < n; m++, i++)
		{
			//printf("Buf m %f\n", buf[m]);
			//buf[m] = q[i];

		}
		//printf("\n\n");

		return n;
	}
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::unpack_reverse_comm(int n, int *list, double *buf)
{
	if (pack_flag == 5) {
		int m = 0;
		for(int i = 0; i < n; i++) {
			int indxI = 2 * list[i];
			q[indxI  ] += buf[m++];
			q[indxI+1] += buf[m++];
		}
		printf("Not used \n");
		exit(0);
	}
	else
	{
		/*for (int m = 0; m < n; m++)
		{
			q[list[m]] += buf[m];
		}*/
		//printf("Copying q to device \n");

		Cuda_UpdateQ_And_Copy_To_Device_Comm_Fix(buf,qeq_gpu,n);

		//printf("\n\n");
	}
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixQEqReax::memory_usage()
{
	double bytes;

	bytes = atom->nmax*nprev*2 * sizeof(double); // s_hist & t_hist
	bytes += atom->nmax*11 * sizeof(double); // storage
	bytes += n_cap*2 * sizeof(int); // matrix...
	bytes += m_cap * sizeof(int);
	bytes += m_cap * sizeof(double);

	if (dual_enabled)
		bytes += atom->nmax*4 * sizeof(double); // double size for q, d, r, and p

	return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqReax::grow_arrays(int nmax)
{
	memory->grow(s_hist,nmax,nprev,"qeq:s_hist");
	memory->grow(t_hist,nmax,nprev,"qeq:t_hist");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqReax::copy_arrays(int i, int j, int /*delflag*/)
{
	for (int m = 0; m < nprev; m++) {
		s_hist[j][m] = s_hist[i][m];
		t_hist[j][m] = t_hist[i][m];
	}
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixQEqReax::pack_exchange(int i, double *buf)
{
	for (int m = 0; m < nprev; m++) buf[m] = s_hist[i][m];
	for (int m = 0; m < nprev; m++) buf[nprev+m] = t_hist[i][m];
	return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixQEqReax::unpack_exchange(int nlocal, double *buf)
{
	for (int m = 0; m < nprev; m++) s_hist[nlocal][m] = buf[m];
	for (int m = 0; m < nprev; m++) t_hist[nlocal][m] = buf[nprev+m];
	return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixQEqReax::parallel_norm( double *v, int n)
{
	int  i;
	double my_sum, norm_sqr;

	int ii;
	int *ilist;

	if (reaxc)
		ilist = reaxc->list->ilist;
	else
		ilist = list->ilist;

	my_sum = 0.0;
	norm_sqr = 0.0;
	for (ii = 0; ii < n; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit)
			my_sum += SQR( v[i]);
	}

	MPI_Allreduce( &my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

	return sqrt( norm_sqr);
}

/* ---------------------------------------------------------------------- */

double FixQEqReax::parallel_dot( double *v1, double *v2, int n)
{
	int  i;
	double my_dot, res;

	int ii;
	int *ilist;

	if (reaxc)
		ilist = reaxc->list->ilist;
	else
		ilist = list->ilist;

	my_dot = 0.0;
	res = 0.0;
	for (ii = 0; ii < n; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit)
			my_dot += v1[i] * v2[i];
	}

	MPI_Allreduce( &my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

	return res;
}

/* ---------------------------------------------------------------------- */

double FixQEqReax::parallel_vector_acc( double *v, int n)
{
	int  i;
	double my_acc, res;

	int ii;
	int *ilist;

	if (reaxc)
		ilist = reaxc->list->ilist;
	else
		ilist = list->ilist;

	my_acc = 0.0;
	res = 0.0;
	for (ii = 0; ii < n; ++ii) {
		i = ilist[ii];
		if (atom->mask[i] & groupbit)
			my_acc += v[i];
	}

	MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

	return res;
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::vector_sum( double* dest, double c, double* v,
		double d, double* y, int k)
{
	int kk;
	int *ilist;

	if (reaxc)
		ilist = reaxc->list->ilist;
	else
		ilist = list->ilist;

	for (--k; k>=0; --k) {
		kk = ilist[k];
		if (atom->mask[kk] & groupbit)
			dest[kk] = c * v[kk] + d * y[kk];
	}
}

/* ---------------------------------------------------------------------- */

void FixQEqReax::vector_add( double* dest, double c, double* v, int k)
{
	int kk;
	int *ilist;

	if (reaxc)
		ilist = reaxc->list->ilist;
	else
		ilist = list->ilist;

	for (--k; k>=0; --k) {
		kk = ilist[k];
		if (atom->mask[kk] & groupbit)
		{
			dest[kk] += c * v[kk];

		}
	}
}

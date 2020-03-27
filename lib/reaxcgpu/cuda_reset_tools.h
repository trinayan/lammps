
#ifndef __CUDA_RESET_TOOLS_H__
#define __CUDA_RESET_TOOLS_H__

#include "reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Reset_Workspace( reax_system *, gpu_storage * );

void Cuda_Reset_Atoms( reax_system *, control_params *, gpu_storage *);

int  Cuda_Reset_Neighbor_Lists( reax_system *, control_params *,
        gpu_storage *, reax_list ** );

void Cuda_Reset( reax_system*, control_params*, simulation_data*,
        gpu_storage*, reax_list** );

#ifdef __cplusplus
}
#endif


#endif

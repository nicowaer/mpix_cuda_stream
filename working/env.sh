#!/bin/bash

# Modules
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$PREFIX/bin:$PATH
module load EasyBuild/2023a
module load CUDA/12.2.0
module load GCC/12.3.0
module load GDRCopy/2.3.1-GCCcore-12.3.0.lua

# OFI env vars
export FI_PROVIDER="verbs,ofi_rxm,shm" #,,ofi_rxm ,verbs shm, ;ofi_rxm
export FI_HMEM_CUDA_USE_GDRCOPY=1

export FI_OFI_RXM_BUFFER_SIZE=512
export FI_OFI_RXM_SAR_LIMIT=512     # we don't want SAR

# OFI debug stuff
#export FI_LOG_LEVEL=debug

# MPICH env vars
#export MPIR_CVAR_NOLOCAL=1 # disable local communication makes all communication go through the network stack
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1
#export MPIR_CVAR_CH4_OFI_MAX_NICS=2
export MPIR_CVAR_CH4_RESERVE_VCIS=2

# MPICH debug stuff
#export MPICH_DBG=yes
export MPIR_CVAR_DEBUG_SUMMARY=1
export MPIR_CVAR_CH4_OFI_ENABLE_DEBUG=1
export MPIR_CVAR_PRINT_ERROR_STACK=1

# HYDRA env vars
export HYDRA_TOPO_DEBUG=1

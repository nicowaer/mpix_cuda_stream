#!/bin/bash

# Get my rank in the world comm
nrank_world=${PMI_SIZE}
irank_world=${PMI_RANK}

# Get my rank in the node-local comm
nrank_local=${MPI_LOCALNRANKS}
irank_local=${MPI_LOCALRANKID}

# Select the device & nic I see
if   [ "${nrank_local}" == 1 ]; then
    # 1 GPU/node
    dev=3
    nic=0

elif [ "${nrank_local}" == 2 ]; then
    # 2 GPU/node
    if   [ "${irank_local}" == 0 ]; then
        dev=3
        nic=0
    elif [ "${irank_local}" == 1 ]; then
        dev=1
        nic=1
    fi

elif [ "${nrank_local}" == 4 ]; then
    # 4 GPU/node
    if   [ "${irank_local}" == 0 ]; then
        dev=3
        nic=0
    elif [ "${irank_local}" == 1 ]; then
        dev=2
        nic=0
    elif [ "${irank_local}" == 2 ]; then
        dev=1
        nic=1
    elif [ "${irank_local}" == 3 ]; then
        dev=0
        nic=1
    fi

else
    raise error "nrank_local = ${nrank_local}, not supported"
fi


export CUDA_VISIBLE_DEVICES=${dev}
export MPIR_CVAR_CH4_OFI_IFNAME=mlx5_${nic}

echo "Setting my GPU to device n°${CUDA_VISIBLE_DEVICES} with ifname ${MPIR_CVAR_CH4_OFI_IFNAME}"

# Run...
$@

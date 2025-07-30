#!/bin/bash
#SBATCH --job-name=pingpong
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --account=examples
#SBATCH --output=cuda-%j.out

# DBS-compiled MPICH
export PREFIX=$HOME/soft/lib-MPICH-4.2.2-OFI-1.22.0-CUDA-12.2.0-opt
source env.sh

mpic++ -O3 -I${EBROOTCUDA}/include -L${EBROOTCUDA}/lib64 -lcudart src/cuda_stream.cpp -o src/cuda_stream

nvidia-smi
nvidia-smi topo -m

mpiexec -l -n 2 -ppn 2 ./split.sh ./src/cuda_stream

rm src/cuda_stream

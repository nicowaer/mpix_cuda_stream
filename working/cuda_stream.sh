#!/bin/bash
#SBATCH --job-name=pingpong
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00
#SBATCH --account=examples
#SBATCH --output=cuda-%j.out

# DBS-compiled MPICH
export PREFIX=$HOME/soft/lib-MPICH-4.2.2-OFI-1.22.0-CUDA-12.2.0-opt
source env.sh

#nvcc -g -o src/stream_test src/stream_test.cu -I$PREFIX/include -L$PREFIX/lib -lmpi -lcudart

mpic++ -O3 -I${EBROOTCUDA}/include -L${EBROOTCUDA}/lib64 -lcudart src/stream_test.cpp -o src/stream_test

nvidia-smi
nvidia-smi topo -m

#mpiexec -l -n 2 -ppn 2 ./src/cuda_stream
mpiexec -l -n 2 -ppn 2 ./split.sh ./src/stream_test

rm src/stream_test

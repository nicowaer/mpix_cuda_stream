/* CC: nvcc -g */
/* lib_list: -lmpi */
/* run: mpirun -l -n 2 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpi_errno;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Process %d / %d\n", rank, size);

    //cudaSetDevice(rank % 2); // Use different GPUs for each rank

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    int N = 1000000;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
        }
    } else if (rank == 1) {
        for (int i = 0; i < N; i++) {
            y[i] = 2.0f;
        }
    }

    char str_stream[20];
    snprintf(str_stream, 20, "%d", stream);

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "type", "cudaStream_t");
    //MPI_Info_set(info, "value", &stream);
    MPI_Info_set(info, "id", str_stream);
    MPIX_Info_set_hex(info, "value", (void*)&stream, sizeof(cudaStream_t));

    MPIX_Stream mpi_stream;
    MPIX_Stream_create(info, &mpi_stream);

    MPI_Comm stream_comm;
    MPIX_Stream_comm_create(MPI_COMM_WORLD, mpi_stream, &stream_comm);

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

#if 1    
    if (rank == 0) {
      #if 1  
        cudaMemcpyAsync(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice, stream);

        mpi_errno = MPIX_Send_enqueue(d_x, N, MPI_FLOAT, 1, 0, stream_comm);
      #else  
        mpi_errno = MPIX_Send_enqueue(x, N, MPI_FLOAT, 1, 0, stream_comm);
      #endif  
        assert(mpi_errno == MPI_SUCCESS);

        cudaStreamSynchronize(stream);
    } else if (rank == 1) {
        cudaMemcpyAsync(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice, stream);

        mpi_errno = MPIX_Recv_enqueue(d_x, N, MPI_FLOAT, 0, 0, stream_comm, MPI_STATUS_IGNORE);
        assert(mpi_errno == MPI_SUCCESS);

        // Perform SAXPY on 1M elements
        saxpy<<<(N+255)/256, 256, 0, stream>>>(N, 2.0f, d_x, d_y);

        cudaMemcpyAsync(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
    }

#else
    if (rank == 0) {
        cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
        mpi_errno = MPI_Send(d_x, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        assert(mpi_errno == MPI_SUCCESS);
    } else if (rank == 1) {
        cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
        mpi_errno = MPI_Recv(d_x, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        assert(mpi_errno == MPI_SUCCESS);

        // Perform SAXPY on 1M elements
        saxpy<<<(N+255)/256, 256, 0>>>(N, 2.0f, d_x, d_y);

        cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    }

#endif
    end_time = MPI_Wtime();

    if (rank == 1) {
        float maxError = 0.0f;
        int errs = 0;
        for (int i = 0; i < N; i++) {
            if (abs(y[i] - 4.0f) > 0.01) {
                errs++;
                maxError = max(maxError, abs(y[i]-4.0f));
            }
        }
        printf("%d errors, Max error: %f\n", errs, maxError);
    }
    printf("Process %d finished in %f seconds\n", rank, end_time - start_time);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    cudaStreamDestroy(stream);
    MPI_Finalize();
}

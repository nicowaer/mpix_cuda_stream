#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

// Macro for checking errors in CUDA API calls
#define m_cudart_check_errors(res)                                                              \
    do {                                                                                        \
        cudaError_t __err = res;                                                                \
        if (__err != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA RT error %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, \
                    __LINE__);                                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("Process %d / %d\n", world_rank, world_size);

    if (world_size < 2) {
        if (world_rank == 0)
            std::cerr << "Please run with at least 2 ranks\n";
        MPI_Finalize();
        return -1;
    }

    // Create CUDA stream
    cudaStream_t stream;
    m_cudart_check_errors(cudaStreamCreate(&stream));

    // found in MPIX+stream paper
    MPI_Info info ;
    MPI_Info_create(&info);
    MPI_Info_set(info, "type", "cudaStream_t");
    int err_set_hex = MPIX_Info_set_hex(info, "value", &stream, sizeof(stream));
    if (err_set_hex != MPI_SUCCESS) {
        std::cerr << "MPIX_Info_set_hex failed on rank " << world_rank << "\n";
        MPI_Info_free(&info);
        MPI_Finalize();
        return -1;
    }
    //}

    MPIX_Stream mpi_stream;
    int err_stream = MPIX_Stream_create(info, &mpi_stream);
    if (err_stream != MPI_SUCCESS) {
        std::cerr << "MPIX_Stream_create failed on rank " << world_rank << "\n";
        MPI_Info_free(&info);
        MPI_Finalize();
        return -1;
    }
    MPI_Info_free(&info);

    // Create stream communicator based on MPI_COMM_WORLD
    MPI_Comm stream_comm;
    int err = MPIX_Stream_comm_create(MPI_COMM_WORLD, mpi_stream, &stream_comm);
    if (err != MPI_SUCCESS) {
        std::cerr << "MPIX_Stream_comm_create failed on rank " << world_rank << "\n";
        MPI_Finalize();
        return -1;
    }

    int rank;
    MPI_Comm_rank(stream_comm, &rank);

    int partner = (world_rank + 1) % 2;  // rank 0 <-> 1 communication, other ranks idle

    MPI_Request send_req, recv_req;
    MPI_Status status;
    double N = 1 << 12;

    if (world_rank == 0){
        double *d_x;
        m_cudart_check_errors(cudaMalloc(&d_x, N*sizeof(double)));
        double* x = (double*)malloc(N*sizeof(double));
            for(int i = 0; i < N; i++) {
                x[i] = static_cast<double>(i);
        }
        m_cudart_check_errors(cudaMemcpyAsync(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice));
        
        int err = MPIX_Send_enqueue(d_x, (int) N, MPI_DOUBLE, partner, 0, stream_comm);
        //int err = MPIX_Isend_enqueue(d_x, (int) N, MPI_DOUBLE, partner, 0, stream_comm, &send_req);
        //int err = MPI_Isend(d_x, (int) N, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, &send_req);
        //int err = MPI_Send(d_x, (int) N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        
        if (err != MPI_SUCCESS) {
            char err_string[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(err, err_string, &len);
            std::cerr << "MPIX_Isend_enqueue failed: " << err_string << std::endl;
        }

        cudaStreamSynchronize(stream);
    }

    if (world_rank == 1) {
        double *d_recvbuf_1;
        m_cudart_check_errors(cudaMalloc(&d_recvbuf_1, N*sizeof(double)));

        //int err = MPIX_Recv_enqueue(d_recvbuf_1, (int)N, MPI_DOUBLE, 0, 0, stream_comm, &status);
        //int err = MPI_Irecv(d_recvbuf_1, (int)N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recv_req);
        int err = MPI_Recv(d_recvbuf_1, (int)N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if (err != MPI_SUCCESS) {
            char err_string[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(err, err_string, &len);
            std::cerr << "MPI_Recv failed: " << err_string << std::endl;
        }

        double* recv_host = (double*)malloc(N * sizeof(double));
        m_cudart_check_errors(cudaMemcpyAsync(recv_host, d_recvbuf_1, N * sizeof(double), cudaMemcpyDeviceToHost, stream));

        printf("Rank %d received data successfully.\n", world_rank);
        cudaStreamSynchronize(stream);

        for (int i = 0; i < 10; i++) {
            printf("recv_host[%d] = %f\n", i, recv_host[i]);
        }
        
    }

    cudaFree(d_recvbuf_1);
    MPI_Comm_free(&stream_comm);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    return 0;
}

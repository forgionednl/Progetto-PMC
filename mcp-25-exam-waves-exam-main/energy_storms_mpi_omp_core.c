#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "energy_storms.h"

#define CELL_BLOCK 1024

/* Aggiorno una cella usando la radice gia calcolata per quella distanza. */
static inline void update( float *layer, const float *attenuation, float threshold,
                           int k_loc, int k_glob, int pos, float energy ) {
    int distance = pos - k_glob;
    if ( distance < 0 ) distance = - distance;
    float energy_k = energy / attenuation[distance];

    layer[k_loc] = (fabsf(energy_k) >= threshold)
                  ? layer[k_loc] + energy_k : layer[k_loc];
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, k;
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int base_size = layer_size / num_procs;
    int remainder = layer_size % num_procs;

    int local_size;
    int start_index;

    if ( rank < remainder ) {
        local_size = base_size + 1;
        start_index = rank * (base_size + 1);
    } else {
        local_size = base_size;
        start_index = remainder * (base_size + 1) + (rank - remainder) * base_size;
    }

    int alloc_size = local_size + 2;
    float *local_layer = (float *)malloc( sizeof(float) * alloc_size );
    float *local_layer_copy = (float *)malloc( sizeof(float) * alloc_size );

    /* Salvo le radici una volta sola: la distanza si ripete per ogni particella.
       Considero anche eventuali impatti fuori dal layer. */
    int max_distance = layer_size - 1;
    for (i = 0; i < num_storms; i++) {
        for (int j = 0; j < storms[i].size; j++) {
            int pos = storms[i].posval[2*j];
            int d = abs(pos);
            if (d > max_distance) max_distance = d;
            d = abs(pos - (layer_size - 1));
            if (d > max_distance) max_distance = d;
        }
    }
    float *attenuation = malloc(sizeof(float) * ((size_t)max_distance + 1));
    if (!local_layer || !local_layer_copy || !attenuation) {
        fprintf(stderr, "Errore allocazione memoria sul processo %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const float threshold = THRESHOLD / layer_size;
    #pragma omp parallel for schedule(static)
    for (k = 0; k <= max_distance; k++)
        attenuation[k] = sqrtf((float)(k + 1));

    #pragma omp parallel for schedule(static)
    for( k = 0; k < alloc_size; k++ ) {
        local_layer[k] = 0.0f;
        local_layer_copy[k] = 0.0f;
    }

    int left_neighbor  = (local_size == 0 || rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_neighbor = (local_size == 0 || start_index + local_size == layer_size)
                         ? MPI_PROC_NULL : rank + 1;
    MPI_Request rq[4];

    /* Riduco il blocco se il processo ha poche celle rispetto ai thread. */
    int block_size = local_size / omp_get_max_threads();
    if (block_size < 1) block_size = 1;
    if (block_size > CELL_BLOCK) block_size = CELL_BLOCK;

    for( i = 0; i < num_storms; i++ ) {

        /* 4.1. Ogni thread lavora su blocchi piccoli, che restano in cache.
           Le particelle restano in ordine; SIMD lavora su celle indipendenti. */
        #pragma omp parallel for schedule(static)
        for (int first = 1; first <= local_size; first += block_size) {
            int end = first + block_size;
            if (end > local_size + 1) end = local_size + 1;
            for (int j = 0; j < storms[i].size; j++) {
                float energy = ((float)storms[i].posval[j*2+1] * 1000) / layer_size;
                int position = storms[i].posval[j*2];
                #pragma omp simd
                for (int cell = first; cell < end; cell++) {
                    int k_glob = start_index + cell - 1;
                    update(local_layer, attenuation, threshold, cell, k_glob, position, energy);
                }
            }
        }

        /* 4.2.1. Copia dei valori aggiornati */
        #pragma omp parallel for schedule(static)
        for( k = 1; k <= local_size; k++ ) 
            local_layer_copy[k] = local_layer[k];

        /* 4.2.2. Scambio i bordi e aspetto che le ghost cells siano pronte. */
        MPI_Irecv(&local_layer_copy[local_size + 1], 1, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD, &rq[0]);
        MPI_Irecv(&local_layer_copy[0],              1, MPI_FLOAT, left_neighbor,  1, MPI_COMM_WORLD, &rq[1]);
        MPI_Isend(&local_layer_copy[1],              1, MPI_FLOAT, left_neighbor,  0, MPI_COMM_WORLD, &rq[2]);
        MPI_Isend(&local_layer_copy[local_size],     1, MPI_FLOAT, right_neighbor, 1, MPI_COMM_WORLD, &rq[3]);

        MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);

        /* 4.2.3. Rilassamento */
        #pragma omp parallel for schedule(static)
        for( k = 1; k <= local_size; k++ ) {
            int k_glob = start_index + (k - 1);
            if ( k_glob > 0 && k_glob < layer_size - 1 ) {
                local_layer[k] = ( local_layer_copy[k-1] + local_layer_copy[k] + local_layer_copy[k+1] ) / 3.0f;
            }
        }

        /* Copia per il secondo scambio del massimo locale */
        #pragma omp parallel for schedule(static)
        for( k = 1; k <= local_size; k++ ) 
            local_layer_copy[k] = local_layer[k];

        /* Secondo scambio asincrono */
        MPI_Irecv(&local_layer_copy[local_size + 1], 1, MPI_FLOAT, right_neighbor, 2, MPI_COMM_WORLD, &rq[0]);
        MPI_Irecv(&local_layer_copy[0],              1, MPI_FLOAT, left_neighbor,  3, MPI_COMM_WORLD, &rq[1]);
        MPI_Isend(&local_layer_copy[1],              1, MPI_FLOAT, left_neighbor,  2, MPI_COMM_WORLD, &rq[2]);
        MPI_Isend(&local_layer_copy[local_size],     1, MPI_FLOAT, right_neighbor, 3, MPI_COMM_WORLD, &rq[3]);

        MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);

        /* 4.3. Ricerca del Massimo Locale */
        struct {
            float val;
            int   rank;
        } local_max_pair, global_max_pair;

        /* Come nel sequenziale: se non ci sono picchi positivi, lascio 0. */
        local_max_pair.val = 0.0f;
        local_max_pair.rank = rank;
        int local_best_pos = 0;

        #pragma omp parallel
        {
            float thread_max = 0.0f;
            int thread_pos = -1;

            #pragma omp for nowait schedule(static)
            for( k = 1; k <= local_size; k++ ) {
                int k_glob = start_index + (k - 1);

                if ( k_glob > 0 && k_glob < layer_size - 1 ) {
                    if ( local_layer[k] > local_layer_copy[k-1] && local_layer[k] > local_layer_copy[k+1] ) {
                        if ( local_layer[k] > thread_max ) {
                            thread_max = local_layer[k];
                            thread_pos = k_glob;
                        }
                    }
                }
            }

            #pragma omp critical(local_max_update)
            {
                if ( thread_max > local_max_pair.val ||
                     (thread_max == local_max_pair.val && thread_pos >= 0 &&
                      (local_best_pos < 0 || thread_pos < local_best_pos)) ) {
                    local_max_pair.val = thread_max;
                    local_best_pos = thread_pos;
                }
            }
        }

        MPI_Allreduce(&local_max_pair, &global_max_pair, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        maximum[i] = global_max_pair.val;

        if ( rank == global_max_pair.rank ) {
            positions[i] = local_best_pos;
        }

        MPI_Bcast(&positions[i], 1, MPI_INT, global_max_pair.rank, MPI_COMM_WORLD);
    }

    free(local_layer);
    free(local_layer_copy);
    free(attenuation);
}
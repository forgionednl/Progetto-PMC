#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "energy_storms.h"

#define BLOCK_SIZE 256

/* Funzione dispositivo CUDA per calcolare l'aggiornamento di una cella k */
__device__ inline void update_gpu(float *layer, int layer_size, int k, int pos, float energy) {
    int distance = pos - k;
    if (distance < 0) distance = -distance;
    distance = distance + 1;

    float atenuacion = sqrtf((float)distance);
    float energy_k = energy / layer_size / atenuacion;

    if (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size) {
        layer[k] += energy_k;
    }
}

/* KERNEL 1: Calcola l'impatto delle particelle della tempesta corrente sulle celle */
__global__ void kernel_storm_impact(float *d_layer, int layer_size, const int *d_posval, int num_particles) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < layer_size) {
        for (int j = 0; j < num_particles; j++) {
            int position = d_posval[j * 2];
            float energy = (float)d_posval[j * 2 + 1] * 1000.0f;
            update_gpu(d_layer, layer_size, k, position, energy);
        }
    }
}

/* KERNEL 2: Rilassamento/Media mobile per le celle interne */
__global__ void kernel_relax(const float *d_layer_copy, float *d_layer, int layer_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= 1 && k < layer_size - 1) {
        d_layer[k] = (d_layer_copy[k - 1] + d_layer_copy[k] + d_layer_copy[k + 1]) / 3.0f;
    }
}

/* FUNZIONE PRINCIPALE EXPORTATA */
/* extern "C" */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    float *d_layer = NULL;
    float *d_layer_copy = NULL;

    // 1. Allocazione memoria GPU per il layer
    cudaMalloc((void **)&d_layer, sizeof(float) * layer_size);
    cudaMalloc((void **)&d_layer_copy, sizeof(float) * layer_size);

    // Inizializzazione a 0.0f
    cudaMemset(d_layer, 0, sizeof(float) * layer_size);
    cudaMemset(d_layer_copy, 0, sizeof(float) * layer_size);

    // Allocazione temporanea host per leggere i risultati
    float *h_layer = (float *)malloc(sizeof(float) * layer_size);

    int grid_size = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 2. Loop sulle tempeste
    for (int i = 0; i < num_storms; i++) {
        int num_particles = storms[i].size;
        int *d_posval = NULL;

        if (num_particles > 0) {
            // Alloca e copia le particelle della tempesta corrente su GPU
            cudaMalloc((void **)&d_posval, sizeof(int) * num_particles * 2);
            cudaMemcpy(d_posval, storms[i].posval, sizeof(int) * num_particles * 2, cudaMemcpyHostToDevice);

            // Launch Kernel 1: Impatto
            kernel_storm_impact<<<grid_size, BLOCK_SIZE>>>(d_layer, layer_size, d_posval, num_particles);
            cudaDeviceSynchronize();

            cudaFree(d_posval);
        }

        // 3. Rilassamento (Copia layer -> layer_copy, poi esegui kernel_relax)
        cudaMemcpy(d_layer_copy, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice);
        kernel_relax<<<grid_size, BLOCK_SIZE>>>(d_layer_copy, d_layer, layer_size);
        cudaDeviceSynchronize();

        // 4. Trasferimento risultati su host per ricerca del massimo
        cudaMemcpy(h_layer, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToHost);

        maximum[i] = 0.0f;
        positions[i] = 0;

        for (int k = 1; k < layer_size - 1; k++) {
            if (h_layer[k] > h_layer[k - 1] && h_layer[k] > h_layer[k + 1]) {
                if (h_layer[k] > maximum[i]) {
                    maximum[i] = h_layer[k];
                    positions[i] = k;
                }
            }
        }
    }

    // 5. Deallocazione
    cudaFree(d_layer);
    cudaFree(d_layer_copy);
    free(h_layer);
}
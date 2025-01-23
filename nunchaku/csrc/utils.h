#pragma once

#include "common.h"
#include "Tensor.h"

namespace nunchaku::utils {

    void disable_memory_auto_release() {
        int device;
        checkCUDA(cudaGetDevice(&device));
        cudaMemPool_t mempool;
        checkCUDA(cudaDeviceGetDefaultMemPool(&mempool, device));
        uint64_t threshold = UINT64_MAX;
        checkCUDA(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }

    void trim_memory() {
        int device;
        checkCUDA(cudaGetDevice(&device));
        cudaMemPool_t mempool;
        checkCUDA(cudaDeviceGetDefaultMemPool(&mempool, device));
        size_t bytesToKeep = 0;
        checkCUDA(cudaMemPoolTrimTo(mempool, bytesToKeep));
    }

};
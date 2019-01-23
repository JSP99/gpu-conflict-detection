#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cstdio>
#include <iostream>
#include "cudaglobals.h"

#define CUDA_WARN(XXX) \
        do { if (XXX != cudaSuccess) std::cerr << "CUDA Error: " << \
                             cudaGetErrorString(XXX) << ", at line " << __LINE__ \
                                               << std::endl; } while (0)

__global__ void detconflictKernel
(
        tr_event* concat_tracklists, // Sorted section event lists
        int* directions, // Train directions
        sec_attribs* section_attr, // Section attributes
        int2* conflicts, // Conflicts detected on the GPU, in global memory
        const int numb_trains,
        const int numb_sections
)
{
        uint3 bid = blockIdx; // ID of the block

        // 1D grid and 1D blocks
        auto threadsPerBlock = blockDim.x; // Number of threads per block
        auto l = threadIdx.x; // Local thread number in the block
        auto blockNumInGrid = blockIdx.x; // Block number in the grid
        auto i = blockNumInGrid * threadsPerBlock + l; // Global thread number

        /*
         * Shared memory. Make sure that these static arrays are sufficient.
         * The size of shared memory array for concatenated track lists should be
         * > (number of threads in the block + 1)
         */
        __shared__ tr_event sh_concat_tracklists[1025];
        __shared__ int sh_directions[128]; // Train directions
        __shared__ sec_attribs sh_section_attr[128]; // Section attributes

        // Private variable (per GPU thread) to record the conflict event
        int2 conflict;
        conflict.x = -1;
        conflict.y = -1;

        int numb_directions = sizeof(int) * numb_trains;
        int numb_section_attr = sizeof(sec_attribs) * numb_sections;

        /* Allow only certain threads in a block to copy the section attributes and
         * train directions to the shared memory of the block.
         */
        if (l < numb_sections) {
                sh_section_attr[l] = section_attr[l];
        }
        if (l < numb_trains) {
                sh_directions[l] = directions[l];
        }

        // Initiate a copy of a block of sorted section event lists from global to shared memory.
        sh_concat_tracklists[l] = concat_tracklists[i];
        // For the last thread in the block
        if (l == threadsPerBlock - 1) {
                sh_concat_tracklists[l + 1] = concat_tracklists[i + 1];
        }

        // Synchronization point: Ensure all writes to shared memory are completed.
        __syncthreads();

        const int section1 = sh_concat_tracklists[l].sectionid;
        const int section2 = sh_concat_tracklists[l + 1].sectionid;

        const int track1 = sh_concat_tracklists[l].trackid;
        const int track2 = sh_concat_tracklists[l + 1].trackid;

        const int firstevent = sh_concat_tracklists[l].eventid;
        const int nextevent = sh_concat_tracklists[l + 1].eventid;

        const int firsttrain = sh_concat_tracklists[l].trainid;
        const int nexttrain = sh_concat_tracklists[l + 1].trainid;

        // If events belong to the same section and the same track,
        if ((section1 == section2) && (track1 == track2))
        {
                // If the consecutive trains are in the same direction,
                if ((sh_directions[firsttrain] == sh_directions[nexttrain])
                    && // and if the track is part of a line section,
                    (sh_section_attr[section1].sec_type == 1)
                    && // and if the line section has more than 1 block section,
                    (sh_section_attr[section1].num_blocks > 1))
                {
                        // If Headway constraint (encoded as 0) is violated.
                        if ((sh_concat_tracklists[l + 1].begin - sh_concat_tracklists[l].begin
                             < sh_section_attr[section1].headway_time)
                            ||
                            (sh_concat_tracklists[l + 1].end - sh_concat_tracklists[l].end
                             < sh_section_attr[section1].headway_time))
                        {
                                // Events with global index i and i+1 are conflicting each other.
                                conflict.x = i;
                                conflict.y = 0;
                        }
                }
                // Else, the consecutive trains are in the opposite direction or they are
                // on a section with only one block section.
                else
                {
                        // If clear time constraint (encoded as 1) is violated
                        if (sh_concat_tracklists[l + 1].begin - sh_concat_tracklists[l].end
                            < sh_section_attr[section1].clear_time)
                        {
                                // Events with global index i and i+1 are conflicting each other.
                                conflict.x = i;
                                conflict.y = 1;
                        }
                }
        }
        // Coalesced copy the detected conflict to global memory
        conflicts[i] = conflict;
}

// Helper function for using CUDA to detect conflicts in parallel
cudaError_t detect_conflicts_with_cuda(
        tr_event *src_seclists_chunk,
        const int *h_directions,
        const sec_attribs* h_section_attribs,
        int2* detected_conf,
        int numb_events,
        int numb_trains,
        int numb_sections,
        int threads_per_block)
{
        static tr_event* dev_concat_tracklists; // Concatenated track event lists
        static int* dev_directions; // Train directions
        static sec_attribs* dev_section_attr; // Section attributes
        static int2* dev_conflicts; // Conflicts detected on the GPU, in global memory

        cudaError_t cudaStatus;

        // Variable to keep track of first call to this function
        static int first_call = true;

        for (int i = 0; i < 10000; i++)
        {
                if (i % 1000 == 0)
                        std::cout << i << " ";
                // Allocate GPU buffers for input and output data structures
                // INPUT
                if (first_call)
                {
                        // Choose which GPU to run on, change this on a multi-GPU system.
                        CUDA_WARN(cudaSetDevice(0));
                        CUDA_WARN(cudaMalloc((void**)&dev_concat_tracklists,
                                             numb_events * sizeof(tr_event)));
                        CUDA_WARN(cudaMalloc((void**)&dev_directions,
                                             numb_trains * sizeof(int)));
                        CUDA_WARN(cudaMalloc((void**)&dev_section_attr,
                                             numb_sections * sizeof(sec_attribs)));
                        // Allocate memory for output
                        CUDA_WARN(cudaMalloc((void**)&dev_conflicts,
                                             numb_events * sizeof(int2)));
                        // Copy the "constant" data only once
                        CUDA_WARN(cudaMemcpy(dev_directions, h_directions,
                                             numb_trains * sizeof(int),
                                             cudaMemcpyHostToDevice));
                        CUDA_WARN(cudaMemcpy(dev_section_attr, h_section_attribs,
                                             numb_sections * sizeof(sec_attribs),
                                             cudaMemcpyHostToDevice));
                        first_call = false;
                }

                // Copy from host data structures to the GPU data structures.
                CUDA_WARN(cudaMemcpy(dev_concat_tracklists, src_seclists_chunk,
                                     numb_events * sizeof(tr_event), cudaMemcpyHostToDevice));

                int numb_blocks = numb_events / threads_per_block;

                // Launch a kernel on the GPU with one thread for each element.
                detconflictKernel << <numb_blocks, threads_per_block >> > (
                        dev_concat_tracklists, dev_directions,
                        dev_section_attr, dev_conflicts,
                        numb_trains, numb_sections);

                // Check for any errors launching the kernel
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "detconflictKernel launch failed: %s\n",
                                cudaGetErrorString(cudaStatus));
                }

                // Copy output vector from GPU buffer to host memory.
                CUDA_WARN(cudaMemcpy(detected_conf, dev_conflicts,
                                     numb_events * sizeof(int2), cudaMemcpyDeviceToHost));
        }
        return cudaStatus;
}

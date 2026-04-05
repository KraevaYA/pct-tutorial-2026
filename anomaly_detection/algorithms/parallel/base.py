from abc import ABC, abstractmethod
import math
import ctypes
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from numba import cuda, float32, int32

from anomaly_detection.algorithms.parallel.config import *
from anomaly_detection.utils import find_non_overlap_discords


class ParallelDiscordDetector(ABC):
    """
    Base class for parallel time series anomaly detection algorithms on GPU,
    which includes three phases: preprocessing, candidate selection, and discord refinement.
    """

    @staticmethod
    @cuda.jit
    def gpu_compute_statistics(d_T, d_mean, d_std, N, m):
        """
        A Numba CUDA kernel to calculate mean values and standard deviations 
        of all time series subsequences with the minimal length minL from the specified length range.

        Parameters
        ----------
        d_T : numpy.ndarray
            The time series.

        d_mean : numpy.ndarray
            The array including means of time series subsequences.

        d_std : numpy.ndarray
            The array including standard deviations of time series subsequences.

        N : int
            The number of subsequences in time series.

        m : int
            The discord length.

        Returns
        -------
        None
        """

        tid = cuda.threadIdx.x
        thread_idx = (cuda.blockIdx.x*cuda.blockDim.x)+tid

        while (thread_idx < N):
            mean = 0
            std = 0

            for i in range(m):
                mean = mean + d_T[thread_idx+i]
                std = std + d_T[thread_idx+i]*d_T[thread_idx+i]

            std = std/m
            mean = mean/m

            d_mean[thread_idx] = mean
            d_std[thread_idx] = math.sqrt(std - mean*mean)

            thread_idx = thread_idx + cuda.blockDim.x*cuda.gridDim.x


    @staticmethod
    @cuda.jit
    def gpu_candidates_select(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, squared_r, m):
        """
        A Numba CUDA kernel to select candidates with length m, where it collects potential discords.

        Parameters
        ----------
        d_T : numpy.ndarray
            The time series.

        d_mean : numpy.ndarray
            The array including means of time series subsequences.

        d_std : numpy.ndarray
            The array including standard deviations of time series subsequences.

        d_cand : numpy.ndarray
            The boolean-valued array where the ith element is TRUE if the subsequence T_{i,m} is a potential discord, and FALSE otherwise.
            This array is initialized with TRUE values.

        d_neighbor : numpy.ndarray 
            The boolean-valued array where the ith element is TRUE if the nearest neighbor T_{i,m} is not pruned, and FALSE otherwise.
            This array is initialized with TRUE values.

        d_nnDist : numpy.ndarray
            The array of distances, where an element is a distance from a correspondent subsequence to its nearest neighbor
            if the subsequence is a candidate discord and +inf otherwise.

        N : int
            The number of subsequences with length m in time series.

        squared_r : float
            The squared distance threshold.

        m : int
            The discord length.

        Returns
        -------
        None
        """

        tid = cuda.threadIdx.x
        blockSize = cuda.blockDim.x
        segment_ind = cuda.blockIdx.x*blockSize
        chunk_ind = segment_ind + m - 1
        nnDist = FLOAT_MAX
        min_nnDist = FLOAT_MAX
        non_overlap = 0

        segment = cuda.shared.array(shape=0, dtype=float32)[0:SEGMENT_N+m-1]
        chunk = cuda.shared.array(shape=0, dtype=float32)[SEGMENT_N+m-1:2*(SEGMENT_N+m-1)]

        cand = cuda.shared.array(shape=SEGMENT_N, dtype=int32)
        dot_col = cuda.shared.array(shape=SEGMENT_N, dtype=float32)
        dot_row = cuda.shared.array(shape=SEGMENT_N, dtype=float32)
        dot_inter = cuda.shared.array(shape=SEGMENT_N, dtype=float32)
        all_rej = cuda.shared.array(shape=1, dtype=int32)

        cand[tid] = 1

        if (tid == 0):
            all_rej[0] = 1

        ind = tid
        segment_len = SEGMENT_N+m-1

        if (segment_ind+tid < N):
            d_neighbor[segment_ind+tid] = 1
            d_cand[segment_ind+tid] = 1
            d_nnDist[segment_ind+tid] = FLOAT_MAX

        while (ind < segment_len):
            segment[ind] = d_T[segment_ind+ind]
            chunk[ind] = d_T[chunk_ind+ind]
            ind = ind + blockSize

        dot_col[tid] = 0
        dot_row[tid] = 0

        cuda.syncthreads()

        # calculate dot for the first column and row (the first chunk)
        for j in range(m):
            dot_col[tid] = dot_col[tid] + segment[j]*chunk[j+tid]
            dot_row[tid] = dot_row[tid] + segment[j+tid]*chunk[j]

        cuda.syncthreads()

        nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]))

        if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
            nnDist = 2.0*m

        non_overlap = 0 if (abs(segment_ind+tid-chunk_ind) < (m-1)) else 1

        if (non_overlap):
            if (nnDist < squared_r):
                cand[tid] = 0
                cuda.atomic.min(d_neighbor, chunk_ind, 0)
            else:
                min_nnDist = min(min_nnDist, nnDist)

        # calculate dot for rows from second to last (the first chunk)
        for j in range(blockSize-1):
            if (tid > 0):
                dot_inter[tid] = dot_row[tid-1]

            cuda.syncthreads()

            if (tid > 0):
                dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j]

            cuda.syncthreads()

            if (tid == 0):
                dot_row[tid] = dot_col[j+1]

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]))

            if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                nnDist = 2.0*m

            non_overlap = 0 if (abs(segment_ind+tid-chunk_ind-j-1) < (m-1)) else 1

            if (non_overlap):
                if (nnDist < squared_r):
                    cand[tid] = 0
                    cuda.atomic.min(d_neighbor, chunk_ind+j+1, 0)
                else:
                    min_nnDist = min(min_nnDist, nnDist)

        cuda.syncthreads()

        if (tid == 0):
            all_rej[0] = 0
            for k in range(blockSize):
                if (cand[k] == 1):
                    all_rej[0] = cand[k]
                    break

        cuda.syncthreads()

        chunk_ind = chunk_ind + blockSize

        # process chunks from the second to last
        while ((chunk_ind < N) and (all_rej[0] != 0)):
            dot_col[tid] = 0
            ind = tid

            while (ind < segment_len):
                chunk[ind] = d_T[chunk_ind+ind]
                ind = ind + blockSize

            cuda.syncthreads()

            for j in range(m):
                dot_col[tid] += segment[j]*chunk[j+tid]

            if (tid > 0):
                dot_inter[tid] = dot_row[tid-1]

            cuda.syncthreads()

            if (tid > 0):
                dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m-1] - segment[tid-1]*d_T[chunk_ind-1]
            else:
                dot_row[tid] = dot_col[0]

            cuda.syncthreads()

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]))

            if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                nnDist = 2.0*m

            if (cand[tid] != 0):
                if (nnDist < squared_r):
                    cand[tid] = 0
                    cuda.atomic.min(d_neighbor, chunk_ind, 0)
                else:
                    min_nnDist = min(min_nnDist, nnDist)

            for j in range(blockSize-1):
                if (tid > 0):
                    dot_inter[tid] = dot_row[tid-1]

                cuda.syncthreads()

                if (tid > 0):
                    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j]

                if (tid == 0):
                    dot_row[tid] = dot_col[j+1]

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]))

                if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                    nnDist = 2.0*m

                if (cand[tid] != 0):
                    if (nnDist < squared_r):
                        cand[tid] = 0
                        cuda.atomic.min(d_neighbor, chunk_ind+j+1, 0)
                    else:
                        min_nnDist = min(min_nnDist, nnDist)

            cuda.syncthreads()

            if (tid == 0):
                all_rej[0] = 0
                for k in range(blockSize):
                    if (cand[k] == 1):
                        all_rej[0] = cand[k]
                        break

            cuda.syncthreads()

            chunk_ind = chunk_ind + blockSize

        if (segment_ind+tid < N):
            d_cand[segment_ind+tid] = cand[tid]
            d_nnDist[segment_ind+tid] = min_nnDist


    @staticmethod
    @cuda.jit
    def gpu_candidates_clarify(d_cand, d_neighbor, N):
        """
        A Numba CUDA kernel to clarify the potential discords.

        Parameters
        ----------
        d_cand : numpy.ndarray
            The boolean-valued array where the ith element is TRUE if the subsequence T_{i,m} is a potential discord, and FALSE otherwise.
            This array is initialized with TRUE values.

        d_neighbor : numpy.ndarray 
            The boolean-valued array where the ith element is TRUE if the nearest neighbor T_{i,m} is not pruned, and FALSE otherwise.
            This array is initialized with TRUE values.

        N : int
            The number of subsequences with length m in time series.

        Returns
        -------
        None
        """

        tid = cuda.threadIdx.x
        thread_id = cuda.blockIdx.x*cuda.blockDim.x+tid

        if (thread_id < N):
            d_cand[thread_id] = d_cand[thread_id] * d_neighbor[thread_id]
            thread_id = thread_id + cuda.blockDim.x*cuda.gridDim.x


    @staticmethod
    @cuda.jit
    def gpu_discords_refine(d_T, d_mean, d_std, d_cand, d_nnDist, N, m, r):
        """
        A Numba CUDA kernel for refinement of discord with length m, where it discards false positives discords.

        Parameters
        ----------
        d_T : numpy.ndarray
            The time series.

        d_mean : numpy.ndarray
            The array including means of time series subsequences.

        d_std : numpy.ndarray
            The array including standard deviations of time series subsequences.

        d_cand : numpy.ndarray
            The boolean-valued array where the ith element is TRUE if the subsequence T_{i,m} is a potential discord, and FALSE otherwise.
            This array is initialized with TRUE values.

        d_nnDist : numpy.ndarray
            The array of distances, where an element is a distance from a correspondent subsequence to its nearest neighbor
            if the subsequence is a candidate discord and +inf otherwise.

        N : int
            The number of subsequences with length m in time series.

        m : int
            The discord length.

        r : float
            The squared distance threshold.

        Returns
        -------
        None
        """

        tid = cuda.threadIdx.x
        blockSize = cuda.blockDim.x
        segment_ind = cuda.blockIdx.x*blockSize
        chunk_ind = 0
        nnDist = FLOAT_MAX
        min_nnDist = d_nnDist[segment_ind+tid]
        non_overlap = 0
        ind = tid
        step = 0
        segment_len = SEGMENT_N+m-1

        segment = cuda.shared.array(shape=0, dtype=float32)[0:SEGMENT_N+m-1]
        chunk = cuda.shared.array(shape=0, dtype=float32)[SEGMENT_N+m-1:2*(SEGMENT_N+m-1)]

        cand = cuda.shared.array(shape=SEGMENT_N, dtype=int32)
        dot_col = cuda.shared.array(shape=SEGMENT_N, dtype=float32)
        dot_row = cuda.shared.array(shape=SEGMENT_N, dtype=float32)
        dot_inter = cuda.shared.array(shape=SEGMENT_N, dtype=float32)
        all_rej = cuda.shared.array(shape=1, dtype=int32)

        cand[tid] = d_cand[segment_ind+tid]

        cuda.syncthreads()

        if (tid == 0):
            all_rej[0] = 0
            for k in range(blockSize):
                if (cand[k] == 1):
                    all_rej[0] = cand[k]
                    break

        cuda.syncthreads()

        if (all_rej[0] != 0):
            while (ind < segment_len):
                segment[ind] = d_T[segment_ind+ind]
                ind = ind + blockSize

            while ((chunk_ind < segment_ind-blockSize) and (all_rej[0] != 0)):
                dot_col[tid] = 0
                ind = tid

                while (ind < segment_len):
                    chunk[ind] = d_T[chunk_ind+ind]
                    ind = ind + blockSize

                cuda.syncthreads()

                if (step == 0):
                    dot_row[tid] = 0

                    for j in range(m):
                        dot_col[tid] = dot_col[tid] + segment[j]*chunk[j+tid]
                        dot_row[tid] = dot_row[tid] + segment[j+tid]*chunk[j]
                else:
                    for j in range(m):
                        dot_col[tid] = dot_col[tid] + segment[j]*chunk[j+tid]

                    if (tid > 0):
                        dot_inter[tid] = dot_row[tid-1]

                    cuda.syncthreads()

                    if (tid > 0):
                        dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m-1] - segment[tid-1]*d_T[chunk_ind-1]
                    else:
                        dot_row[tid] = dot_col[0]

                cuda.syncthreads()

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]))

                if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                    nnDist = 2.0*m

                if (cand[tid] != 0):
                    if (nnDist < r):
                        cand[tid] = 0
                        min_nnDist = -FLOAT_MAX
                    else:
                        min_nnDist = min(min_nnDist, nnDist)

                for j in range(blockSize-1):
                    if (tid > 0):
                        dot_inter[tid] = dot_row[tid-1]

                    cuda.syncthreads()

                    if (tid > 0):
                        dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j]

                    cuda.syncthreads()

                    if (tid == 0):
                        dot_row[tid] = dot_col[j+1]

                    nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]))

                    if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                        nnDist = 2.0*m

                    if (cand[tid] != 0):
                        if (nnDist < r):
                            cand[tid] = 0
                            min_nnDist = -FLOAT_MAX
                        else:
                            min_nnDist = min(min_nnDist, nnDist)

                cuda.syncthreads()

                if (tid == 0):
                    all_rej[0] = 0
                    for k in range(blockSize):
                        if (cand[k] == 1):
                            all_rej[0] = cand[k]
                            break

                cuda.syncthreads()

                chunk_ind = chunk_ind + blockSize
                step = step + 1

            while ((chunk_ind < segment_ind) and (all_rej[0] != 0)):
                dot_row[tid] = 0
                dot_col[tid] = 0
                ind = tid

                while (ind < segment_len):
                    chunk[ind] = d_T[chunk_ind+ind]
                    ind = ind + blockSize

                cuda.syncthreads()

                for j in range(m):
                    dot_col[tid] = dot_col[tid] + segment[j]*chunk[j+tid]
                    dot_row[tid] = dot_row[tid] + segment[j+tid]*chunk[j]

                cuda.syncthreads()

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]))

                if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                    nnDist = 2.0*m

                non_overlap = 0 if (abs(segment_ind+tid-chunk_ind) < (m-1)) else 1

                if ((non_overlap) and (cand[tid] != 0)):
                    if (nnDist < r):
                        cand[tid] = 0
                        min_nnDist = -FLOAT_MAX
                    else:
                        min_nnDist = min(min_nnDist, nnDist)

                for j in range(blockSize-1):
                    if (tid > 0):
                        dot_inter[tid] = dot_row[tid-1]

                    cuda.syncthreads()

                    if (tid > 0):
                        dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j]

                    cuda.syncthreads()

                    if (tid == 0):
                        dot_row[tid] = dot_col[j+1]

                    nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]))

                    if (math.isinf(abs(nnDist)) or (nnDist < 0) or (nnDist >= FLOAT_MAX)):
                        nnDist = 2.0*m

                    non_overlap = 0 if (abs(segment_ind+tid-chunk_ind-j-1) < (m-1)) else 1

                    if ((non_overlap) and (cand[tid] != 0)):
                        if (nnDist < r):
                            cand[tid] = 0
                            min_nnDist = -FLOAT_MAX
                        else:
                            min_nnDist = min(min_nnDist, nnDist)

                cuda.syncthreads()

                if (tid == 0):
                    all_rej[0] = 0
                    for k in range(blockSize):
                        if (cand[k] == 1):
                            all_rej[0] = cand[k]
                            break

                cuda.syncthreads()

                chunk_ind = chunk_ind + blockSize

            if (segment_ind+tid < N):
                d_cand[segment_ind+tid] = d_cand[segment_ind+tid] * cand[tid]
                d_nnDist[segment_ind+tid] = min_nnDist


    def _define_N_pad(self, n: int, min_m: int, max_m: int) -> int:
        """
        Calculate the maximum number of subsequences with lengths 
        from the specified length range in the padded time series 

        Parameters
        ----------
        n: int 
            The time series length.
        
        min_m : int
            The minimum length of discords.

        max_m : int
            The maximum length of discords.

        Returns
        -------
        pad_N: int 
            The maximum number of subsequences with lengths from the specified length range 
            in the padded time series.
        """
        
        pad_N = 0
        pad = np.ceil((n - min_m + 1)/SEGMENT_N)*SEGMENT_N + 2*min_m - 2 - n
        pad_per_segment = np.ceil(pad/SEGMENT_N)
        delta_m = pad_per_segment*SEGMENT_N - pad - 2
        m_N_max = delta_m + min_m

        if (pad_per_segment == 1):
            m_N_max = m_N_max + min_m

        if (m_N_max > max_m):
            m_N_max = max_m

        pad_N = np.ceil((n-m_N_max+1)/SEGMENT_N)*SEGMENT_N + 2*m_N_max - 2 - m_N_max + 1

        return int(pad_N)
        

    def _cuda_compute_statistics(self, d_T, d_mean, d_std, N, m):
        """
        Call the Numba CUDA kernel to calculate mean values and standard deviations 
        of all time series subsequences with length minL.
        
        Parameters
        ----------
        d_T : numpy.ndarray
            The time series.

        d_mean : numpy.ndarray
            The array including means of time series subsequences.

        d_std : numpy.ndarray
            The array including standard deviations of time series subsequences.

        N : int
            The number of subsequences with length m in time series.

        m : int
            The discord length.

        Returns
        -------
        None
        """

        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = math.ceil(N/THREADS_PER_BLOCK)

        self.gpu_compute_statistics[blocks_per_grid, threads_per_block](d_T, d_mean, d_std, N, m)


    def _perform_phases(self, d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, m, w, squared_r, max_N):
        """
        Call the Numba CUDA kernels that perform phases of PD3 algorithm: candidate selection and discord refinement, respectively.

        Parameters
        ----------
        d_T : numpy.ndarray
            The time series.

        d_mean : numpy.ndarray
            The array including means of time series subsequences.

        d_std : numpy.ndarray
            The array including standard deviations of time series subsequences.

        d_cand : numpy.ndarray
            The boolean-valued array where the ith element is TRUE if the subsequence T_{i,m} is a potential discord, and FALSE otherwise.
            This array is initialized with TRUE values.

        d_neighbor : numpy.ndarray 
            The boolean-valued array where the ith element is TRUE if the nearest neighbor T_{i,m} is not pruned, and FALSE otherwise.
            This array is initialized with TRUE values.

        d_nnDist : numpy.ndarray
            The array of distances, where an element is a distance from a correspondent subsequence to its nearest neighbor
            if the subsequence is a candidate discord and +inf otherwise.

        N : int
            The number of subsequences with length m in time series.

        m : int
            The discord length.

        w : int
            The window size.

        squared_r : float
            The squered distance threshold.

        max_N : int
            The number of subsequences with length minL in the padded time series.

        Returns
        ---------
        phases_result : dict
            Results obtained from the implementation of phases 
            (indices, distances to its nearest neighbor, matrix profile, and candidate count).

        times : list
            Runtimes of phases in ms. 
        """

        start = cuda.event()
        stop = cuda.event()

        # Phase 1. Candidate Selection Algorithm
        blocks_per_grid = math.ceil(N/SEGMENT_N)
        threads_per_block = SEGMENT_N
        N_pad = blocks_per_grid*SEGMENT_N + m - 1

        start.record()
        self.gpu_candidates_select[blocks_per_grid, threads_per_block, 0, 2*(SEGMENT_N+m-1)*ctypes.sizeof(ctypes.c_float)](d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N_pad, squared_r, m)
        stop.record()
        stop.synchronize()
        selection_time = cuda.event_elapsed_time(start, stop)

        # Phase 1. Candidate Clarification
        blocks_per_grid = math.ceil(N_pad/THREADS_PER_BLOCK)
        threads_per_block = THREADS_PER_BLOCK

        start.record()
        self.gpu_candidates_clarify[blocks_per_grid, threads_per_block](d_cand, d_neighbor, N_pad)
        stop.record()
        stop.synchronize()
        clarification_time = cuda.event_elapsed_time(start, stop)

        selection_time += clarification_time 

        h_cand_gpu = d_cand.copy_to_host()
        h_neighbor_gpu = d_neighbor.copy_to_host()

        C_size = np.sum(h_cand_gpu[0:N])
        cand_index = np.flatnonzero(h_cand_gpu[0:N])
        #print("C_size = " + str(C_size))

        # Phase 2. Discord Refinement Algorithm
        blocks_per_grid = math.ceil((N_pad-m+1)/SEGMENT_N)
        threads_per_block = SEGMENT_N
        N_pad = blocks_per_grid*SEGMENT_N

        start.record()
        self.gpu_discords_refine[blocks_per_grid, threads_per_block, 0, 2*(SEGMENT_N+m-1)*ctypes.sizeof(ctypes.c_float)](d_T, d_mean, d_std, d_cand, d_nnDist, N_pad, m, squared_r)
        stop.record()
        stop.synchronize()
        refinement_time = cuda.event_elapsed_time(start, stop)

        h_cand_gpu = d_cand.copy_to_host()
        h_nnDist_gpu = d_nnDist.copy_to_host()

        #D_size = np.sum(h_cand_gpu[0:N])
        #print("D_size = " + str(D_size))

        discord_profile = np.array([-np.inf]*N, dtype=np.float32)
        discord_profile = np.where(h_cand_gpu[0:N] == 1, h_nnDist_gpu[0:N], discord_profile)

        discords = find_non_overlap_discords(discord_profile, w) # discords.keys() = ['indices' , 'distances']

        sqrt_discord_profile = discord_profile.copy()
        np.sqrt(sqrt_discord_profile, where=(sqrt_discord_profile >= 0), out=sqrt_discord_profile)

        phases_result = {
            'indices': discords['indices'],
            'distances': np.sqrt(discords['distances']),
            'mp': sqrt_discord_profile,
            'cand_count': C_size
        }

        times = [selection_time, refinement_time]

        return phases_result, times


    @abstractmethod
    def predict(self, ts: np.ndarray) -> dict:
        """
        Perform the discord discovery in time series on GPU. 

        Parameters
        ----------
        ts : numpy.ndarray
            The time series to detect arbitrary length discords in it.
            
        Returns
        -------
        discords : dict
            The dictionary storing discords (lengths, indices, distances to its nearest neighbor, 
            the nearest neighbors indices, and matrix profile).
        """

        pass
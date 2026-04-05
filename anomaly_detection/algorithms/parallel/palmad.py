import math
import ctypes
import numpy as np
import pandas as pd
import copy
import os
import multiprocessing as mp
from numba import cuda, float32, int32


from anomaly_detection.algorithms.parallel import ParallelDiscordDetector
from anomaly_detection.algorithms.parallel.config import *
from anomaly_detection.utils import *


class PALMAD(ParallelDiscordDetector):
    """
    PALMAD (Parallel Arbitrary Length MERLIN-based Anomaly Discovery) algorithm [1], 
    the parallel algorithm for all-length anomaly discovery in time series on a graphics processor.  
    PALMAD is based on the MERLIN serial algorithm by Nakamura et al. [2].

    Parameters
    ----------
    minL : int
        The minimum length of discords.

    maxL : int
        The maximum length of discords.

    topK : int, default = 3
        The number of discords of each length to be discovered.

    Attributes
    ----------
    metadata_ : dict
        A dictionary storing the results of executing an algorithm (for example, input algorithm parameters, 
        runtime of each phase and the algorithm as a whole, statistics for each phase 
        on the number of the found subsequences).

    References
    ----------
    [1] Zymbler M., Kraeva Y. High-Performance Time Series Anomaly Discovery on Graphics Processors.
    Mathematics. 2023. Vol. 11, no. 14. Article 3193. https://doi.org/10.3390/math11143193.
    [2] Nakamura T., Imamura M., Mercer R., Keogh E.J. MERLIN: parameter-free discovery 
    of arbitrary length anomalies in massive time series archives. ICDM 2020. pp. 1190-1195. 
    IEEE, 2020. https://doi.org/10.1109/ICDM50108.2020.00147.
    """

    def __init__(self, minL: int, maxL: int, topK: int = 3):
        super().__init__()
        self.minL = minL
        self.maxL = maxL
        self.topK = topK
        self.metadata_ = {
            "params": {"minL": self.minL, "maxL": self.maxL, "topK": self.topK},
            "total_time": 0.0,
            "phases": {
                "selection": {"time": 0.0, "count": 0},
                "refinement": {"time": 0.0, "count": 0}
            }
        }


    @staticmethod
    @cuda.jit
    def gpu_update_statistics(d_T, d_mean, d_std, N, m):
        """
        A Numba CUDA kernel to update mean values and standard deviations 
        of all time series subsequences with length from [minL+1; maxL].

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

            mean = (((m-1)/m)*d_mean[thread_idx] + d_T[thread_idx+m-1]/m)
            std = math.sqrt(((m-1)/m)*(d_std[thread_idx]*d_std[thread_idx]+(d_mean[thread_idx]-d_T[thread_idx+m-1])*(d_mean[thread_idx]-d_T[thread_idx+m-1])/m))

            d_mean[thread_idx] = mean
            d_std[thread_idx] = std

            thread_idx = thread_idx + cuda.blockDim.x*cuda.gridDim.x


    def _cuda_update_statistics(self, d_T, d_mean, d_std, N, m):
        """
        Call the Numba CUDA kernel to update mean values and standard deviations 
        of all time series subsequences with length from [minL+1; maxL].

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

        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = math.ceil(N/THREADS_PER_BLOCK)

        self.gpu_update_statistics[blocks_per_grid, threads_per_block](d_T, d_mean, d_std, N, m)


    def predict(self, ts: np.ndarray) -> dict:
        """
        Perform the parallel top-K discord discovery of each length 
        in the specified length range [minL, maxL] in time series on GPU.

        Parameters
        ----------
        ts : numpy.ndarray
            The time series to detect arbitrary length discords in it.

        Returns
        -------
        discords_set : dict
            Top-k discords (indices, distances to its nearest neighbor and distance profile) 
            for each length in a specified length range [minL, maxL].
        """

        discords_set = {}
        discords = {
            'indices': (-np.ones((self.topK))).tolist(),
            'distances': np.full((self.topK), -np.inf).tolist(),
            'mp': np.full(len(ts), -np.inf, dtype=np.float64),
            }

        times = {'preprocessing_time': 0,
                'selection_time': 0,
                'refinement_time': 0 
                }

        cand_count = 0

        statistics_set = {}

        n = len(ts)
        num_lengths = self.maxL - self.minL + 1

        N_pad = self._define_N_pad(n, self.minL, self.maxL)
        n_pad = N_pad + self.maxL

        h_T = np.pad(ts, (0, n_pad-n), 'constant')

        # create event handles
        start = cuda.event()
        stop = cuda.event()

        # copy time series from host to device memory 
        # and allocate memory for additional arrays
        d_T = cuda.to_device(h_T)
        d_mean = cuda.device_array(N_pad, dtype=np.float32)
        d_std = cuda.device_array(N_pad, dtype=np.float32)
        d_cand = cuda.device_array(N_pad, dtype=np.int32)
        d_neighbor = cuda.device_array(N_pad, dtype=np.int32)
        d_nnDist = cuda.device_array(N_pad, dtype=np.float32)

        m = self.minL
        w = self.minL
        N = n - self.minL + 1
        r = 2*np.sqrt(self.minL)

        print("Start PALMAD")
        print(f"Processing on m = {self.minL}")

        # Init mean and std for subsequences with minimum length
        start.record()
        self._cuda_compute_statistics(d_T, d_mean, d_std, N_pad, m)
        stop.record()
        stop.synchronize()
        
        preprocessing_time = cuda.event_elapsed_time(start, stop)
        times['preprocessing_time'] += preprocessing_time

        while (discords['distances'][self.topK-1] == -np.inf):
            #print(f"r = {r}")
            phases_result, phases_times = self._perform_phases(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, m, w, r**2, N)
            topK_discords_num = len(phases_result['indices'])

            times['selection_time'] += phases_times[0] 
            times['refinement_time'] += phases_times[1]

            if (topK_discords_num < self.topK):
                r*=0.5
                continue
            else:
                discords['indices'] = phases_result['indices'][0:self.topK]
                discords['distances'] = phases_result['distances'][0:self.topK]
                discords['mp'][:(n-m+1)] = phases_result['mp']
                cand_count += phases_result['cand_count']

        discords_set[str(self.minL)] = copy.deepcopy(discords)
        
        for m in range(self.minL+1, self.minL+5):

            if (m > self.maxL):
                break

            print(f"Processing on m = {m}")
            w = m
            N = n - m + 1

            discords['indices'] = (-np.ones((self.topK))).tolist()
            discords['distances'] = np.full((self.topK), -np.inf).tolist()
            discords['mp'] = np.full(len(ts), -np.inf, dtype=np.float64)

            start.record()
            self._cuda_update_statistics(d_T, d_mean, d_std, N_pad, m)
            stop.record()
            stop.synchronize()

            preprocessing_time = cuda.event_elapsed_time(start, stop)
            times['preprocessing_time'] += preprocessing_time

            r = 0.99 * discords_set[str(m-1)]['distances'][0]   

            while (discords['distances'][self.topK-1] == -np.inf):
                #print(f"r = {r}")
                phases_result, phases_times = self._perform_phases(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, m, w, r**2, N)
                topK_discords_num = len(phases_result['indices'])

                times['selection_time'] += phases_times[0] 
                times['refinement_time'] += phases_times[1]

                if (topK_discords_num < self.topK):
                    r = 0.99 * r
                    continue
                else:
                    discords['indices'] = phases_result['indices'][0:self.topK]
                    discords['distances'] = phases_result['distances'][0:self.topK]
                    discords['mp'][:(n-m+1)] = phases_result['mp']
                    cand_count += phases_result['cand_count']

            discords_set[str(m)] = copy.deepcopy(discords)


        for m in range(self.minL+5, self.maxL+1):
            
            print(f"Processing on m = {m}")
            
            w = m
            N = n - m + 1

            mu = np.mean([discords_set[str(m_i)]['distances'][0] for m_i in range(m-5, m)])
            sigma = np.std([discords_set[str(m_i)]['distances'][0] for m_i in range(m-5, m)])
            r = mu - 2 * sigma

            discords['indices'] = (-np.ones((self.topK))).tolist()
            discords['distances'] = np.full((self.topK), -np.inf).tolist()
            discords['mp'] = np.full(len(ts), -np.inf, dtype=np.float64)

            start.record()
            self._cuda_update_statistics(d_T, d_mean, d_std, N_pad, m)
            stop.record()
            stop.synchronize()
            preprocessing_time = cuda.event_elapsed_time(start, stop)
            times['preprocessing_time'] += preprocessing_time

            while (discords['distances'][self.topK-1] == -np.inf):
                #print(f"r = {r}")
                phases_result, phases_times = self._perform_phases(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, m, w, r**2, N)
                topK_discords_num = len(phases_result['indices'])
                
                times['selection_time'] += phases_times[0] 
                times['refinement_time'] += phases_times[1]

                if (topK_discords_num < self.topK):
                    r = r - sigma
                    continue
                else:
                    discords['indices'] = phases_result['indices'][0:self.topK]
                    discords['distances'] = phases_result['distances'][0:self.topK]
                    discords['mp'][:(n-m+1)] = phases_result['mp']
                    cand_count += phases_result['cand_count']
            
            discords_set[str(m)] = copy.deepcopy(discords)

        self.metadata_["total_time"] = times['preprocessing_time'] + times['selection_time'] + times['refinement_time']
        self.metadata_["phases"]["selection"] = {"time": times['selection_time'], "count": cand_count}
        self.metadata_["phases"]["refinement"] = {"time": times['refinement_time'], "count": num_lengths*self.topK}

        print("Finish PALMAD")

        return discords_set
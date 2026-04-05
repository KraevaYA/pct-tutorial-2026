import math
import ctypes
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from numba import cuda, float32, int32


from anomaly_detection.algorithms.parallel import ParallelDiscordDetector


class PD3(ParallelDiscordDetector):
    """
    PD3 (Parallel DRAG-based Discord Discovery) algorithm [1], 
    parallel algorithm for anomaly discovery in time series with a graphics processor.
    PD3 is based on the DRAG serial algorithm by Yankov et al. [2]. 
    The PD3 uses the concept of discord, which is a subsequence of the time series farthest from its nearest neighbor, 
    while the nearest neighbor of this subsequence is a non-self-match at a minimum distance away.

    Parameters
    ----------
    m : int
        The length of discords.

    r : float
        The distance threshold.

    Arguments
    ----------
    metadata_ : dict
        A dictionary storing the results of executing an algorithm (for example, input algorithm parameters, 
        runtime of each phase and the algorithm as a whole, statistics for each phase 
        on the number of the found subsequences).

    References
    ----------
    [1] Kraeva Y., Zymbler M. A Parallel Discord Discovery Algorithm for a Graphics Processor.
    Pattern Recognition and Image Analysis. 2023. Vol. 33, no. 2. P. 101–112. https://doi.org/10.1134/S1054661823020062.
    [2] Yankov D., Keogh E.J., Rebbapragada U. Disk aware discord discovery: Finding unusual 
    time series in terabyte sized datasets. ICDM 2007. pp. 381-390. IEEE, 2007. https://doi.org/10.1109/ICDM.2007.61.
    """

    def __init__(self, m: int, r: float):
        super().__init__()
        self.m = m
        self.r = r
        self.metadata_ = {
            "params": {"m": self.m, "r": self.r},
            "total_time": 0.0,
            "phases": {
                "selection": {"time": 0.0, "count": 0},
                "refinement": {"time": 0.0, "count": 0}
            }
        }

    def predict(self, ts: np.ndarray) -> dict:
        """
        Perform the range-discord discovery in time series, which includes three parallel phases on GPU: 
        preprocessing, candidate selection, and discord refinement. 

        Parameters
        ----------
        ts : numpy.ndarray
            The time series to detect arbitrary length discords in it.

        Returns
        -------
        discords : dict
            Top-k discords (indices, distances to its nearest neighbor and distance profile).
        """

        n = len(ts)
        w = self.m
        N = n - self.m + 1

        # preprocessing data
        N_pad = self._define_N_pad(n, self.m, self.m)
        n_pad = N_pad + self.m
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

        # Init mean and std for subsequences of time series
        start.record()
        self._cuda_compute_statistics(d_T, d_mean, d_std, N_pad, self.m)
        stop.record()
        stop.synchronize()

        preprocessing_time = cuda.event_elapsed_time(start, stop)

        phases_result, phases_times = self._perform_phases(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, self.m, w, self.r**2, N)
       
        discords = {
            'm': [self.m]*len(phases_result['indices']),
            'indices': phases_result['indices'],
            'distances': phases_result['distances'],
            'mp': phases_result['mp']
        }

        self.metadata_["total_time"] = preprocessing_time + phases_times[0] + phases_times[1]
        self.metadata_["phases"]["selection"] = {"time": phases_times[0], "count": phases_result['cand_count']}
        self.metadata_["phases"]["refinement"] = {"time": phases_times[1], "count": len(phases_result['indices'])}

        return discords
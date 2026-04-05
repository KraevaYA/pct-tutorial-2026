import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config

from anomaly_detection.utils import *


class STOMP:
    """
    STOMP (Scalable Time series Ordered-search Matrix Profile) algorithm [1] 
    for the top-k discord detection based on matrix profile.

    Parameters
    ----------
    m : int
        The discord length.

    top_k : int, default = 3
        The number of discords.

    exclusion_zone : int, default = 1
        The exclusion zone.

    References
    ----------
    [1] Yeh C.-C.M. et al. Matrix Profile I: All pairs similarity joins for time series: 
    A unifying view that includes motifs, discords and shapelets. ICDM 2016. pp. 1317-1322. 
    https://doi.org/10.1109/ICDM.2016.0179.
    """

    def __init__(self, m: int, top_k: int = 3, exclusion_zone: int = 1):
        self.m = m
        self.top_k = top_k
        self.exclusion_zone = exclusion_zone
        
        
    def predict(self, ts1: np.ndarray, ts2: np.ndarray = None) -> dict:
        """
        Compute the matrix profile and perform the top-k discord discovery 
        in time series based on the matrix profile.

        Parameters
        ----------
        ts1 : numpy.ndarrray
            The first time series.

        ts2 : numpy.ndarrray, default = None
            The second time series.
        
        Returns
        -------
        discords : dict
            Top-k discords (lengths, indices, distances to its nearest neighbor, 
            the nearest neighbors indices, and matrix profile).
        """
    
        discords_idx = []
        discords_dist = []
        discords_nn_idx = []
    
        ts1 = ts1.astype(np.float64)
        if (ts2 is None):
            ignore_trivial = True
        else:
            ignore_trivial = False
            ts2 = ts2.astype(np.float64)

        config.STUMPY_EXCL_ZONE_DENOM = self.exclusion_zone
        exclusion_zone = int(np.ceil(self.m/self.exclusion_zone))
        mp = stumpy.stump(ts1, self.m, ts2, ignore_trivial)

        mp_values = np.copy(mp[:, 0]).astype(np.float64)
        mp_index = mp[:, 1]
        
        for i in range(self.top_k):
            discord_idx = np.argmax(mp_values)
            discord_dist = mp_values[discord_idx]
            nn_idx = mp_index[discord_idx]

            if is_nan_inf(discord_dist):
                break

            mp_values = apply_exclusion_zone(mp_values, discord_idx, exclusion_zone, val=-np.inf)

            discords_idx.append(discord_idx)
            discords_dist.append(discord_dist)
            discords_nn_idx.append(nn_idx)

        return {
            'm': [self.m]*self.top_k, 
            'indices': discords_idx,
            'distances': discords_dist,
            'nn_indices': discords_nn_idx,
            'mp': np.copy(mp[:, 0]).astype(np.float64)    
        }
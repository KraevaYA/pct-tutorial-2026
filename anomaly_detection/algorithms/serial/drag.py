import numpy as np
from stumpy import core, config
import time
from typing import Callable


from anomaly_detection.algorithms.serial import SerialDiscordDetector


class DRAG(SerialDiscordDetector):
    """
    DRAG (Discord Range Aware Gathering) algorithm [1], a serial algorithm 
    for discovering the range-discords of the user-defined length m.
    Range-discord is the subsequence of the time series that is at least r 
    away from its nearest neighbor, while the nearest neighbor of this subsequence 
    is a non-self-match at a minimum distance away. 

    Parameters
    ----------
    m : int
        The length of discords.

    r : float
        The distance threshold.

    exclusion_zone : int, default = 1
        The exclusion zone.

    Attributes
    ----------
    metadata_ : dict
        A dictionary storing the results of executing an algorithm (for example, input algorithm parameters, 
        runtime of each phase and the algorithm as a whole, statistics for each phase 
        on the number of the found subsequences).

    References
    ----------
    [1] Yankov D., Keogh E.J., Rebbapragada U. Disk aware discord discovery: 
    Finding unusual time series in terabyte sized datasets. ICDM 2007. pp. 381-390. 
    IEEE, 2007. https://doi.org/10.1109/ICDM.2007.61.
    """

    def __init__(self, m: int, r: float, exclusion_zone: int = 1):
        self.m = m
        self.r = r
        self.exclusion_zone = exclusion_zone
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
        Perform the range-discord discovery in time series, which includes three phases: 
        preprocessing, candidate selection, and discord refinement. 

        Parameters
        ----------
        ts : numpy.ndarray
            The time series to detect arbitrary length discords in it.
            
        Returns
        -------
        discords : dict
            Top-k discords (lengths, indices, distances to its nearest neighbor, 
            the nearest neighbors indices, and matrix profile).
        """

        config.STUMPY_EXCL_ZONE_DENOM = self.exclusion_zone
        excl_zone = int(np.ceil(self.m / config.STUMPY_EXCL_ZONE_DENOM)) 

        # preprocessing time series
        start = time.time()
        T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(ts, self.m)
        preprocessing_time = (time.time() - start) * 1e3

        phases_result, phases_times = self._perform_phases(T, M_T, Σ_T, excl_zone, T_subseq_isconstant, self.m, self.r)

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
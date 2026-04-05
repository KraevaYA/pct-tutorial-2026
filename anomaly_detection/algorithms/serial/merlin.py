import numpy as np
from stumpy import core, config
import copy
import time


from anomaly_detection.algorithms.serial import SerialDiscordDetector


class MERLIN(SerialDiscordDetector):
    """
    MERLIN algorithm [1], serial algorithm for discovering the top-k discords 
    of each length in the specified length range in time series.
    MERLIN calls DRAG repeatedly and adaptively selects the parameter r.

    Parameters
    ----------
    minL : int
        The minimum length of discords.

    maxL : int
        The maximum length of discords.

    topK : int, default = 3
        The number of discords of each length to be discovered.

    exclusion_zone : int, default = 1
        The exclusion zone.
    
    Attributes
    ----------
    metadata_ : dict
        A dictionary storing the results of executing an algorithm 
        (for example, input algorithm parameters, runtime of each phase 
        and the algorithm as a whole, statistics for each phase 
        on the number of the found subsequences).

    References
    ----------
    [1] Nakamura T., Imamura M., Mercer R., Keogh E.J. MERLIN: parameter-free discovery 
    of arbitrary length anomalies in massive time series archives. ICDM 2020. pp. 1190-1195. 
    IEEE, 2020. https://doi.org/10.1109/ICDM50108.2020.00147.
    """

    def __init__(self, minL: int, maxL: int, topK: int = 3, exclusion_zone: int = 1):
        super().__init__()
        self.minL = minL
        self.maxL = maxL
        self.topK = topK
        self.exclusion_zone = exclusion_zone
        self.metadata_ = {
            "params": {"minL": self.minL, "maxL": self.maxL, "topK": self.topK},
            "total_time": 0.0,
            "phases": {
                "selection": {"time": 0.0, "count": 0},
                "refinement": {"time": 0.0, "count": 0}
            }
        }


    def predict(self, ts: np.ndarray) -> dict:
        """
        Perform the top-K discord discovery of each length 
        in the specified length range [minL, maxL] in time series.
        
        Parameters
        ----------
        ts : numpy.ndarray
            The time series to detect arbitrary length discords in it.

        Returns
        -------
        discords_set : dict
            Top-k discords (lengths, indices, distances to its nearest neighbor and matrix profile) 
            for each length in a specified length range [minL, maxL].
        """

        config.STUMPY_EXCL_ZONE_DENOM = self.exclusion_zone
    
        discords_set = {}
        discords = {
            'indices': (-np.ones((self.topK))).tolist(),
            'distances': np.full((self.topK), -np.inf).tolist(),
            'mp': np.full(len(ts), -np.inf, dtype=np.float64),
            }

        all_preprocessing_time = 0
        all_selection_time = 0
        all_refinement_time = 0 
        all_cand_count = 0
        all_discords_count = 0

        n = len(ts)
        
        r = 2*np.sqrt(self.minL)
        print("Start MERLIN")
        print(f"Processing on m = {self.minL}")
        #print(f"===== m = {self.minL} =====")

        excl_zone = int(np.ceil(self.minL / config.STUMPY_EXCL_ZONE_DENOM)) 

        start = time.time()
        T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(ts, self.minL)
        preprocessing_time = (time.time() - start) * 1e3
        all_preprocessing_time += preprocessing_time

        while (discords['distances'][self.topK-1] == -np.inf):
            #print(f"r = {r}")

            phases_result, phases_times = self._perform_phases(T, M_T, Σ_T, excl_zone, T_subseq_isconstant, self.minL, r)

            all_selection_time += phases_times[0]
            all_refinement_time += phases_times[1]

            discords_count = len(phases_result['indices'])
            if (discords_count < self.topK):
                r*=0.5
                continue
            else:
                discords['indices'] = phases_result['indices'][0:self.topK]
                discords['distances'] = phases_result['distances'][0:self.topK]
                discords['mp'][:(n-self.minL+1)] = phases_result['mp']

                all_cand_count += phases_result['cand_count']
                all_discords_count += discords_count

        discords_set[str(self.minL)] = copy.deepcopy(discords)


        for m in range(self.minL+1, self.minL+5):
            
            if (m > self.maxL):
                break

            excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)) 

            discords['indices'] = (-np.ones((self.topK))).tolist()
            discords['distances'] = np.full((self.topK), -np.inf).tolist()
            discords['mp'] = np.full(len(ts), -np.inf, dtype=np.float64)

            r = 0.99 * discords_set[str(m-1)]['distances'][0]

            start = time.time()
            T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(ts, m)
            preprocessing_time = (time.time() - start) * 1e3
            all_preprocessing_time += preprocessing_time

            #print(f"===== m = {m} =====")
            print(f"Processing on m = {m}")
            while (discords['distances'][self.topK-1] == -np.inf):
                #print(f"r = {r}")
                
                phases_result, phases_times = self._perform_phases(T, M_T, Σ_T, excl_zone, T_subseq_isconstant, m, r)

                all_selection_time += phases_times[0]
                all_refinement_time += phases_times[1]

                discords_count = len(phases_result['indices'])
                if (discords_count < self.topK):
                    r = 0.99 * r
                    continue
                else:
                    discords['indices'] = phases_result['indices'][0:self.topK]
                    discords['distances'] = phases_result['distances'][0:self.topK]
                    discords['mp'][:(n-m+1)] = phases_result['mp']

                    all_cand_count += phases_result['cand_count']
                    all_discords_count += discords_count

            discords_set[str(m)] = copy.deepcopy(discords)
             
        
        for m in range(self.minL+5, self.maxL+1):

            excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)) 
                
            mu = np.mean([discords_set[str(m_i)]['distances'][0] for m_i in range(m-5, m)])
            sigma = np.std([discords_set[str(m_i)]['distances'][0] for m_i in range(m-5, m)])
            r = mu - 2 * sigma

            discords['indices'] = (-np.ones((self.topK))).tolist()
            discords['distances'] = np.full((self.topK), -np.inf).tolist()
            discords['mp'] = np.full(len(ts), -np.inf, dtype=np.float64)   

            start = time.time()
            T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(ts, m)
            preprocessing_time = (time.time() - start) * 1e3
            all_preprocessing_time += preprocessing_time

            #print(f"===== m = {m} =====")
            print(f"Processing on m = {m}")
            while (discords['distances'][self.topK-1] == -np.inf):
                #print(f"r = {r}")

                phases_result, phases_times = self._perform_phases(T, M_T, Σ_T, excl_zone, T_subseq_isconstant, m, r)

                all_selection_time += phases_times[0]
                all_refinement_time += phases_times[1]

                discords_count = len(phases_result['indices'])
                if (discords_count < self.topK):
                    r = r - sigma
                    continue
                else:
                    discords['indices'] = phases_result['indices'][0:self.topK]
                    discords['distances'] = phases_result['distances'][0:self.topK]
                    discords['mp'][:(n-m+1)] = phases_result['mp']

                    all_cand_count += phases_result['cand_count']
                    all_discords_count += discords_count
                
            discords_set[str(m)] = copy.deepcopy(discords)
            
        print("Finish MERLIN")

        self.metadata_['total_time'] =  all_preprocessing_time + all_selection_time + all_refinement_time
        self.metadata_["phases"]["selection"] = {"time": all_selection_time, "count": all_cand_count}
        self.metadata_["phases"]["refinement"] = {"time": all_refinement_time, "count": all_discords_count}
        
        return discords_set
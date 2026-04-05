from abc import ABC, abstractmethod
import numpy as np
from stumpy import core, config
import time
from typing import Callable


class SerialDiscordDetector(ABC):
    """
    Base class for serial time series anomaly detection algorithms,
    which includes three phases: preprocessing, candidate selection, and discord refinement.
    """

    def _get_chunks_ranges(self, a: np.ndarray, shift: int = None) -> np.ndarray:
        """
        Take an array that contains only integer numbers in ascending order, and return the
        `(inclusive) start index` and `(exclusive) stop index + shift` for each continuous segment of array.
        
        Parameters
        --------
        a : numpy.ndarray
            The 1-dim array that contains integer numbers in ascending order.
        
        shift : int, default = None
            An integer number by which the stop index of each segement should be shifted. If None, no shift will be applied.
            
        Returns
        -------
        out : numpy.ndarray
            A 2-dim numpy array. The first column is the (inclusive) start index of each segment. The second column is the
            (exclusive) stop index shifted by `shift` units.
        """

        repeats = np.full(len(a), 2)
        diff_is_one = np.diff(a) == 1
        repeats[1:] -= diff_is_one
        repeats[:-1] -= diff_is_one
        out = np.repeat(a, repeats).reshape(-1, 2)
        out[:, 1] += 1
        
        if shift is not None:
            out[:, 1] += shift

        return out

    
    def _find_candidates(self, T: np.ndarray, m: int, M_T: np.ndarray, Σ_T: np.ndarray, r: float, excl_zone: int, T_subseq_isconstant: np.ndarray | Callable[[np.ndarray, int], np.ndarray] = None, init_cands: np.ndarray = None, right: bool = True, finite: bool = False) -> np.ndarray:
        """
        Find a set of the time series candidates, whose distance to all of their right (left) neighbors 
        is at least r when parameter `right` is TRUE (FALSE). If there is no such candidate, all elements of is_cands
        becomes False.
        
        Parameters
        ---------
        T : numpy.ndarray
            The time series or sequence from which the candidates are being selected.
        
        m : int
            Window size.
        
        M_T : ndarray
            Sliding mean of time series.
        
        Σ_T : ndarray
            Sliding standard deviation of time series.
        
        r : float 
            An estimate of discord_dist. The selected candidates retuned by this function have distances of at least `r` 
            to all of their right(left) neighbors when input `right` is set to True(False).
            
            Choosing different values for `r`can affect the performance of the algorithm 
            (see Fig. 5 of the paper). For instance, choosing a very large value for `r` may result in no candidates 
            while choosing a very small value may result in a lot of candidates.  
            (note: `r` is passed to this private function when it is called inside the top-level function `_discords`).
        
        T_subseq_isconstant : numpy.ndarray or function, default = None
            A boolean array that indicates whether a subsequence in ``T`` is constant (``True``). 
            Alternatively, a custom, user-defined function that returns a boolean array that indicates 
            whether a subsequence in ``T`` is constant (``True``). The function must only take two arguments, ``a``, 
            a 1-D array, and ``w``, the window size, while additional arguments may be specified
            by currying the user-defined function using ``functools.partial``. Any subsequence with at least one 
            ``np.nan``/``np.inf`` will automatically have its corresponding value set to ``False`` in this boolean array.

        init_cands : numpy.ndarray, default = None
            A 1-dim boolean array, with shape=(k,) where `k` is the total number of subsquences in the time series. 
            `init_cands[i]` is True if the subsequence with start index `i` is considered as one of the 
            prospective candidates.
            
        right : bool, default = True
            If True (False), candidates returned by the function are guaranteed to have at least the distance of `r` 
            to all of their 'right`('left') neighbors.
        
        finite : bool, default = False
            If True, subsequence with infinite values will not be considered as candidates.   
        
        Returns
        --------
        is_cands : numpy.ndarray
            A 1-dim boolean array, with shape=(k,) where `k` is the total number of subsquences in the time series. 
            `is_cands[i]` is True if the subsequence with start index `i` has minimum distance of `r` to all of its 
            right (left) neighbors when right is True (False).
       """

        k = T.shape[0] - m + 1 
        
        is_cands = np.ones(k, dtype=bool)
        if init_cands is not None:
            is_cands[:] = init_cands
        
        T_subseq_isfinite = np.isfinite(M_T)
        if not finite:
            T_subseq_isfinite[:] = True
        is_cands[~T_subseq_isfinite] = False
        
        for i in np.flatnonzero(T_subseq_isfinite):
            if np.all(is_cands == False):
                break

            cands_idx = np.flatnonzero(is_cands)
            
            if right: 
                non_trivial_cands_idx = cands_idx[cands_idx < max(0, i - excl_zone)]
            else:
                non_trivial_cands_idx = cands_idx[cands_idx > i + excl_zone]
            
            if len(non_trivial_cands_idx) > 0:        
                cand_idx_chunks = self._get_chunks_ranges(non_trivial_cands_idx, shift=m-1) 
                
                for start, stop in cand_idx_chunks:
                    QT = core._sliding_dot_product(T[i:i+m], T[start:stop])
                    D = core._mass(T[i:i+m], T[start:stop], QT,  M_T[i], Σ_T[i], M_T[start:stop-m+1], Σ_T[start:stop-m+1], T_subseq_isconstant[i], T_subseq_isconstant)

                    mask = np.flatnonzero(D < r)   
                    is_cands[start:stop-m+1][mask] = False

                    if len(mask):
                        is_cands[i] = False
            
        return is_cands


    def _refine_candidates(self, T: np.ndarray, m: int, M_T: np.ndarray, Σ_T: np.ndarray, excl_zone: int, is_cands: np.ndarray, T_subseq_isconstant: np.ndarray | Callable[[np.ndarray, int], np.ndarray] = None) -> np.ndarray:
        """
        Search the time series candidates (i.e. subsequences indicated by `is_cands`) and 
        return candidates discords in descending order according to their distance to their nearest neighbor.
        After finding the top-discord among candidates, the discord subsequence and its trivial neighbors will be excluded 
        from candidates before finding the next top-discord.
    
        Parameters
        ---------
        T : numpy.ndarray
            The time series or sequence from which the top discord (out of selected candidates) is discovered. 
        
        m : int
            Window size.
        
        M_T : numpy.ndarray
            Sliding mean of time series.
        
        Σ_T : numpy.ndarray
            Sliding standard deviation of time series.

        is_cands : numpy.ndarray
            A 1-dim boolean array, with shape=(k,) where `k` is the total number of subsquences in the time series 
            when `is_cands[i]` is True, a subsequence with start index `i` is a discord candidate.
        
        T_subseq_isconstant : numpy.ndarray or function, default = None
            A boolean array that indicates whether a subsequence in ``T`` is constant (``True``). 
            Alternatively, a custom, user-defined function that returns a boolean array that indicates 
            whether a subsequence in ``T`` is constant (``True``). The function must only take two arguments, ``a``, 
            a 1-D array, and ``w``, the window size, while additional arguments may be specified
            by currying the user-defined function using ``functools.partial``. Any subsequence with at least one 
            ``np.nan``/``np.inf`` will automatically have its corresponding value set to ``False`` in this boolean array.
        
        Returns
        ---------
        out : numpy.ndarray
            A 2-dim array with three columns. The first column is indices of discords, sorted according to their 
            corresponding distances to their nearest neighbor, provided in the second column. 
            The third column is the indices of the discords' nearest neighbor. 
        """

        k = T.shape[0] - m + 1
        
        P = np.full(k, -np.inf, dtype=np.float64) # matrix profile
        I = np.full(k, -1, dtype=np.int64) # index of nearest neighbor 
        
        for idx in np.flatnonzero(is_cands): 
            Q = T[idx:idx+m]
            QT = core._sliding_dot_product(Q, T)
            D = core._mass(Q, T, QT, M_T[idx], Σ_T[idx], M_T, Σ_T, T_subseq_isconstant[idx], T_subseq_isconstant)
            core.apply_exclusion_zone(D, idx, excl_zone, val=np.inf)
            
            nn_idx = np.argmin(D)  
            if D[nn_idx] == np.inf:
                nn_idx = -1
            P[idx] = D[nn_idx]
            I[idx] = nn_idx
        
        discords_idx = []
        discords_dist = []
        discords_nn_idx = []

        mp = np.copy(P)

        while np.any(mp >= 0):
            idx = np.argmax(mp)
            discords_idx.append(idx)
            discords_dist.append(mp[idx])
            discords_nn_idx.append(I[idx])  
            core.apply_exclusion_zone(mp, idx, excl_zone, -np.inf)
        
        return discords_idx, discords_dist, discords_nn_idx, P


    def _perform_phases(self, T, M_T, Σ_T, excl_zone, T_subseq_isconstant, m, r):
        """
        Perform phases for candidate selection and discord refinement in time series.

        Parameters
        ---------
        T : numpy.ndarray
            The time series or sequence from which the top discord (out of selected candidates) is discovered. 
        
        M_T : numpy.ndarray
            Sliding mean of time series.
        
        Σ_T : numpy.ndarray
            Sliding standard deviation of time series.

        exclusion_zone : int, default = 1
            The exclusion zone.
        
        T_subseq_isconstant : numpy.ndarray or function, default = None
            A boolean array that indicates whether a subsequence in ``T`` is constant (``True``). 
            Alternatively, a custom, user-defined function that returns a boolean array that indicates 
            whether a subsequence in ``T`` is constant (``True``). The function must only take two arguments, ``a``, 
            a 1-D array, and ``w``, the window size, while additional arguments may be specified
            by currying the user-defined function using ``functools.partial``. Any subsequence with at least one 
            ``np.nan``/``np.inf`` will automatically have its corresponding value set to ``False`` in this boolean array.
        
        m : int
            Window size.
        
        r : float
            The distance threshold.
        
        Returns
        ---------
        phases_result : dict
            Results obtained from the implementation of phases 
            (indices, distances to its nearest neighbor, matrix profile, and candidate count).

        times : list
            Runtimes of phases in ms. 
        """

        include = np.ones(len(T)-m+1, dtype=bool)
    
        # phase 1: selection of candidates
        start = time.time()
        is_cands = self._find_candidates(T, m, M_T, Σ_T, r, excl_zone, T_subseq_isconstant, init_cands=include, right=True)
        cand_index = np.flatnonzero(is_cands)
        is_cands = self._find_candidates(T, m, M_T, Σ_T, r, excl_zone, T_subseq_isconstant, init_cands=is_cands, right=False)
        cand_index = np.flatnonzero(is_cands)
        selection_time = (time.time() - start) * 1e3

        # phase 2: refinement of discords
        start = time.time()
        refinement_results = self._refine_candidates(T, m, M_T, Σ_T, excl_zone, is_cands, T_subseq_isconstant) 
        refinement_time = (time.time() - start) * 1e3

        phases_result = {
            'indices': refinement_results[0],
            'distances': refinement_results[1],
            'mp': refinement_results[3],
            'cand_count': len(cand_index)
        }

        times = [selection_time, refinement_time]

        return phases_result, times


    @abstractmethod
    def predict(self, ts: np.ndarray) -> dict:
        """
        Perform the discord discovery in time series. 

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
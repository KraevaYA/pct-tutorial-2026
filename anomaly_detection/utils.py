import numpy as np
import pandas as pd
import math
import copy
import os
import json


def read_ts(file_path: str) -> np.ndarray:
    """
    Read time series

    Parameters
    ----------
    file_path : str 
        Path to file where time series data are stored.
     
    Returns
    -------
    ts : np.ndarray
        The time series data.
    """

    ts = pd.read_csv(file_path, header=None, delim_whitespace=True)
    
    return ts.to_numpy()


def read_json_file(file_path: str):
    """
    Read json file

    Parameters
    ----------
    file_path : str 
        Path to json file.
     
    Returns
    -------
    data : Data from json file.
    """

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    return data


def z_normalize(ts: np.ndarray) -> np.ndarray:
    """
    Calculate the z-normalized time series by subtracting the mean and
    dividing by the standard deviation along a given axis

    Parameters
    ----------
    ts : np.ndarray
        The time series.
    
    Returns
    -------
    norm_ts : np.ndarray
        The z-normalized time series.
    """

    norm_ts = (ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)

    return norm_ts


def is_nan_inf(val):
    """
    Check if the array contains np.nan, -np.nan, or np.inf values.

    Parameters
    ----------
    a : numpy.ndarray
        Array.

    Returns
    -------
    output : bool
    """

    return np.isnan(val) or np.isinf(abs(val))


def apply_exclusion_zone(a, idx: int, excl_zone: int, val: float = -np.inf) -> np.ndarray:
    """ 
    Set all values of array to `val` in a window around a given index.  

    Parameters
    ----------
    a : numpy.ndarray
        Array.

    idx : int
        The index around which the window should be centered.

    excl_zone : int
        Size of the exclusion zone.

    val : float, dafault -np.inf 
        The elements within the exclusion zone will be set to this value.

    Returns
    -------
    a : numpy.ndarray
        Array that is applied an exclusion zone.
    """
    
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(a.shape[-1], idx + excl_zone)

    a[zone_start : zone_stop + 1] = val

    return a


def find_non_overlap_discords(discords_profile: np.ndarray, excl_zone: int) -> dict:
    """
    Search the topK discords based on discords distance profile.

    Parameters
    ----------
    dist_profile: np.ndarray 
        Distances between discords and its nearest neighbors of time series.
    
    excl_zone: int 
        Size of the exclusion zone.
    
    Returns
    -------
    topK_match_results: dict 
        Results of algorithm.
    """

    discords = {
        'indices': [],
        'distances': []
    }

    discords_profile_len = len(discords_profile)
    discords_profile = np.copy(discords_profile).astype(float)

    while np.any(discords_profile >= 0):
        max_idx = np.argmax(discords_profile)
        max_dist = discords_profile[max_idx]

        if (np.isnan(abs(max_dist))) or (np.isinf(abs(max_dist))):
            break

        discords_profile = apply_exclusion_zone(discords_profile, max_idx, excl_zone, -np.inf)

        discords['indices'].append(max_idx)
        discords['distances'].append(max_dist)

    return discords


def select_topk_interest_discords(discords: dict, topK_interest: int = 3) -> dict:
    """
    Select topK interesting discords among all discovered variable-length discords. 
    The discord’s interest is defined as a normalized distance to its 
    nearest neighbor: dist(discord, nearest neightbor) / 2m, 
    where m is the length of discord and nearest neightbor.

    Parameters
    ----------
    discords : dict
        Top-k discords (indices, distances to its nearest neighbor and distance profile) 
        for each length in a specified length range [minL, maxL].

    topK_interest: int, default = 3
        Number of the interest_discords.

    Returns
    -------
    interest_discords : dict
        Top-k interesting discords (indices, distances to its nearest neighbor, lengths, and distance profile).
    """

    discords_list = []

    minL = int(list(discords.keys())[0])
    maxL = int(list(discords.keys())[-1])
    n = len(discords[str(minL)]['mp'])

    anomaly_scores = np.full(n, -np.inf, dtype=np.float64).tolist()

    norm_discords = copy.deepcopy(discords)
    for m in range(minL, maxL+1):
        norm_discords[str(m)]['distances'] = list(map(lambda item: (item**2) /(2*m), norm_discords[str(m)]['distances']))
        norm_discords[str(m)]['mp'][:(n-m+1)] = list(map(lambda item: (item**2) /(2*m), norm_discords[str(m)]['mp'][0:(n-m+1)]))
        neg_inf_idxs = [index for index, item in enumerate(discords[str(m)]['mp']) if item == -np.inf]
        norm_discords[str(m)]['mp'] = np.array(norm_discords[str(m)]['mp'])
        norm_discords[str(m)]['mp'][neg_inf_idxs] = -np.inf
        anomaly_scores = list(map(max, anomaly_scores, norm_discords[str(m)]['mp']))

    for m in range(minL, maxL+1):
        for idx, dist, _ in zip(*norm_discords[str(m)].values()):
            discords_list.append((m, idx, dist))
    sorted_discords_list = sorted(discords_list, key=lambda x: x[2], reverse=True)

    topK_idxs = []
    topK_m = []
    topK_distances = []

    j = 0
    while (j < len(sorted_discords_list)) and (len(topK_idxs) < topK_interest):
        if (len(topK_idxs) > 0):
            non_self_match = 0
            discord_idx = sorted_discords_list[j][1]
            discord_m = sorted_discords_list[j][0]

            for k in range(len(topK_idxs)):
                diff_subs = set(np.arange(discord_idx, discord_idx+discord_m)) - set(np.arange(topK_idxs[k], topK_idxs[k]+topK_m[k]))
                if (len(diff_subs) < m):
                    non_self_match = 1
                    break
            if (non_self_match == 0):
                topK_idxs.append(discord_idx)
                topK_m.append(discord_m)
                topK_distances.append(sorted_discords_list[j][2])
        else:
            topK_m.append(sorted_discords_list[j][0])
            topK_idxs.append(sorted_discords_list[j][1])
            topK_distances.append(sorted_discords_list[j][2])

        j = j + 1

    return {'m': topK_m,
            'indices': topK_idxs,
            'distances': topK_distances,
            'mp': np.array(anomaly_scores)
            }
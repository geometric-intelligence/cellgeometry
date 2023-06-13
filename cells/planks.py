"""Manage Shear Planks."""

import geomstats.backend as gs
import numpy as np

def interpolate_dicrete_curve(curve, n_sampling_points):
    """
    Interpolate a discrete curve so that it gets a given number of sampling points.

    Parameters
    ----------
    curve : array-like, shape=[n_points, 2]
    n_sampling_points : int
    
    Returns
    -------
    interpolation : array-like, shape=[n_sampling_points, 2]
       Discrete curve with n_sampling_points
    """
    old_length = curve.shape[0]
    interpolation = np.zeros((n_sampling_points, 2))
    incr = old_length / n_sampling_points
    pos = np.array(0.0, dtype=np.float32)
    for i in range(n_sampling_points):
        index = int(np.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return gs.array(interpolation, dtype=gs.float32)

def get_array_perplank(data_dict: dict, img: str, min_points: int, k_sampling_points: int) -> list:
    """
    Helper function that gets all the cell ROIs for an entire plank.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of the data that has "ROI" as a key.
    plank : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    min_points : int
        minimum number of points allowable in a ROI.
    k_sampling_points : int
        How many sampling points to interpolate the curve to.
    
    Returns
    -------
    out : list
        A list of all ROIs in a plank.
    """
    planks = []
    #for img in data_dict:
        # print(data_dict[key]["ROIs"][img].keys())
    for cellnum in data_dict[img].keys():
        if (data_dict[img][cellnum]['n'] ) > min_points:
            stacked = np.column_stack([data_dict[img][cellnum]['x'], data_dict[img][cellnum]['y']])
            planks.append(interpolate_dicrete_curve(stacked, k_sampling_points))
    return planks


def exhaustive_align(curve, base_curve):
    """
    Align curve to base_curve to minimize the L² distance.
    
    Parameters
    ----------
    curve : array
    
    base_curve : array

    Returns
    -------
    aligned_curve : discrete curve
    
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.align(point=gs.array(reparametrized), base_point=base_curve)
        distances[shift] = PRESHAPE_METRIC.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve

def build_rois(path: str) -> dict:
    """
    build_rois(path: str)
    
    Given an ROI filepath, extract all the ROIs.
    
    Parameters
    ----------
    path : str
        Filepath to ROIs.
        
    
    Returns
    -------
    out : dict
        A dictionary of all ROIs in a filepath.
    """
    rois = {}
    for roi in sorted(os.listdir(path)):
        # print(roi.split(".")[0])
        rois[roi.split(".")[0]] = read_roi_zip(os.path.join(path,roi))
    return rois

def dictPath(ogPath: str, path: str, dictionary: dict, sep: str="/"):
    """
    dictPath(ogPath: str, path: str, dictionary: dict, sep: str="/")
    
    Given an ROI filepath, extract all the ROIs within that folder.
    
    Parameters
    ----------
    ogPath : str
        Original file path for data folder.
    path : str
        Needed for recursive call, path is first set as ogPath.
    dictionary: dict
        Dictionary, can be empty or an existing one to add new data to.  
    sep: str
        Seperator for the filepath, most likely will be "/" which is the default.
    
    Returns
    -------
    out : dict
        A dictionary of all ROIs in a filepath.
        
    """
    while path.startswith(sep):
        path = path[1:]
        # print(path)
    parts = path.split(sep, 1)
    if len(parts) > 1:
        branch = dictionary.setdefault(parts[0], {})
        dictPath(ogPath, parts[1], branch, sep)
        if parts[0] == "ROIs":
            dictionary[parts[0]] = build_rois(ogPath)

def find_key(dictionary, target_key):
    """
    Recursively searches for a key in a nested dictionary.

    Parameters:
        dictionary (dict): The nested dictionary to search.
        target_key (str): The key to find.

    Returns:
        object: The value associated with the target key, or None if the key is not found.

    Example:
        >>> data = {
        ...     'key1': {
        ...         'key2': {
        ...             'key3': 'value3',
        ...             'key4': 'value4'
        ...         }
        ...     }
        ... }
        >>> result = find_key(data, 'key4')
        >>> print(result)
        value4
    """
    if target_key in dictionary:
        return dictionary[target_key]
    
    for value in dictionary.values():
        if isinstance(value, dict):
            result = find_key(value, target_key)
            if result is not None:
                return result
    
    return None



def find_all_instances(dictionary, target_key1, target_key2, results_list):
    """
    Recursively finds instances of two target keys in a nested dictionary and appends their corresponding values together.

    Parameters:
        dictionary (dict): The nested dictionary to search.
        target_key1 (hashable): The first target key to find.
        target_key2 (hashable): The second target key to find.
        results_list (list): The list where the corresponding values will be appended.

    Returns:
        None

    Example:
        >>> my_dict = {
        ...     "a": 1,
        ...     "b": {"c": 2, "d": 3},
        ...     "e": {"f": 4, "g": {"a": 5, "c": 6}},
        ...     "i": 7
        ... }
        >>> target_key1 = "a"
        >>> target_key2 = "c"
        >>> instances = []
        >>> find_all_instances(my_dict, target_key1, target_key2, instances)
        >>> print(instances)
        [5, 6]
    """
    found_keys = set()
    for key, value in dictionary.items():
        if key == target_key1 or key == target_key2:
            found_keys.add(key)
        elif isinstance(value, dict):
            find_all_instances(value, target_key1, target_key2, results_list)

    if {target_key1, target_key2}.issubset(found_keys):
        results_list.append(np.array([dictionary[target_key1], dictionary[target_key2]]).T)
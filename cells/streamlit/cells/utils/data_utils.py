import os
from read_roi import read_roi_zip
import numpy as np


def build_rois(path) -> dict:
    """
    Builds a dictionary of region of interest (ROI) data from a directory of ROI files.

    Parameters:
        path (str): The path to the directory containing ROI files.

    Returns:
        dict: A dictionary where the keys are ROI names and the values are the corresponding ROI data.

    Example:
        >>> roi_directory = '/path/to/roi_directory'
        >>> rois = build_rois(roi_directory)
        >>> print(rois)
        {'roi1': <ROI data>, 'roi2': <ROI data>, ...}
    """
    rois = {}
    for roi in sorted(os.listdir(path)):
        # print(roi.split(".")[0])
        rois[roi.split(".")[0]] = read_roi_zip(os.path.join(path,roi))
    return rois


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
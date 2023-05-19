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

def get_array_perplank(data_dict: dict, img: str) -> list:
    """
    Helper function that gets all the cell ROIs for an entire plank.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of the data that has "ROI" as a key.
    plank : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    
    Returns
    -------
    out : list
        A list of all ROIs in a plank.
    """
    planks = []
    #for img in data_dict:
        # print(data_dict[key]["ROIs"][img].keys())
    for cellnum in data_dict[img].keys():
        stacked = np.column_stack([data_dict[img][cellnum]['x'], data_dict[img][cellnum]['y']])
        planks.append(interpolate_dicrete_curve(stacked, 50))
        #print(cellnum)
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

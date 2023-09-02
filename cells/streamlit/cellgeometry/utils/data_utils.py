import os
from read_roi import read_roi_zip
import numpy as np
import pandas as pd
import streamlit as st
import glob


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
    st.write(os.listdir(path))
    rois = {}
    for roi in sorted(os.listdir(path)):
        # print(roi.split(".")[0])
        rois[roi.split(".")[0]] = read_roi_zip(os.path.join(path, roi))
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
        if key in (target_key1, target_key2):
            found_keys.add(key)
        elif isinstance(value, dict):
            find_all_instances(value, target_key1, target_key2, results_list)

    if {target_key1, target_key2}.issubset(found_keys):
        results_list.append(
            np.array([dictionary[target_key1], dictionary[target_key2]]).T
        )


def get_files_from_folder(folder_path):
    """
    Retrieves a list of files from a specific folder.

    Parameters:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of file paths.

    Example:
        >>> folder_path = '/path/to/folder'
        >>> files = get_files_from_folder(folder_path)
        >>> print(files)
        ['/path/to/folder/file1.txt', '/path/to/folder/file2.csv', '/path/to/folder/file3.jpg']
    """
    files = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            files.append(os.path.join(folder_path, filename))
    return files


def infer_read_csv_args(file_path):
    with open(file_path, "r") as file:
        # Read the first line of the file
        first_line = file.readline()

        # Check for potential delimiters (sep)
        delimiters = [",", ";", "\t", "|"]  # List of potential delimiters
        sep = None

        for delimiter in delimiters:
            if delimiter in first_line:
                sep = delimiter
                break

        # Check for header row (header)
        header = "infer" if pd.read_csv(file_path, nrows=2).shape[0] == 2 else None

    return sep, header


def check_file_extensions(file_paths):
    valid_extensions = [".zip", ".txt", ".csv"]
    file_extensions = []

    for file_path in file_paths:
        extension = file_path[file_path.rfind(".") :].lower()
        if extension in valid_extensions:
            file_extensions.append(extension)

    return file_extensions


def parse_coordinates(file_path):
    """
    Parses a text file containing x-y coordinates of cells separated by line breaks.
    Each cell's coordinates are stored as a NumPy array in a dictionary.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        dict: A dictionary where the keys represent cell IDs and the values are NumPy arrays of coordinates.

    Example:
        Given the following input file ('coordinates.txt'):

        250 -553
        253 -553
        253 -552
        254 -551
        254 -546

        250 -553
        253 -553
        253 -552
        254 -551
        254 -546

        The function call `parse_coordinates('coordinates.txt')` would return:

        {
            1: np.array([[250, -553],
                         [253, -553],
                         [253, -552],
                         [254, -551],
                         [254, -546]]),
            2: np.array([[250, -553],
                         [253, -553],
                         [253, -552],
                         [254, -551],
                         [254, -546]])
        }
    """
    coordinates = {}

    with open(file_path, "r") as file:
        cell_id = 1
        cell_data = []

        for line in file:
            line = line.strip()

            if line:
                try:
                    x, y = map(int, line.split())
                    cell_data.append([x, y])
                except ValueError:
                    print(f"Skipping invalid line: {line}")
            else:
                if cell_data:
                    coordinates[cell_id] = np.array(cell_data)
                    cell_id += 1
                    cell_data = []

    # Handle the last cell if it doesn't have a line break after it
    if cell_data:
        coordinates[cell_id] = np.array(cell_data)

    return coordinates


def get_file_or_folder_type(path):
    """Determine whether a given path points to a file, a folder, or neither.

    Parameters
    ----------
    path : str
        The path to the item (file or folder) you want to check.

    Returns
    -------
    str
        A string indicating the type of the item:
        - 'File' if the path points to a file.
        - 'Folder' if the path points to a folder.
        - 'Neither' if the path does not point to a file or folder.

    Examples
    --------
    >>> get_file_or_folder_type('/path/to/your/selected/file_or_folder')
    'Folder'
    >>> get_file_or_folder_type('/path/to/your/selected/example.txt')
    'File'
    >>> get_file_or_folder_type('/path/to/nonexistent/item')
    'Neither'
    """
    if os.path.isfile(path):
        return ":page_facing_up: File"
    elif os.path.isdir(path):
        return ":file_folder: Folder"
    else:
        return "Neither"


def get_csv_txt_files(upload_folder):
    # Get a list of all CSV and TXT files in the directory
    csv_files = glob.glob(os.path.join(upload_folder, "*.csv"))
    txt_files = glob.glob(os.path.join(upload_folder, "*.txt"))

    # Extract file names from paths
    csv_file_names = [os.path.basename(file) for file in csv_files]
    txt_file_names = [os.path.basename(file) for file in txt_files]

    # Combine the lists of file names
    filtered_file_names = csv_file_names + txt_file_names

    return filtered_file_names

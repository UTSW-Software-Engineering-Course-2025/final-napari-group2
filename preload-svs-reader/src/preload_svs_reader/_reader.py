"""
This module is a basic implementation of a reader for .svs files using the OpenSlide library.


It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
import numpy as np
import openslide
import logging
import concurrent.futures
from typing import Callable, Optional

def napari_get_reader(path: str | list[str]) -> Optional[Callable]:
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # Ensure path is either a string or a list of strings
    if isinstance(path, str):
        paths = [path]
    elif isinstance(path, list) and all(isinstance(p, str) for p in path):
        paths = path
    else:
        logging.error("Invalid path type. Expected str or list of str.")
        return None

    # Check if all paths end with ".svs"
    if not all(p.endswith(".svs") for p in paths):
        logging.error("Invalid file format. Expected .svs files.")
        return None

    # Return the reader function if the paths are valid
    logging.info("Creating reader for paths: %s", paths)
    return reader_function


def reader_function(path: str | list[str]) -> list[tuple[np.ndarray, dict, str]]:
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    layer_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_read_svs_levels, paths))
    for result in results:
        layer_data.extend(result)
    return layer_data


def _read_svs_levels(_path):
    """Helper to read all levels from a single SVS file."""
    logging.info(f"Reading file: {_path}")
    slide = openslide.OpenSlide(_path)
    data = []
    for level in range(slide.level_count):
        logging.info(f"Reading level {level} of {_path}")
        dims = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, dims)
        arr = np.array(img)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]  # Drop alpha channel if present
        add_kwargs = {"name": f"{_path} [level {level}]"}
        layer_type = "image"
        data.append((arr, add_kwargs, layer_type))
    return data
"""
This module is a threaded, lazy SVS loader for napari using OpenSlide and dask.
"""
import numpy as np
import openslide
import logging
import concurrent.futures
import dask.array as da
from typing import Callable, Optional

def napari_get_reader(path: str | list[str]) -> Optional[Callable]:
    """Gets the napari reader function for SVS files.
    This function checks if the provided path is a string or a list of strings,
    and if the files have the correct .svs extension. If valid, it returns a reader function.
    Args:
        path (str | list[str]): Path to a single SVS file or a list of SVS files.
    Returns:
        Optional[Callable]: A reader function that can be used by napari to read the SVS files,
        or None if the path is invalid.
    """
    if isinstance(path, str):
        paths = [path]
    elif isinstance(path, list) and all(isinstance(p, str) for p in path):
        paths = path
    else:
        logging.error("Invalid path type. Expected str or list of str.")
        return None

    if not all(p.endswith(".svs") for p in paths):
        logging.error("Invalid file format. Expected .svs files.")
        return None

    logging.info("Creating reader for paths: %s", paths)
    return reader_function

def reader_function(path: str | list[str]) -> list[tuple[da.Array, dict, str]]:
    """Take a path or list of paths and return a list of LayerData tuples (dask-backed).
    Args:
        path (str | list[str]): Path to a single SVS file or a list of SVS files.
    Returns:
        list[tuple[da.Array, dict, str]]: A list of tuples containing dask arrays,
        metadata dictionaries, and layer types for each level in the SVS files.
    """
    paths = [path] if isinstance(path, str) else path

    layer_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_dask_svs_levels, paths))
    for result in results:
        layer_data.extend(result)
    return layer_data

def _dask_svs_levels(_path: str) -> list[tuple[da.Array, dict, str]]:
    """Helper to lazily read all levels from a single SVS file using dask.
    Args:
        _path (str): Path to a single SVS file.
    Returns:
        list[tuple[da.Array, dict, str]]: A list of tuples containing dask arrays,
        metadata dictionaries, and layer types for each level in the SVS file.
    """
    logging.info(f"Preparing dask-backed layers for file: {_path}")
    slide = openslide.OpenSlide(_path)
    data = []

    # Order levels from largest to smallest
    levels = sorted(
        range(slide.level_count),
        key=lambda lvl: slide.level_dimensions[lvl][0] * slide.level_dimensions[lvl][1],
        reverse=True
    )

    for level in levels:
        dims = slide.level_dimensions[level]
        h, w = dims[1], dims[0]

        def get_region(y0, x0, h, w, level=level, slide_path=_path):
            slide = openslide.OpenSlide(slide_path)
            img = slide.read_region((x0, y0), level, (w, h))
            arr = np.array(img)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr

        # dask array shape: (height, width, 3)
        chunk_size = (min(1024, h), min(1024, w), 3)
        dask_arr = da.map_blocks(
            lambda block: get_region(
                block.location[0], block.location[1], block.shape[0], block.shape[1]
            ),
            dtype=np.uint8,
            chunks=chunk_size,
            shape=(h, w, 3)
        )

        add_kwargs = {"name": f"{_path} [level {level}]"}
        layer_type = "image"
        data.append((dask_arr, add_kwargs, layer_type))
    return data
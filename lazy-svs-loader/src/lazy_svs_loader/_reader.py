import logging
import os
from typing import Callable, Optional

import dask.array as da
import numpy as np
import openslide
from dask import delayed
from openslide import deepzoom


def napari_get_reader(path: str | list[str]) -> Optional[Callable]:
    """
    Return a function that reads SVS files as multiscale images for napari.
    This function checks if the provided path is a string or a list of strings,
    and ensures that the files have the correct .svs extension.
    If the path is valid, it returns a function that can be used by napari to read
    the images as dask arrays.
    If the path is invalid, it logs an error and returns None.
    Parameters
    ----------
    path : str or list of str
        The path to the SVS file or a list of paths to SVS files.
    Returns
    -------
    Optional[Callable]
        A function that reads the SVS files as multiscale images if the path is valid,
        otherwise None.
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

    logging.info("Creating deepzoom reader for paths: %s", paths)
    return deepzoom_reader_function


def deepzoom_reader_function(
    path: str | list[str],
) -> list[tuple[da.Array, dict, str]]:
    """Returns a list of multiscale images as dask arrays for napari.

    Parameters
    ----------
    path : str or list of str
        The path to the SVS file or a list of paths to SVS files.
    Returns
    -------
    list[tuple[da.Array, dict, str]]
        A list of tuples where each tuple contains:
        - A dask array representing the multiscale image.
        - A dictionary with additional metadata (e.g., name).
        - A string indicating the type of data (always "image").
    """
    paths = [path] if isinstance(path, str) else path

    layer_data = []
    for p in paths:
        if not os.path.isfile(p):
            raise ValueError(f"Invalid SVS file: {path}")
        # high tile sizes load faster, but are less responsive
        pyramid = _build_dask_deepzoom_pyramid(p, tile_size=512, overlap=0)
        add_kwargs = {"name": p}

        layer_data.append((pyramid, add_kwargs, "image"))
    return layer_data


def _build_dask_deepzoom_pyramid(
    path: str, tile_size: int = 256, overlap: int = 0
) -> list[da.Array]:
    """
    Build a multiscale image pyramid from a DeepZoom SVS file using Dask arrays.
    Parameters
    ----------
    path : str
        The path to the SVS file.
    tile_size : int, optional
        The tile size for the pyramid, by default 256.
    overlap : int, optional
        The overlap between tiles, by default 0.
    Returns
    -------
    list[da.Array]
        A list of dask arrays representing the multiscale image pyramid.
        Each array corresponds to a different level of the pyramid.
    """

    # Open the SVS file using OpenSlide
    logging.info("Building Dask DeepZoom pyramid for %s", path)
    slide = openslide.OpenSlide(path)
    limit_bounds = True

    dz = deepzoom.DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds
    )

    # Get the number of levels in the .svs file
    levels = dz.level_count

    pyramid = []
    # Iterate through each level of the pyramid, napari expects the highest resolution first,
    for level in reversed(range(levels)):
        cols, rows = dz.level_tiles[level]
        width, height = dz.level_dimensions[level]

        # Create delayed tile reading functions for each tile
        delayed_tiles = []
        for row in range(rows):
            row_tiles = []
            for col in range(cols):
                # Create delayed tile reader
                delayed_tile = delayed(_read_dz_tile)(
                    dz, level, col, row, tile_size
                )
                row_tiles.append(delayed_tile)
            delayed_tiles.append(row_tiles)

        # Convert delayed tiles to dask arrays
        tile_arrays = []
        for row_tiles in delayed_tiles:
            row_arrays = []
            for delayed_tile in row_tiles:
                tile_array = da.from_delayed(
                    delayed_tile,
                    shape=(tile_size, tile_size, 3),
                    dtype=np.uint8,
                )
                row_arrays.append(tile_array)
            tile_arrays.append(row_arrays)

        # Concatenate tiles to form the complete level
        if tile_arrays:
            # Concatenate tiles within each row
            row_concatenated = []
            for row_arrays in tile_arrays:
                if row_arrays:
                    row_concat = da.concatenate(row_arrays, axis=1)
                    row_concatenated.append(row_concat)

            # Concatenate rows to form the complete level
            if row_concatenated:
                level_array = da.concatenate(row_concatenated, axis=0)

                # Crop to exact dimensions to handle edge tiles
                level_array = level_array[:height, :width, :]
            else:
                level_array = da.zeros((height, width, 3), dtype=np.uint8)
        else:
            level_array = da.zeros((height, width, 3), dtype=np.uint8)

        pyramid.append(level_array)

    return pyramid


@delayed
def _read_dz_tile(
    dz: openslide.deepzoom.DeepZoomGenerator,
    level: int,
    col: int,
    row: int,
    standard_tile_size: int = 256,
) -> np.ndarray:
    """Optimized tile reading with minimal memory allocation and delayed execution.

    Parameters
    ----------
    dz : openslide.deepzoom.DeepZoomGenerator
        The DeepZoom generator for the slide.
    level : int
        The level of the pyramid.
    col : int
        The column index of the tile.
    row : int
        The row index of the tile.
    standard_tile_size : int, optional
        The standard tile size, by default 256.

    Returns
    -------
    np.ndarray
        The tile image as a NumPy array with shape (standard_tile_size, standard_tile_size, 3).
    """
    # Get the tile from from the DeepZoom generator
    img = dz.get_tile(level, (col, row))

    # Convert to numpy array
    arr = np.asarray(img, dtype=np.uint8)

    # Handle RGBA to RGB conversion if needed by throwing away the alpha channel
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    # Pre-allocate output array and copy the tile data onto it
    # This ensures that the output is always of the standard tile size
    output = np.zeros(
        (standard_tile_size, standard_tile_size, 3), dtype=np.uint8
    )

    # Copy actual tile data
    h, w = arr.shape[:2]
    output[:h, :w, :] = arr

    return output
import numpy as np
import openslide
from openslide import deepzoom
import dask.array as da
from dask import delayed
import logging
from typing import Callable, Optional

import openslide.deepzoom


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
        pyramid = _build_dask_deepzoom_pyramid(p)
        add_kwargs = {"name": p}

        layer_data.append((pyramid, add_kwargs, "image"))
    return layer_data


def _build_dask_deepzoom_pyramid(path: str) -> list[da.Array]:
    """
    Build a multiscale image pyramid from a DeepZoom SVS file using Dask arrays.
    Parameters
    ----------
    path : str
        The path to the SVS file.
    Returns
    -------
    list[da.Array]
        A list of dask arrays representing the multiscale image pyramid.
        Each array corresponds to a different level of the pyramid.
    """
    slide = openslide.OpenSlide(path)
    tile_size = 254
    overlap = 1
    limit_bounds = True

    dz = deepzoom.DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds
    )
    levels = dz.level_count

    pyramid = []

    for level in reversed(range(levels)):
        cols, rows = dz.level_tiles[level]
        width, height = dz.level_dimensions[level]

        # Build rows of tiles as dask arrays
        row_arrays = []
        for row in range(rows):
            tile_arrays = []
            for col in range(cols):
                # Read tile size for this tile (may be smaller on edges)
                tile_w, tile_h = _get_tile_size(
                    dz, level, col, row, tile_size, overlap
                )

                delayed_tile = delayed(_read_dz_tile)(
                    dz, level, col, row, tile_w, tile_h
                )
                dask_tile = da.from_delayed(
                    delayed_tile,
                    shape=(tile_h, tile_w, 3),
                    dtype=np.uint8,
                )

                # Remove overlap pixels from all but last tile in the row
                if col < cols - 1 and tile_w > overlap:
                    dask_tile = dask_tile[:, :-overlap, :]

                tile_arrays.append(dask_tile)

            # Concatenate tiles horizontally for this row
            row_concat = da.concatenate(tile_arrays, axis=1)

            # Remove overlap pixels from all but last row
            if row < rows - 1 and height > overlap:
                row_concat = row_concat[:-overlap, :, :]

            row_arrays.append(row_concat)

        # Concatenate all rows vertically
        level_array = da.concatenate(row_arrays, axis=0)

        # Crop to exact size (safe check)
        level_array = level_array[:height, :width, :]

        pyramid.append(level_array)

    return pyramid


def _get_tile_size(
    dz: openslide.deepzoom.DeepZoomGenerator,
    level: int,
    col: int,
    row: int,
    tile_size: int,
    overlap: int,
) -> tuple[int, int]:
    """Calculate actual tile width and height at (level, col, row).
    Handles edge tiles that may be smaller than tile_size.
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
    tile_size : int
        The standard tile size.
    overlap : int
        The overlap between tiles.
    Returns
    -------
    tuple[int, int]
        The actual width and height of the tile at (level, col, row).
    """
    level_width, level_height = dz.level_dimensions[level]

    # Coordinates of top-left pixel of tile
    x = col * (tile_size - overlap)
    y = row * (tile_size - overlap)

    # Actual tile width and height (handle edge tiles)
    tile_w = min(tile_size, level_width - x)
    tile_h = min(tile_size, level_height - y)
    return tile_w, tile_h


def _read_dz_tile(
    dz: openslide.deepzoom.DeepZoomGenerator,
    level: int,
    col: int,
    row: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    """Read a single tile with actual width and height (for edge tiles).
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
    tile_w : int
        The actual width of the tile.
    tile_h : int
        The actual height of the tile.
    Returns
    -------
    np.ndarray
        The tile image as a NumPy array.
    """
    img = dz.get_tile(level, (col, row))
    arr = np.array(img)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # Crop tile to actual size if smaller (edge tiles)
    arr = arr[:tile_h, :tile_w, :]
    return arr

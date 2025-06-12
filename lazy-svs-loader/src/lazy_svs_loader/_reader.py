import numpy as np
import openslide
from openslide import deepzoom
import dask.array as da
from dask import delayed
import logging
from typing import Callable, Optional
from scipy import linalg

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
        pyramid, h_label = _build_dask_deepzoom_pyramid(p)
        add_kwargs = {"name": p}

        layer_data.append((pyramid, add_kwargs, "image"))

        layer_data.append((h_label, {'name': 'hematoxylin'}, 'image'))


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
    h_tile_arrays = []

    for level in reversed(range(levels)):
        cols, rows = dz.level_tiles[level]
        width, height = dz.level_dimensions[level]

        # Build rows of tiles as dask arrays
        row_arrays = []
        h_row_arrays = []

        for row in range(rows):
            tile_arrays = []
            h_tile_arrays = []


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


                # Only process HED for the highest resolution
                if level == levels - 1:
                    h = delayed(get_hematoxylin)(delayed_tile)  # grayscale
                    h_tile = da.from_delayed(
                        h, shape=(tile_h, tile_w), dtype=np.uint8
                    )
                    h_tile_arrays.append(h_tile)

                # Remove overlap pixels from all but last tile in the row
                if col < cols - 1 and tile_w > overlap:
                    dask_tile = dask_tile[:, :-overlap, :]

                tile_arrays.append(dask_tile)


            if level == levels - 1:
                h_row_concat = da.concatenate(h_tile_arrays, axis=1)
                h_row_arrays.append(h_row_concat)

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

        if level == levels - 1:
            h_label_array = da.concatenate(h_row_arrays, axis=0)
            h_label_array = h_label_array[:height, :width]        

    print('len pyramid:', len(pyramid))
    print('# level:', levels)

    return pyramid, h_label_array


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



# convert from rgb space to hed space
def rgb2hed(rgb):
    """
    Convert RGB image to HED using Dask arrays.

    Args:
        rgb (dask.array): RGB image in [0, 1] range, shape (H, W, 3)

    Returns:
        dask.array: HED image, shape (H, W, 3)
    """
    rgb = rgb.astype(np.float32) / 255.0
    # Ensure RGB is in [1e-6, 1.0] to avoid log(0)
    rgb = da.clip(rgb, 1e-6, 1.0)

    # Optical Density (OD) transform: OD = -log(RGB)
    OD = -da.log(rgb)

    # HED stain matrix from Ruifrok & Johnston (columns: H, E, D)
    rgb_from_hed = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    hed_from_rgb = linalg.inv(rgb_from_hed)

    # Convert to Dask array so it's compatible with Dask ops
    hed_from_rgb_dask = da.from_array(hed_from_rgb.T, chunks=(3, 3))

    # Matrix multiplication: (H, W, 3) x (3, 3) → (H, W, 3)
    stains = da.einsum('ijk,kl->ijl', OD, hed_from_rgb_dask)

    # Optional: clip negative stain values
    stains = da.clip(stains, 0, None)

    return stains


# perform rgb2hed and return hematoxylin
def get_hematoxylin(rgb):

    hed_img = rgb2hed(rgb)

    # You can now extract the H, E, and D channels separately:
    h, e, d = np.transpose(hed_img, (2, 0, 1))

    empty = np.zeros_like(h)
    h_rgb = np.dstack((h, empty, empty))

    h_np =  h_rgb.compute()

    h_np = (h_np - np.min(h_np)) / np.max(h_np)
    # print('h_rgb.compute():',h_np)
    # # print('type:', type(h_np))

    # print('min h_np', np.min(h_np))
    # print('max h_np', np.max(h_np))

    # return hematoxylin
    return h_np > 0.1
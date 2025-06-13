import numpy as np
import openslide
from openslide import deepzoom
import dask.array as da
from dask import delayed
import logging
import os
from typing import Callable, Optional
from functools import lru_cache
from scipy import linalg
from vispy.color import Colormap

from collections import deque

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
        
        # Build the RGB image pyramid with hematoxylin labels integrated
        pyramid, hematoxylin_pyramid = _build_dask_deepzoom_pyramid_with_labels(p, tile_size=512, overlap=0)
        print([layer.shape for layer in hematoxylin_pyramid])
        print([layer.shape for layer in pyramid])
        add_kwargs = {"name": f"{p} - RGB"}
        layer_data.append((pyramid, add_kwargs, "image"))
        layer_data.append((hematoxylin_pyramid, {"name": f"{p} - Hematoxylin", "contrast_limits": [0, 1], 'colormap': ('label_red', Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]))}, "image"))
    
 

        '''hematoxylin_label_pyramid = _build_hematoxylin_label_pyramid(p, tile_size=512, overlap=0)
        layer_data.append((hematoxylin_label_pyramid, {"name": f"{p} - Hematoxylin"}, "labels"))'''
    
    return layer_data


def _build_dask_deepzoom_pyramid_with_labels(
    path: str, tile_size: int = 256, overlap: int = 0
) -> list[da.Array]:
    """
    Build a multiscale image pyramid from a DeepZoom SVS file using Dask arrays.
    The highest resolution level will trigger hematoxylin label calculation only when accessed.
    
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
    logging.info("Building Dask DeepZoom pyramid with lazy hematoxylin labels for %s", path)
    slide = openslide.OpenSlide(path)
    limit_bounds = True

    dz = deepzoom.DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds
    )

    # Get the number of levels in the .svs file
    levels = dz.level_count

    pyramid = []
    pyramid_hematoxylin = []
    # Iterate through each level of the pyramid, napari expects the highest resolution first,
    for level in reversed(range(levels)):
        cols, rows = dz.level_tiles[level]
        width, height = dz.level_dimensions[level]

        # Determine if this is the highest resolution level
        # is_highest_res = (level == levels - 1)
        is_highest_res = (level == levels - 1)

        # Create delayed tile reading functions for each tile
        delayed_tiles = []
        for row in range(rows):
            row_tiles = []
            for col in range(cols):
                # if is_highest_res:
                #     # For highest resolution, use the hematoxylin-aware tile reader
                #     '''level_array_hema = _build_hematoxylin_label_pyramid(
                #         path, tile_size=tile_size, overlap=overlap)'''
                # else:
                #     level_array_hema = da.zeros(
                #         (height, width), dtype=np.uint8)
                # print('always load tile')
                # For other levels, use regular tile reader
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
        # pyramid_hematoxylin.append(level_array_hema)
        if is_highest_res: 
            # For the highest resolution level, build hematoxylin labels
            level_array_hema = rgb2hed(level_array)[:,:,0] > 0.1
        else:
            level_array_hema = da.zeros_like(level_array[:, :, 0], dtype=np.uint8)
        pyramid_hematoxylin.append(level_array_hema)

    return pyramid, pyramid_hematoxylin


# def _build_hematoxylin_label_pyramid(
#     path: str, tile_size: int = 256, overlap: int = 0
# ) -> list[da.Array]:
#     """
#     Build a hematoxylin label array for only the highest resolution level from a DeepZoom SVS file.
    
#     Parameters
#     ----------
#     path : str
#         The path to the SVS file.
#     tile_size : int, optional
#         The tile size for the pyramid, by default 256.
#     overlap : int, optional
#         The overlap between tiles, by default 0.
        
#     Returns
#     -------
#     list[da.Array]
#         A list containing a single dask array for the highest resolution hematoxylin labels.
#     """
    
#     # Open the SVS file using OpenSlide
#     logging.info("Building Dask Hematoxylin Labels for highest resolution level of %s", path)
#     slide = openslide.OpenSlide(path)
#     limit_bounds = True

#     dz = deepzoom.DeepZoomGenerator(
#         slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds
#     )

#     # Get the number of levels in the .svs file
#     levels = dz.level_count

#     # Only process the last (highest resolution) level
#     level = levels - 1
#     cols, rows = dz.level_tiles[level]
#     width, height = dz.level_dimensions[level]

#     # Create delayed tile reading functions for each tile
#     delayed_tiles = []
#     for row in range(rows):
#         row_tiles = []
#         for col in range(cols):
#             # Create delayed hematoxylin tile reader
#             delayed_tile = delayed(_read_dz_tile_with_hematoxylin_trigger)(
#                 dz, level, col, row, tile_size, path
#             )
#             row_tiles.append(delayed_tile)
#         delayed_tiles.append(row_tiles)

#     # Convert delayed tiles to dask arrays
#     tile_arrays = []
#     for row_tiles in delayed_tiles:
#         row_arrays = []
#         for delayed_tile in row_tiles:
#             tile_array = da.from_delayed(
#                 delayed_tile,
#                 shape=(tile_size, tile_size),
#                 dtype=np.uint8,
#             )
#             row_arrays.append(tile_array)
#         tile_arrays.append(row_arrays)

#     # Concatenate tiles to form the complete level
#     if tile_arrays:
#         # Concatenate tiles within each row
#         row_concatenated = []
#         for row_arrays in tile_arrays:
#             if row_arrays:
#                 row_concat = da.concatenate(row_arrays, axis=1)
#                 row_concatenated.append(row_concat)

#         # Concatenate rows to form the complete level
#         if row_concatenated:
#             level_array = da.concatenate(row_concatenated, axis=0)
#             # Crop to exact dimensions to handle edge tiles
#             level_array = level_array[:height, :width]
#         else:
#             level_array = da.zeros((height, width), dtype=np.uint8)
#     else:
#         level_array = da.zeros((height, width), dtype=np.uint8)

#     # Return as single-item list to match napari's multiscale format
#     return [level_array]


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


# Global cache for hematoxylin labels to avoid recalculation
_hematoxylin_cache = {}

# @delayed
# def _read_dz_tile_with_hematoxylin_trigger(
#     dz: openslide.deepzoom.DeepZoomGenerator,
#     level: int,
#     col: int,
#     row: int,
#     standard_tile_size: int = 256,
# ) -> np.ndarray:
#     """Read a tile and convert it to hematoxylin labels.

#     Parameters
#     ----------
#     dz : openslide.deepzoom.DeepZoomGenerator
#         The DeepZoom generator for the slide.
#     level : int
#         The level of the pyramid.
#     col : int
#         The column index of the tile.
#     row : int
#         The row index of the tile.
#     standard_tile_size : int, optional
#         The standard tile size, by default 256.

#     Returns
#     -------
#     np.ndarray
#         The hematoxylin labels as a NumPy array with shape (standard_tile_size, standard_tile_size).
#     """
#     # Get the tile from the DeepZoom generator
#     img = dz.get_tile(level, (col, row))

#     # Convert to numpy array
#     arr = np.asarray(img, dtype=np.uint8)

#     # Handle RGBA to RGB conversion if needed by throwing away the alpha channel
#     if arr.shape[-1] == 4:
#         arr = arr[..., :3]

#     # Pre-allocate RGB output array and copy the tile data onto it
#     rgb_output = np.zeros(
#         (standard_tile_size, standard_tile_size, 3), dtype=np.uint8
#     )

#     # Copy actual tile data
#     h, w = arr.shape[:2]
#     rgb_output[:h, :w, :] = arr

#     # Convert to hematoxylin labels
#     hematoxylin_labels = get_hematoxylin(rgb_output)
    
#     '''
#         # Trigger hematoxylin label creation
#     _trigger_hematoxylin_layer_creation(output, level, col, row, path)'''


#     # Convert boolean labels to uint8 (0 for background, 1 for hematoxylin)
#     return hematoxylin_labels.astype(np.uint8)


# def _trigger_hematoxylin_layer_creation(tile_rgb, level, col, row, path):
#     """
#     Trigger the creation of hematoxylin labels when highest resolution tiles are accessed.
#     """
#     if path is None:
#         logging.warning("No path provided; skipping hematoxylin cache")
#         return

#     cache_key = f"{path}_{level}_{col}_{row}"

#     if cache_key not in _hematoxylin_cache:
#         try:
#             hematoxylin_labels = get_hematoxylin(tile_rgb)
#             _hematoxylin_cache[cache_key] = hematoxylin_labels.astype(np.uint8)
#             logging.info(f"Calculated hematoxylin labels for tile ({col}, {row}) at level {level}")
#         except Exception as e:
#             logging.error(f"Failed to calculate hematoxylin for tile ({col}, {row}): {e}")


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
    rgb_from_hed = da.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    hed_from_rgb = linalg.inv(rgb_from_hed)

    # Convert to Dask array so it's compatible with Dask ops
    hed_from_rgb_dask = da.from_array(hed_from_rgb.T, chunks=(3, 3))

    # Matrix multiplication: (H, W, 3) x (3, 3) → (H, W, 3)
    stains = da.einsum('ijk,kl->ijl', OD, hed_from_rgb_dask)

    # Optional: clip negative stain values
    stains = da.clip(stains, 0, None)

    return stains


# def get_hematoxylin(rgb):

#     hed_img = rgb2hed(rgb)

#     # You can now extract the H, E, and D channels separately:
#     h, e, d = np.transpose(hed_img, (2, 0, 1))

#     empty = np.zeros_like(h)
#     h_rgb = np.dstack((h, empty, empty))

#     # return hematoxylin
#     return h_rgb.compute() > 0.5
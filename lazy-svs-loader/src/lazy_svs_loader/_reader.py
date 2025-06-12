from collections.abc import Callable
from typing import Optional
import concurrent.futures

import dask
import dask.array as da
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator


def napari_get_reader(path: str | list[str]) -> Optional[Callable]:
    """Return a reader function if the path is a .svs file or list of .svs files."""
    if isinstance(path, str):
        paths = [path]
    elif isinstance(path, list) and all(isinstance(p, str) for p in path):
        paths = path
    else:
        return None
    if not all(p.lower().endswith(".svs") for p in paths):
        return None
    return reader_function

def reader_function(path: str | list[str]) -> list[tuple[da.Array, dict, str]]:
    """Return a list of (dask array, metadata, layer_type) tuples for napari."""
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_dask_svs_levels, paths))
    for data in results:
        layer_data.extend(data)
    return layer_data

def _dask_svs_levels(_path: str) -> list[tuple[da.Array, dict, str]]:
    """Lazily read all DeepZoom levels from a single SVS file using dask."""
    slide = openslide.OpenSlide(_path)
    dz = DeepZoomGenerator(slide, tile_size=1024, overlap=0, limit_bounds=False)
    data = []
    for dz_level in range(dz.level_count):
        w, h = dz.level_dimensions[dz_level]
        n_tiles_x, n_tiles_y = dz.level_tiles[dz_level]

        def get_tile(x, y, dz_level=dz_level):
            arr = np.array(dz.get_tile(dz_level, (x, y)))
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr

        sample_tile = get_tile(0, 0)
        tile_shape = sample_tile.shape

        def get_tile_padded(x, y, dz_level=dz_level, tile_shape=tile_shape):
            arr = get_tile(x, y, dz_level)
            pad_h = tile_shape[0] - arr.shape[0]
            pad_w = tile_shape[1] - arr.shape[1]
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            return arr

        tiles = [
            [dask.delayed(get_tile_padded)(x, y) for x in range(n_tiles_x)]
            for y in range(n_tiles_y)
        ]
        dask_arr = da.block(
            [[da.from_delayed(tile, shape=tile_shape, dtype=np.uint8) for tile in row] for row in tiles]
        )
        dask_arr = dask_arr[:h, :w, :]
        meta = {"name": f"{_path} [DeepZoom level {dz_level}]"}
        data.append((dask_arr, meta, "image"))
    return data

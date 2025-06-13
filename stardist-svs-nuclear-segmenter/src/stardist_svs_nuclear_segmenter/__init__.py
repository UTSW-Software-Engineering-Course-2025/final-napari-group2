__version__ = "0.0.1"

from napari.utils.notifications import show_info

from . import stardist_segmenter
from ._reader import napari_get_reader
from ._widget import StarDistWidget

show_info("Loading StarDist segmentation widget...")
__all__ = (
    "StarDistWidget",
    "napari_get_reader",
)
print("Pre-loading StarDist model...")
stardist_segmenter.get_model()
print("StarDist model loaded!")
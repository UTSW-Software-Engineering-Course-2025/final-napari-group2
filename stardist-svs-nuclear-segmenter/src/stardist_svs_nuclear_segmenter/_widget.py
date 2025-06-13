from typing import TYPE_CHECKING, Optional

import napari.layers
import numpy as np
from magicgui.widgets import Container, Label, PushButton
from napari.utils.notifications import show_error

import stardist_svs_nuclear_segmenter.stardist_segmenter as stardist_segmenter

if TYPE_CHECKING:
    import napari


def run_stardist_on_region(
    viewer: "napari.viewer.Viewer", ymin: int, ymax: int, xmin: int, xmax: int
) -> "napari.types.LabelsData":
    """Run StarDist segmentation on the current viewport at highest magnification."""

    # Get the current image layer
    image_layer = viewer.layers.selection.active
    if image_layer is None:
        raise ValueError("Please select an image layer.")

    # Get the current viewport data
    data = image_layer.data
    # print("data type:", type(data))

    # # Debug: print all available attributes
    # print(
    #     "data attributes:",
    #     [attr for attr in dir(data) if not attr.startswith("_")],
    # )

    current_level = getattr(image_layer, "data_level", 0)
    print(f"Current display level: {current_level}")
    if current_level != 0:
        raise Exception(
            f"Current display level is {current_level}, but StarDist requires the highest resolution (level 0). Zoom in."
        )

    highest_res_dask = None

    try:
        print(f"Data length: {len(data)}")
        highest_res_dask = data[0]
        print(
            f"Level 0 type: {type(highest_res_dask)}, shape: {highest_res_dask.shape}"
        )
    except Exception as e:
        print(f"Failed to access data[0]: {e}")

    # Validate the data shape
    if highest_res_dask.ndim < 2:
        raise ValueError(
            f"Expected 2D or 3D image data, got {highest_res_dask.ndim}D with shape {highest_res_dask.shape}"
        )

    # Ensure bounds are within the image
    if highest_res_dask.ndim == 2:
        H, W = highest_res_dask.shape
    elif highest_res_dask.ndim == 3:
        H, W = highest_res_dask.shape[:2]
    else:
        raise ValueError(
            f"Unsupported image dimensionality: {highest_res_dask.ndim}D"
        )

    print(f"Image dimensions: H={H}, W={W}")

    # Perform segmentation using StarDist

    print(f"Viewport bounds: y={ymin}:{ymax}, x={xmin}:{xmax}")

    ymin = max(0, ymin)
    ymax = min(H, ymax)
    xmin = max(0, xmin)
    xmax = min(W, xmax)

    # Get the tile from the dask array
    tile = highest_res_dask[ymin:ymax, xmin:xmax].compute()
    # print("tile type:", type(tile), "shape:", tile.shape)

    labels = stardist_segmenter.segment_nuclei(tile)
    print("Segmentation completed, labels shape:", labels.shape)

    # Return just the labels for the viewport region
    return labels


class StarDistWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create status label
        self._status_label = Label(value="Ready to run StarDist segmentation.")
        self._current_labels_layer: Optional[napari.layers.Labels] = None

        self._run_button = PushButton(text="Run StarDist on Viewport")
        self._run_button.enabled = True
        self._run_button.clicked.connect(self._run_segmentation)

        # Layout
        self.extend([self._status_label, self._run_button])

    def _run_segmentation(self):
        """Run segmentation on current viewport."""
        try:
            image_layer = self._viewer.layers.selection.active
            if image_layer is None or not isinstance(
                image_layer, napari.layers.Image
            ):
                raise ValueError("Please select an image layer.")

            (ymin, xmin), (ymax, xmax) = image_layer.corner_pixels
            ymin, ymax = int(np.floor(ymin)), int(np.ceil(ymax))
            xmin, xmax = int(np.floor(xmin)), int(np.ceil(xmax))

            result = run_stardist_on_region(
                self._viewer, ymin, ymax, xmin, xmax
            )

            # Ensure the result is in the correct data type for labels
            if result.dtype != np.uint16:
                result = result.astype(np.uint16)

            # Check if we actually have labels
            if result.max() == 0:
                self._status_label.value = "No objects detected in viewport."
                return

            # Add new labels layer
            layer_name = f"StarDist_y{ymin}_x{xmin}"

            layer = self._viewer.add_labels(
                result,
                name=layer_name,
                translate=(ymin, xmin),  # This positions the labels correctly
                opacity=0.7,
                blending="translucent_no_depth",
            )

            self._status_label.value = f"Found {result.max()} objects. "

        except Exception as e:
            # pop up error
            show_error(f"Segmentation error: {str(e)}")
            print(f"Full error: {e}")
            import traceback

            traceback.print_exc()

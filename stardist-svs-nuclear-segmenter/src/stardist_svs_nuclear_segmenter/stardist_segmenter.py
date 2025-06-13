import dask.array as da
from csbdeep.utils import normalize
from stardist.models import StarDist2D

_model = None


def get_model():
    """Get or initialize the StarDist model."""
    global _model
    if _model is None:
        print("Loading StarDist model...")
        # Initialize your model here
        _model = StarDist2D.from_pretrained("2D_versatile_he")
        print("StarDist model loaded successfully!")
    return _model


def segment_nuclei(image: da.Array) -> da.Array:
    """
    Segment nuclei in a given image using the pre-trained StarDist model.

    Parameters
    ----------
    image : da.Array
        The input image as a dask array.

    Returns
    -------
    da.Array
        A dask array containing the segmented nuclei.
    """
    # Perform segmentation using the StarDist model
    labels, _ = _model.predict_instances(normalize(image))
    return labels

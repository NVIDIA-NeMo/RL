import pytest
from PIL import Image

from nemo_rl.data.datasets.utils import pil_to_base64


def test_pil_to_base64_mime_type_matches_format():
    """The data URI MIME type must match the requested image format."""
    img = Image.new("RGB", (2, 2), color="red")

    bmp_uri = pil_to_base64(img, "BMP")
    assert bmp_uri.startswith("data:image/bmp;base64,"), bmp_uri

    png_uri = pil_to_base64(img, "PNG")
    assert png_uri.startswith("data:image/png;base64,"), png_uri

    jpeg_uri = pil_to_base64(img, "JPEG")

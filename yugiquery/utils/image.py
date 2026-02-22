# yugiquery/utils/image.py

# -*- coding: utf-8 -*-

# ============ #
# Image module #
# ============ #

# ======= #
# Imports #
# ======= #

# Standard library imports
import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple, List

# Third-party imports
import qrcode
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# ========= #
# Functions #
# ========= #


def crop_section(
    im: Image.Image,
    *,
    ref: Tuple[int, int] = (690, 1000),
    offset: Tuple[int, int] = (82, 182),
    crop_size: Tuple[int, int] = (528, 522),
    out_size: Tuple[int, int] | None = None,
) -> Image.Image:
    """Crop a Yu-Gi-Oh card image using ratios derived from a reference image.

    Crops the artwork section from a card image and optionally resizes it.
    The image is converted to RGB if necessary.

    Args:
        im (Image.Image): The PIL Image to crop.
        ref (Tuple[int, int], optional): Reference image dimensions (width, height). Defaults to (690, 1000).
        offset (Tuple[int, int], optional): Crop offset from top-left (x, y) in reference dimensions. Defaults to (82, 182).
        crop_size (Tuple[int, int], optional): Crop size (width, height) in reference dimensions. Defaults to (528, 522).
        out_size (Tuple[int, int] | None, optional): Output size to resize to. Defaults to None (no resize).

    Returns:
        Image.Image: The cropped (and optionally resized) image in RGB mode.
    """
    w, h = im.size

    # Match reference aspect by center-cropping the original image if necessary
    ref_w, ref_h = ref
    ref_aspect = ref_w / ref_h
    aspect = w / h
    if abs(aspect - ref_aspect) > 1e-6:
        if aspect > ref_aspect:
            # image is wider -> crop width
            new_w = int(round(h * ref_aspect))
            new_w = min(new_w, w)
            left = max(0, (w - new_w) // 2)
            right = left + new_w
            im = im.crop((left, 0, right, h))
        else:
            # image is taller -> crop height
            new_h = int(round(w / ref_aspect))
            new_h = min(new_h, h)
            top = max(0, (h - new_h) // 2)
            bottom = top + new_h
            im = im.crop((0, top, w, bottom))
        w, h = im.size

    # Compute ratios from reference
    ox = offset[0] / ref_w
    oy = offset[1] / ref_h
    cw = crop_size[0] / ref_w
    ch = crop_size[1] / ref_h

    left = int(round(ox * w))
    top = int(round(oy * h))
    right = left + int(round(cw * w))
    bottom = top + int(round(ch * h))

    # Clamp to image bounds and shift if necessary
    if right > w:
        right = w
        left = max(0, w - int(round(cw * w)))
    if bottom > h:
        bottom = h
        top = max(0, h - int(round(ch * h)))

    cropped = im.crop((left, top, right, bottom))
    if out_size:
        # Use LANCZOS resampling (handles both old and new PIL versions)
        resampling = Image.Resampling.LANCZOS
        cropped = cropped.resize(out_size, resampling)
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")

    return cropped


def make_qrcode_html(url: str, size: Tuple[int, int] | None = None) -> str:
    """
    Generates a QR code for a given URL and converts it to an HTML tag.

    Args:
        url (str): The URL to encode in the QR code.
        size (Tuple[int, int] | None, optional): The size of the QR code. Defaults to None.

    Returns:
        str: The HTML tag.
    """
    qr_image = qrcode.make(url).get_image()
    return to_html(qr_image, size=size)


def to_html(im: Image.Image, size: Tuple[int, int] | None = None) -> str:
    """
    Converts an image to an HTML tag that can be displayed in a pandas dataframe.

    Args:
        im (Image.Image): The image to convert.
        size (Tuple[int, int] | None, optional): The size to resize the image to. Defaults to None.

    Returns:
        str: The HTML tag.
    """
    if size is not None:
        im.thumbnail(size)
    with BytesIO() as buffer:
        im.save(buffer, "png")
        return '<img src="data:image/png;base64,{}">'.format(base64.b64encode(buffer.getvalue()).decode("utf-8"))


def make_mosaic(
    image_paths: List[str | Path], output_path: str | None = "mosaic.png", cell_width: int = 69, cell_height: int = 100
) -> Image.Image:
    """
    Creates a mosaic image from a list of image files.

    Args:
        image_paths (list): The list of image file paths (str or Path objects).
        output_path (str | None, optional): The path to save the mosaic image. Defaults to "mosaic.png".
        cell_width (int, optional): The width of each image in the mosaic. Defaults to 69.
        cell_height (int, optional): The height of each image in the mosaic. Defaults to 100.

    Returns:
        Image.Image: The mosaic image.
    """
    # Calculate the number of rows and columns based on the aspect ratio
    num_columns = int((len(image_paths) * (cell_height / cell_width)) ** 0.5)
    num_rows = np.ceil(len(image_paths) / num_columns)

    # Create the mosaic image
    mosaic_height = int(cell_height * num_rows)
    mosaic_width = int(cell_width * num_columns)
    mosaic = Image.new("RGBA", (mosaic_width, mosaic_height), (0, 0, 0, 0))
    # Iterate over the PNG files and add them to the mosaic
    widths = []
    iterator = tqdm(image_paths)
    for i, file in enumerate(iterator):
        file_path = Path(file)
        iterator.set_postfix(file=file_path.name)
        # Load the individual image
        image = Image.open(file_path)
        # Resize the image to a height of 100 while maintaining its aspect ratio
        image.thumbnail((cell_width, cell_height))
        widths.append(image.width)
        # Calculate the position to paste the image in the mosaic
        row = i // num_columns
        col = i % num_columns
        mosaic.paste(image, (col * cell_width, row * cell_height))

    # Save the mosaic image
    if output_path is not None:
        mosaic.save(output_path)

    return mosaic


def make_html_gallery(
    image_paths: List[str | Path], output_path: str | None = "gallery.html", cell_width: int = 69, cell_height: int = 100
) -> str:
    """
    Creates an HTML file with a gallery of images from a list of image files.

    Args:
        image_paths (list): The list of image file paths (str or Path objects).
        output_path (str | None, optional): The path to save the HTML file. Defaults to "gallery.html".
        cell_width (int, optional): The width of each cell in the gallery. Defaults to 69.
        cell_height (int, optional): The height of each cell in the gallery. Defaults to 100.

    Returns:
        str: The HTML content.
    """
    num_images = len(image_paths)
    max_cols = int(np.sqrt(num_images))
    styles = ["<style>"]
    styles.append("body {margin: 0; padding: 0;}")
    styles.append(".table-wrap {display: grid; grid-gap: 0px;}")
    styles.append(
        f".table-cell {{ width: {cell_width}px; height: {cell_height}px; overflow: hidden; display: flex; align-items: center; justify-content: center;}}"
    )
    styles.append(".table-img { height: 100%; width: auto; }")
    for i in range(1, max_cols + 1):
        styles.append(
            f"@media screen and (min-width: {cell_width * (i-1)}px)  and (max-width: {cell_width * i}px) {{ .table-wrap {{ grid-template-columns: repeat({i}, {cell_width}px); }}}}"
        )

    styles.append("</style>")

    html = ["<html><head>{}</head><body><div class='table-wrap'>".format("\n".join(styles))]
    for i, image_path in enumerate(tqdm(image_paths)):
        with Image.open(image_path) as im:
            html.append(
                f"<div class='table-cell'><a href='{image_path}' target='_blank'>{to_html(im, (max(cell_width, cell_height),max(cell_width, cell_height)))}</a></div>"
            )

    html.append("</div></body></html>")

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write("\n".join(html))

    return "\n".join(html)


def make_gif(image_paths: List[str | Path], output_path: str | None = "animation.gif", width: int = 69, height: int = 100):
    """
    Generates a GIF from a list of image files.

    Args:
        image_paths (list): The list of image file paths (str or Path objects).
        output_path (str | None, optional): The path to save the GIF. Defaults to "animation.gif".
        width (int, optional): The width of each frame in the GIF. Defaults to 69.
        height (int, optional): The height of each frame in the GIF. Defaults to 100.

    Returns:
        None
    """
    frames = []
    # Loop through each frame and add it to the GIF
    iterator = tqdm(image_paths)
    for i, file in enumerate(iterator):
        file_path = Path(file)
        iterator.set_postfix(file=file_path.name)
        # Load the individual image
        image = Image.open(file_path)
        # Resize the image to a height of 100 while maintaining its aspect ratio
        image.thumbnail((width, height))
        # Add frames to list
        frames.append(image)

    # Save the GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=100,
        loop=0,
    )

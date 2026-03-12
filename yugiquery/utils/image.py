# yugiquery/utils/image.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple, List

# --- Imports: Third-Party --- #
import qrcode
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# --- Imports: Local Application --- #
from .dirs import dirs

# --- Functions --- #


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


def make_gif(
    image_paths: List[str | Path], output_path: str | None = "animation.gif", width: int = 69, height: int = 100
) -> None:
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

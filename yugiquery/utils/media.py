# yugiquery/utils/media.py

# -*- coding: utf-8 -*-

# ============ #
# Image module #
# ============ #

# ======= #
# Imports #
# ======= #

# Standard library imports
import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path
import requests
from typing import Tuple, List
import urllib.parse as up
from typing import Dict

# Third-party imports
import aiohttp
import qrcode
import numpy as np
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm, trange

# Local application imports
from .dirs import dirs
from .api import URLS
from .helpers import md5


# ========= #
# Functions #
# ========= #


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


# ================== #
# API call functions #
# ================== #


def fetch_page_images(
    *titles: str,
    featured: bool = False,
    batch_size: int = 50,
    imlimit: int = 500,
) -> Dict[str, List[str]] | Dict[str, str]:
    """
    Fetches images from the MediaWiki API.

    Args:
        titles (str): Multiple page titles for which to fetch image file names.
        featured (bool, optional): If True, fetches only the main/featured image for each page.
            If False, fetches all images on the page (up to `imlimit` images per page). Defaults to False.
        batch_size (int, optional): Number of page titles to query per API request. Applies to both modes. Defaults to 50.
        imlimit (int, optional): Maximum number of images to fetch per page. Only applies when featured=False.
            Ignored when featured=True (always returns 1 image). Defaults to 500.

    Returns:
        Dict[str, List[str]] | Dict[str, str]: Returns a dictionary mapping each page title to a list of image file names.
    """
    results = {}

    # Process titles in batches (applies to both featured and non-featured modes)
    for i in trange(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        titles_str = up.quote("|".join(batch))

        if featured:
            # Fetch featured/main image only (returns 1 image per page, imlimit is ignored)
            action = "?action=query&format=json&prop=pageimages&piprop=original&titles="
        else:
            # Fetch all images on page (up to imlimit per page)
            action = f"?action=query&format=json&prop=images&imlimit={imlimit}&titles="

        response = requests.get(
            url=URLS.base + action + titles_str,
            headers=URLS.headers,
        ).json()

        pages = response.get("query", {}).get("pages", {})

        for page in pages.values():
            if featured:
                original = page.get("original")
                if original and "source" in original:
                    results[page["title"]] = original["source"].split("/")[-1]
            else:
                images = page.get("images", [])
                results[page["title"]] = [img["title"].lstrip("File:") for img in images]

    return results


# TODO: Refactor
async def download_media(
    *file_names: str,
    output_path: str | Path = "media",
    max_tasks: int = 10,
) -> List[Dict[str, bool]]:
    """
    Downloads a set of files given their names and saves them to a specified folder.
    Returns a dictionary listing file names and their download success status.

    Args:
        file_names (str): Multiple names of the media files to be downloaded.
        output_path (str | Path, optional): The path to the folder where the downloaded files will be saved. Defaults to "media".
        max_tasks (int, optional): The maximum number of files to download at once. Defaults to 10.

    Returns:
        pandas.DataFrame: A DataFrame with columns "file_name", "url" and "success" for each download.
    """
    # Prepare URLs from file names
    file_names_series = pd.Series(file_names)
    file_names_md5 = file_names_series.apply(md5)
    urls = file_names_md5.apply(lambda x: f"/{x[0]}/{x[0]}{x[1]}/") + file_names_series
    download_results = []

    # Download media from URL
    async def download_file(session, url, save_folder, semaphore, pbar):
        async with semaphore:
            save_name = url.split("/")[-1]
            save_file = Path(save_folder).joinpath(save_name)
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"URL {url} returned status code {response.status}")
                    total_size = int(response.headers.get("Content-Length", 0))
                    progress = tqdm(
                        unit="B",
                        total=total_size,
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=save_name,
                        leave=False,
                        dynamic_ncols=(not dirs.is_notebook),
                        disable=("PM_IN_EXECUTION" in os.environ),
                    )

                    # Remove existing file if already exists to ensure a fresh download
                    if save_file.is_file():
                        save_file.unlink()

                    # Write downloaded content in chunks
                    with open(save_file, "wb") as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            progress.update(len(chunk))
                    progress.close()
                download_results.append({"file_name": save_name, "url": URLS.media + url, "success": True})
            except Exception as e:
                # Cleanup if any error occurs and log the failure
                if save_file.is_file():
                    save_file.unlink()
                download_results.append({"file_name": save_name, "url": URLS.media + url, "success": False})
                tqdm.write(f"Failed to download {save_name}: {e}")
            finally:
                pbar.update()

    # Parallelize file downloads
    semaphore = asyncio.Semaphore(max_tasks)
    async with aiohttp.ClientSession(base_url=URLS.media, headers=URLS.headers) as session:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with tqdm(
            total=len(urls),
            unit="file",
            dynamic_ncols=(not dirs.is_notebook),
            disable=("PM_IN_EXECUTION" in os.environ),
        ) as pbar:
            tasks = [
                download_file(
                    session=session,
                    url=url,
                    save_folder=output_path,
                    semaphore=semaphore,
                    pbar=pbar,
                )
                for url in urls
            ]
            # Run tasks as they complete to handle failures gracefully
            await asyncio.gather(*tasks, return_exceptions=True)

    # Return failed downloads as a DataFrame
    return download_results

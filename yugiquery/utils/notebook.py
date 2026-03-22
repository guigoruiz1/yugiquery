# yugiquery/utils/notebook.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import os
import logging
from pathlib import Path

# --- Imports: Third-Party --- #
import arrow
from ipylab import JupyterFrontEnd
from IPython.core.getipython import get_ipython
import nbformat
from nbconvert import HTMLExporter
from nbconvert.writers.files import FilesWriter
from traitlets.config import Config
from IPython.display import HTML, Markdown

# --- Imports: Local Application --- #
from .helpers import lock, unlock
from .dirs import dirs
from .logging import LoggerConfig

# --- Logger Setup --- #
logger = LoggerConfig.get_logger()

# --- Helper Functions --- #


def get_notebook_path() -> Path | None:
    """
    Gets the path of the current notebook opened in JupyterLab.
    If the path cannot be obtained, returns None.

    Args:
        None

    Returns:
        Path: The path of the current notebook.
    """

    file_path = (
        getattr(get_ipython(), "user_ns", {}).get("__vsc_ipynb_file__")
        or os.environ.get("JPY_SESSION_NAME")
        or os.environ.get("PM_IN_EXECUTION")
        or JupyterFrontEnd().sessions.current_session.get("name")
    )

    return Path(file_path) if file_path else None


# --- Save Functions --- #


def save_notebook() -> None:
    """
    Save the current notebook opened in JupyterLab to disk.

    Args:
        None

    Returns:
        None
    """
    app = JupyterFrontEnd()
    app.commands.execute("docmanager:save")
    print("Notebook saved to disk")
    logger.info("Notebook saved to disk")


# --- Export Functions --- #


def export_notebook(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    template: str = "lab",
    theme: str | None = None,
    no_input: bool = True,
) -> None:
    """
    Convert a Jupyter notebook to HTML using nbconvert and save the output to disk.

    Args:
        input_path (str | Path | None, optional): The path to the Jupyter notebook file to convert. If None, gets the notebook path with `get_notebook_path`. Defaults to None.
        output_path (str | Path | None, optional): The path to save the converted HTML file. If None, saves the file to the `REPORTS` directory. Defaults to None.
        template (str, optional): The name of the nbconvert template to use. Defaults to "lab".
        theme (str | None, optional): The name of the nbconvert theme to use. Defaults to None. If template is "lab" and "auto" theme is installed, defaults to the "auto" theme.
        no_input (bool, optional): If True, excludes input cells from the output. Defaults to True.

    Raises:
        ValueError: If no notebook path is provided and cannot be found with `get_notebook_path`.

    Returns:
        None
    """
    if input_path is None:
        input_path = get_notebook_path()
        if input_path is None:
            raise ValueError(
                "Notebook path not found. Please provide an input path or ensure the notebook is opened in JupyterLab."
            )
        input_path = str(get_notebook_path())
    if output_path is None:
        output_path = str(dirs.REPORTS / Path(input_path).stem)

    if template == "lab":
        if theme == None and dirs.NBCONVERT.joinpath("lab/static/theme-auto.css").is_file():
            theme = "auto"

    # Configure the HTMLExporter
    c = Config()
    c.HTMLExporter.template_name = template
    if theme:
        c.HTMLExporter.theme = theme
    if no_input:
        c.TemplateExporter.exclude_output_prompt = True
        c.TemplateExporter.exclude_input = True
        c.TemplateExporter.exclude_input_prompt = True

    # Initialize the HTMLExporter
    html_exporter = HTMLExporter(config=c)

    # Read the notebook content
    with open(input_path, mode="r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert the notebook to HTML
    ipkernel_logger = logging.getLogger("IPKernelApp")
    ipkernel_logger.setLevel(logging.ERROR)

    (body, resources) = html_exporter.from_notebook_node(notebook_content)
    # Write the output to the specified directory
    writer = FilesWriter()
    writer.write(output=body, resources=resources, notebook_name=output_path)

    ipkernel_logger.setLevel(logging.WARNING)

    print(f"Notebook converted to HTML and saved to {output_path}.html")
    logger.info("Notebook converted to HTML and saved to %s.html", output_path)


# --- Jekyll Pages --- #


def make_jekyll_page(
    title: str | None = None,
    placeholders: dict[str, str] | None = None,
    output_path: str | Path | None = None,
) -> None:
    """
    Generate a Jekyll markdown page from the ``assets/html/index.md`` template.

    Args:
        title (str | None, optional): Report title used for ``@title@`` replacement and default output filename.
            If None or empty, uses the current notebook stem from get_notebook_path(); falls back to "index" when unavailable. Defaults to None.
        placeholders (dict[str, str] | None, optional): Additional placeholder values.
            Keys should be provided without @ symbols, e.g. ``{"title": "Cards"}``.
        output_path (str | Path | None, optional): Output path. If None, defaults to
            ``reports/<title>.md``. Relative paths are resolved relative to the reports directory.
    Returns:
        None
    """

    if not title:
        path = get_notebook_path()
        title = path.stem if path else "index"

    lock("make_jekyll_page_{title}")
    try:
        template_path = dirs.get_asset("html", "index.md")
        with open(template_path, encoding="utf-8") as f:
            content = f.read()

        values = {"title": title}
        if placeholders:
            values.update({str(key): str(value) for key, value in placeholders.items()})

        for key, value in values.items():
            # Replace placeholders such as @title@
            placeholder = f"@{key}@"
            content = content.replace(placeholder, value)

        if output_path:
            output_path = Path(output_path)
            # Only add .md if no extension provided
            if not output_path.suffix:
                output_path = output_path.with_suffix(".md")
            if not output_path.is_absolute():
                output_path = dirs.REPORTS / output_path
        else:
            output_path = dirs.REPORTS / f"{title}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, mode="w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Report page generated at %s", output_path)
    finally:
        try:
            unlock("make_jekyll_page_{title}")
        except Exception as e:
            logger.error("Failed to release lock for make_jekyll_page_{title}. %s", e)


# --- HTML Functions --- #


def header(name: str | None = None, timestamp: arrow.Arrow | None = None) -> HTML | None:
    """
    Generates an HTML header with a timestamp and the name of the notebook (if provided).
    If there is no header.html file in the `ASSETS` directory, prints an error message and returns None.

    Args:
        name (str | None, optional): The name of the notebook. If None, uses the current notebook stem from get_notebook_path(); falls back to "Unnamed" when unavailable. Defaults to None.
        timestamp (arrow.Arrow | None, optional): The timestamp to use. If None, uses the current time. Defaults to None.

    Returns:
        HTML | None: The generated HTML header, or None if an error occurs.
    """
    if name is None:
        path = get_notebook_path()
        name = path.stem if path else "Unnamed"

    header_path = dirs.get_asset("html", "header.html")
    try:
        with open(header_path, encoding="utf-8") as f:
            header = f.read()
    except FileNotFoundError:
        logger.warning("Template file not found: %s. Ignoring.", header_path)
        return None

    timestamp = timestamp or arrow.utcnow()
    header = header.replace(
        "@TIMESTAMP@",
        timestamp.strftime("%d/%m/%Y %H:%M %Z"),
    )
    header = header.replace("@NOTEBOOK@", name)
    return HTML(header)


def footer(timestamp: arrow.Arrow | None = None) -> HTML | None:
    """
    Generates an HTML footer with a timestamp.
    If there is no footer.html file in the `ASSETS` directory, prints an error message and returns None.

    Args:
        timestamp (arrow.Arrow | None, optional): The timestamp to use. If None, uses the current time. Defaults to None.

    Returns:
        HTML | None: The generated HTML footer, or None if an error occurs.

    """
    footer_path = dirs.get_asset("html", "footer.html")
    try:
        with open(footer_path, encoding="utf-8") as f:
            footer = f.read()
    except FileNotFoundError:
        logger.warning("Template file not found: %s. Ignoring.", footer_path)
        return None

    now = timestamp or arrow.utcnow()
    footer = footer.replace("@TIMESTAMP@", now.strftime("%d/%m/%Y %H:%M %Z"))

    return HTML(footer)


def buttons() -> HTML | None:
    """
    Generates HTML buttons for updating the index and rarities/regions dictionaries.
    If there is no buttons.html file in the `ASSETS` directory, prints error message and  an returns None.

    Args:
        None
    Returns:
        HTML | None: The generated HTML buttons, or None if an error occurs.
    """
    buttons_path = dirs.get_asset("html", "buttons.html")
    try:
        with open(buttons_path, encoding="utf-8") as f:
            buttons = f.read()
    except FileNotFoundError:
        logger.warning("Template file not found: %s. Ignoring.", buttons_path)
        return None

    return HTML(buttons)

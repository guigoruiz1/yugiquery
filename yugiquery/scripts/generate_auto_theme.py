#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import logging
import re
import os
import json

# --- Imports: Local Application --- #
from yugiquery.utils import setup_logging


logger = logging.getLogger(__name__)

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # Use platform-specific directories
from jupyter_core.paths import jupyter_path


def extract_variables(css):
    """Extracts CSS variables from CSS content, ignoring comments."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)  # Remove comments
    return dict(re.findall(r"(--[^:]+):\s*([^;]+);", css))


def classify_variables(light, dark):
    """Classifies variables into common, light-only, and dark-only groups."""
    common = {k: v for k, v in light.items() if k in dark and light[k] == dark[k]}
    return (
        common,
        {k: v for k, v in light.items() if k not in common},
        {k: v for k, v in dark.items() if k not in common},
    )


def find_theme_directory():
    """Finds the directory containing theme files."""
    for path in jupyter_path("nbconvert", "templates", "lab"):
        if all(os.path.exists(os.path.join(path, "static", f"theme-{t}.css")) for t in ["light", "dark"]):
            return path
    raise FileNotFoundError("Could not find theme-light.css and theme-dark.css.")


def generate_theme_css(template_dir):
    """Generates theme-auto.css using theme-light.css and theme-dark.css."""
    theme_vars = {}
    for theme in ["light", "dark"]:
        theme_path = os.path.join(template_dir, "static", f"theme-{theme}.css")
        with open(theme_path, encoding="utf-8") as f:
            theme_vars[theme] = extract_variables(f.read())

    light, dark = theme_vars["light"], theme_vars["dark"]
    common, light_only, dark_only = classify_variables(light, dark)

    css = [
        "/* Auto-generated theme-auto.css using theme-light.css and theme-dark.css */",
        ":root {",
        *(f"    {k}: {v};" for k, v in sorted(common.items())),
        "",
        "    /* Light Theme */",
        "    @media (prefers-color-scheme: light) {",
        *(f"        {k}: {v};" for k, v in sorted(light_only.items())),
        "    }",
        "",
        "    /* Dark Theme */",
        "    @media (prefers-color-scheme: dark) {",
        *(f"        {k}: {v};" for k, v in sorted(dark_only.items())),
        "",
        "        /* Invert the colors of rendered SVGs */",
        "        .jp-RenderedSVG img {",
        "            filter: invert(1) hue-rotate(180deg);",
        "        }",
        "    }",
        "}",
    ]

    output_file = os.path.join(template_dir, "static", "theme-auto.css")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(css))
    print(text=f"Generated theme-auto.css at {output_file}")
    logger.info("Generated theme-auto.css at %s", output_file)


def update_index_html(template_dir):
    """Updates the index.html.j2 file with the new theme block."""
    index_file = os.path.join(template_dir, "index.html.j2")
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Update notebook_css block
    css_block = """{{ resources.include_css("static/index.css") }}
{% set available_themes = ['dark', 'light', 'auto'] %}
{% set theme_css = "static/theme-" ~ resources.theme ~ ".css" %}
{% if resources.theme in available_themes %}
    {{ resources.include_css(theme_css) }}
{% else %}
    {{ resources.include_lab_theme(resources.theme) }}
{% endif %}"""

    # Replace the theme block in the notebook_css section only
    # Match from the first include_css call through the theme logic
    pattern = r'\{\{ resources\.include_css\("static/index\.css"\) \}\}\s*\n{% if resources\.theme.*?{% endif %}'
    updated_content = re.sub(pattern, css_block, content, count=1, flags=re.DOTALL)

    # Update body_header block to support all three themes
    body_block = """{%- block body_header -%}
{% if resources.theme == 'dark' %}
<body class="jp-Notebook" data-jp-theme-light="false" data-jp-theme-name="JupyterLab Dark">
{% elif resources.theme == 'light' %}
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
{% elif resources.theme == 'auto' %}
<body class="jp-Notebook" data-jp-theme-name="Auto">
{% else %}
<body class="jp-Notebook">
{% endif %}
<main>
{%- endblock body_header -%}"""

    # Replace the body_header block
    body_pattern = r"\{%- block body_header -%\}.*?\{%- endblock body_header -%\}"
    updated_content = re.sub(body_pattern, body_block, updated_content, flags=re.DOTALL)

    with open(index_file, "w", encoding="utf-8") as f:
        f.write(updated_content)
    print(text=f"Updated {index_file}")
    logger.info("Updated %s", index_file)


def update_conf_json(template_dir):
    """Updates the conf.json file with the new preprocessor."""
    conf_file = os.path.join(template_dir, "conf.json")
    with open(conf_file, "r", encoding="utf-8") as f:
        conf = json.load(f)

    conf["preprocessors"]["100-TagRemovePreprocessor"] = {
        "type": "nbconvert.preprocessors.TagRemovePreprocessor",
        "enabled": True,
        "remove_cell_tags": ["exclude"],
    }

    with open(conf_file, "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=4)
    print(text=f"Updated {conf_file}")
    logger.info("Updated %s", conf_file)


def main():
    """Main function to generate theme-auto.css and update template files."""
    template_dir = find_theme_directory()
    generate_theme_css(template_dir)
    update_index_html(template_dir)
    update_conf_json(template_dir)


if __name__ == "__main__":
    main()

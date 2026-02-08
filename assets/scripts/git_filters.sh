#!/bin/bash

# Define the custom filter in the local Git configuration
git config filter.strip_secrets.clean "sed 's/=[^=]*/=/'"
git config filter.strip_secrets.smudge "cat"

# Check if the .gitattributes file exists and if it contains the filter definition
if [ ! -f .gitattributes ] || ! grep -Fxq "**/*secret* filter=strip_secrets" .gitattributes; then
    echo "**/*secret* filter=strip_secrets" >> .gitattributes
fi

# Install nbstripout with the specified attributes file in the notebooks directory
pip install nbstripout
nbstripout --install --attributes "notebooks/.gitattributes"
git config filter.nbstripout.extrakeys 'metadata.celltoolbar metadata.kernelspec metadata.language_info.codemirror_mode.version metadata.language_info.pygments_lexer metadata.language_info.version metadata.toc metadata.notify_time metadata.varInspector cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell'
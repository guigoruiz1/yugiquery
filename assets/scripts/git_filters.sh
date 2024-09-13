#!/bin/bash

# Define the custom filter in the local Git configuration
git config filter.strip_secrets.clean "sed 's/=[^=]*/=REDACTED/'"
git config filter.strip_secrets.smudge "cat"

# Check if the filter definition is already in the .gitattributes file, add it if not
if ! grep -Fxq "**/*secret* filter=strip_secrets" .gitattributes; then
    echo "**/*secret* filter=strip_secrets" >> .gitattributes

# Install nbstripout with the specified attributes file in the notebooks directory
pip install nbstripout
nbstripout --install --attributes "notebooks/.gitattributes"
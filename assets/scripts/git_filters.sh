#!/bin/bash

# Define the custom filter in the local Git configuration
git config filter.strip_secrets.clean "sed 's/=[^=]*/=REDACTED/'"
git config filter.strip_secrets.smudge "cat"

# Determine the location of the .gitattributes file
if [ -d "notebooks" ]; then
    attributes_file="notebooks/.gitattributes"
else
    attributes_file=".gitattributes"
fi

# Write the filter definition to the .gitattributes file
echo "*.ipynb filter=strip_secrets" > $attributes_file

# Install nbstripout with the specified attributes file
pip install nbstripout
nbstripout --install --attributes $attributes_file
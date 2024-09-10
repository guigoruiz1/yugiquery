#!/bin/bash

# Define the custom filter in the local Git configuration
git config filter.strip_secrets.clean "sed 's/=[^=]*/=REDACTED/'"
git config filter.strip_secrets.smudge "cat"

# Write the filter definition to the repository root .gitattributes file
echo "**/*secret* filter=strip_secrets" >> .gitattributes

# Install nbstripout with the specified attributes file in the notebooks directory
pip install nbstripout
nbstripout --install --attributes "notebooks/.gitattributes"
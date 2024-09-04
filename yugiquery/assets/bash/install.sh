#!/usr/bin/env bash

pushd "$(dirname "$0")"

# Make the scripts executable
chmod +x ../../yugiquery.py
chmod +x ../../bot.py

# Install Python packages
pip install -r ../../../requirements.txt
pip install --no-deps git+https://github.com/guigoruiz1/tqdm.git

bash kernel.sh

bash nbconvert.sh
# Finish

popd
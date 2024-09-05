#!/usr/bin/env bash

pushd "$(dirname "$0")"

# Make the scripts executable
chmod +x ../../yugiquery.py
chmod +x ../../bot.py

# Install Python packages
pip install -r ../../../requirements.txt
python ../post_install.py

bash nbconvert.sh
# Finish

popd
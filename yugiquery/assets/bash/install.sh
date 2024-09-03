#! /bin/bash

pushd "$(dirname "$0")"

# Make the scripts executable
chmod +x ../../yugiquery.py
chmod +x ../../bot.py

# Install Python packages
pip3 install -U pip
pip3 install -r ../../../requirements.txt
pip3 install --no-deps git+https://github.com/guigoruiz1/halo.git
pip3 install --no-deps git+https://github.com/guigoruiz1/tqdm.git

# Install Jupyter kernel for the virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    if python3 -m ipykernel install --user --name=yugiquery; then
        echo "Virtual environment Jupyter kernel successfully installed."
    else
        echo "Error: Virtual environment Jupyter kernel could not be installed."
    fi
fi

# Finish

popd
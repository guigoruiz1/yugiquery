#!/usr/bin/env bash

# Install Jupyter kernel for the virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    if python -m ipykernel install --user --name=yugiquery; then
        echo "Virtual environment Jupyter kernel successfully installed."
    else
        echo "Error: Virtual environment Jupyter kernel could not be installed."
    fi
fi
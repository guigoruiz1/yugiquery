#! /bin/bash
CURRENT_DIR=$PWD
pushd "$(dirname "$0")"

# Install Python packages
pip3 install -U pip
pip3 install -r requirements.txt
pip3 install --no-deps git+https://github.com/guigoruiz1/tqdm.git

# Install Jupyter kernel for the virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    python3 -m ipykernel install --user --name=yugiquery
fi

# Install nbconvert template

# Check if nbconvert is installed in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    # Virtual environment is active, construct path accordingly
    templates_directory="$VIRTUAL_ENV/share/jupyter/nbconvert/templates"
else
    # Virtual environment is not active, use default path
    config_directories=$(jupyter --paths | awk '/data:/ {getline; getline; print}')
    templates_directory="$config_directories/nbconvert/templates"
fi

# Check if a valid config directory is found
if [ -n "$templates_directory" ]; then
    # Check if the nbconvert templates directory exists
    if [ -d "$templates_directory" ]; then
        # Create the destination folder if it does not exist
        mkdir -p "$templates_directory/labdynamic"
        
        # Copy the folder to nbconvert templates directory
        cp -r ../assets/nbconvert/* "$templates_directory/labdynamic"
        
        # Check if the copy was successful
        if [ $? -eq 0 ]; then
            echo "nbconvert template successfully installed in $templates_directory"
        else
            echo "Error: Failed to install nbconvert template."
            echo "Be sure to install it manually or change the template used when generating the HTML report."
        fi

    else
        echo "Error: Nbconvert templates directory not found in the specified Jupyter data directory."
        echo "Be sure to install nbconvert and try again or install the template manually."
    fi
else
    echo "Error: Data directory not found in the Jupyter paths output."
    echo "Make sure Jupyter is installed and configured correctly."
fi
# Finish

popd
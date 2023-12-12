#! /bin/bash
CURRENT_DIR=$PWD
cd "$(dirname "$0")"

# Install Python packages
pip3 install -U pip
pip3 install -r requirements.txt
pip3 install git+https://github.com/guigoruiz1/halo.git
pip3 install git+https://github.com/guigoruiz1/tqdm.git
pip3 install -U pynacl
pip3 install -U nbstripout

# Install nbconvert template

# Get the second line after "data:" from jupyter --paths output
config_directories=$(jupyter --paths | awk '/data:/ {getline; getline; print}')

# Check if a valid config directory is found
if [ -n "$config_directories" ]; then
    templates_directory="$config_directories/nbconvert/templates"

    # Check if the nbconvert templates directory exists
    if [ -d "$templates_directory" ]; then
        # Create the destination folder if it does not exist
        mkdir -p "$templates_directory/yugiquery"
        
        # Copy the folder to nbconvert templates directory
        cp -r "../assets/nbconvert"/* "$templates_directory/yugiquery"
        
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

cd $CURRENT_DIR
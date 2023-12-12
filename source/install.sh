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

# Get the config directories from jupyter --paths output
config_directories=$(jupyter --paths | awk '/data:/,/runtime:/ {if (!/data:/ && !/runtime:/) print}')

# Iterate over each config directory and copy the folder to nbconvert templates
for config_dir in $config_directories; do
    templates_directory="$config_dir/nbconvert/templates"

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

        break  # Exit the loop if found
    fi
done

# Check if nbconvert templates directory was not found
if [ ! -d "$templates_directory" ]; then
    echo "Error: Nbconvert templates directory not found in any Jupyter data directory."
    echo "Be sure to install you have nbconvert installed and try again or install the template manualy."
fi
# Finish

cd $CURRENT_DIR
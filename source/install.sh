#! /bin/bash

pushd "$(dirname "$0")"

# Install Python packages
pip3 install -U pip
pip3 install -r requirements.txt
pip3 install git+https://github.com/guigoruiz1/halo.git
pip3 install git+https://github.com/guigoruiz1/tqdm.git
pip3 install -U pynacl
pip3 install -U nbstripout

# Install nbconvert template

# Get the Jupyter installation prefix
jupyter_install_prefix=$(which jupyter | xargs dirname | xargs dirname)

# Set the destination templates directory
templates_directory="$jupyter_install_prefix/share/jupyter/nbconvert/templates"

# Create the destination folder if it does not exist
mkdir -p "$templates_directory/yugiquery"

# Copy the folder to nbconvert templates directory
cp -r "../assets/nbconvert"/* "$templates_directory/yugiquery"

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "yugiquery nbconvert template successfully installed in $templates_directory"
else
    echo "Error: Failed to install yugiquery nbconvert template."
    echo "Be sure to install it manually or change the template used when generating the HTML report."
fi

# Finish

popd

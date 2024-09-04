#!/usr/bin/env bash

# The file passed by Git
FILE="$1"

# Check if the file exists
if [ -f "$FILE" ]; then
    # Process each line of the file
    while IFS= read -r line; do
        # Extract and replace secrets
        if echo "$line" | grep -qE '^[A-Z_]+='; then
            echo "$(echo "$line" | sed -E 's/=[^=]*$//=REDACTED')"
        else
            echo "$line"
        fi
    done < "$FILE"
else
    echo "File not found!"
fi
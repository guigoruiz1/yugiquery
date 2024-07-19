#!/usr/bin/env bash

credential_store=$(git config --get credential.credentialstore)

# Hangle other authentication methods as desired
if [ "$credential_store" = "gpg" ]; then
    passphrase=$1 # You need to define how to get the passphrase
    if [ -z "$passphrase" ]; then
        echo "Passphrase needed."
        exit 1
    else
        echo "$passphrase" | gpg --batch --yes --passphrase-fd 0 --pinentry-mode loopback --sign >> /dev/null
    fi
else
    echo "GPG not used"
    exit 0
fi
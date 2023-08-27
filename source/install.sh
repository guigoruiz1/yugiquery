#! /bin/bash
CURRENT_DIR=$PWD
cd "$(dirname "$0")"
pip3 install -U pip
pip3 install -r requirements.txt
pip3 install git+https://github.com/b1naryth1ef/disco
pip3 install -U gevent
pip3 install -U pynacl
pip3 install -U nbstripout
cd $CURRENT_DIR
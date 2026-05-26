#!/bin/bash
#
ENVIRONMENTS="ashlar bardensr barseq cellpose" 

for environment in $ENVIRONMENTS; do 
    echo "Installing $environment ..."
    echo conda env create -f ~/git/barseq-processing/envs/$environment.environment.yaml 
    conda env create -f ~/git/barseq-processing/envs/$environment.environment.yaml 
    echo "Done."
    echo ""
done
#!/bin/bash
# source /home/prsh7458/anaconda3/etc/profile.d/conda.sh
date
pwd
source activate test
conda env list
which python
python ./reverbspeech_preprocess.py
date

#!/usr/bin/env bash

exp=experiment
jupyter nbconvert --to script $exp.ipynb
cat $exp.py | sed '/get_ipython().magic/d' > scripts/$exp.py
rm $exp.py

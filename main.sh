#!/bin/bash
export PYTHONPATH=/local/vdelaitr/python/lib/python2.7/site-packages/
export THEANO_FLAGS=device=gpu0,base_compiledir=/meleze/data2/vdelaitr/theano
python main.py

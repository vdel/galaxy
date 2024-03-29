#!/bin/bash
PATH=$PATH:/usr/share/cuda-5.5/bin:/usr/share/cuda-5.5/include
export PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/share/cuda-5.5/lib64
export LD_LIBRARY_PATH
export PYTHONPATH=/local/vdelaitr/python/lib/python2.7/site-packages/
export THEANO_FLAGS=device=gpu0,floatX=float32,base_compiledir=/meleze/data2/vdelaitr/theano
#export THEANO_FLAGS=floatX=float32,base_compiledir=/meleze/data2/vdelaitr/theano

if [ $# -ne 1 ]; then
    python train.py
else
    python train.py $1
fi

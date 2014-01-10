#!/bin/bash
#if [ $# -ne 1 ]; 
#    then echo "illegal number of parrameters"
#else
#PATH=$PATH:/usr/share/cuda-5.5/bin:/usr/share/cuda-5.5/include
#export PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/share/cuda-5.5/lib64
export LD_LIBRARY_PATH
export PYTHONPATH=/local/vdelaitr/python/lib/python2.7/site-packages/
export THEANO_FLAGS=device=gpu0,floatX=float32,base_compiledir=/meleze/data2/vdelaitr/theano
#export THEANO_FLAGS=floatX=float32,base_compiledir=/meleze/data2/vdelaitr/theano
python main.py $1
fi

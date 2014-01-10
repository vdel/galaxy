import sys
sys.path.append('data')
sys.path.append('code')

import process
import convolutional_mlp as cnn
from os import listdir
from os.path import isfile, join

assert(len(sys.argv) > 1)
imgDir = sys.argv[1]
if len(sys.argv) > 2:
    tasks, imgs = process.readGT(sys.argv[2])
else:
    tasks = None
    imgs = [ f[:-4] for f in listdir(imgDir) \
            if isfile(join(imgDir, f)) and f[-4:] == '.jpg' ]

nTasks = 1
cnns = [None] * nTasks
for i in range(nTasks):
    cnns[i] = cnn.loadConvNet('task%d.pkl' % i, 1)

for imgName in imgs:
    img = process.readImg(imgDir, imgName)
    pred = [None] * nTasks
    for i in range(nTasks):
        pred[i] = tuple(cnns[i].predict(img)[1])
    print pred[i]

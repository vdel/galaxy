import math
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
    tasks, imgs = process.readGT(sys.argv[2], False)
else:
    tasks = None
    imgs = [ f[:-4] for f in listdir(imgDir) \
            if isfile(join(imgDir, f)) and f[-4:] == '.jpg' ]

nTasks = 1
cnns = [None] * nTasks
for i in range(nTasks):
    cnns[i] = cnn.loadConvNet('task%d.pkl' % i, 1)

sump = [0] * 11
count = [0] * 11
for imgName in imgs:
    img = process.readImg(imgDir, imgName)
    pred = [None] * nTasks
    for i in range(nTasks):
        pred[i] = tuple(cnns[i].predict(img)[0])
    pred = process.makePred(pred)
    if tasks:        
        for i in range(nTasks):
            if imgName in tasks[i]:
                gt = tasks[i][imgName]
                for i, p in enumerate(pred[i]):
                    sump[i] += (p - gt[i]) ** 2
                    count[i] += 1
            else:
                print pred, i
                for p in pred[i]:
                    sump[i] += p ** 2
                    count[i] += 1

if count > 0:
    for i in range(11):
        sys.stderr.write("MSE: " + str(math.sqrt(sump[i] / count[i])))
    sys.stderr.write("MSE: " + str(math.sqrt(sum(sump) / sum(count))))

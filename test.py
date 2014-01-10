import math
import sys
sys.path.append('data')
sys.path.append('code')

import process
import convolutional_mlp as cnn
from os import listdir
from os.path import isfile, join

def printHeader(cnns):
    sys.stdout.write("GalaxyId")
    for i, net in enumerate(cnns):
        for j in range(net.meta['nLabels']):
            sys.stdout.write(",Class%d.%d" % (i + 1, j + 1))
    print ""

def printPred(imgName, pred):
    sys.stdout.write(imgName)
    for i, pp in enumerate(pred):
        for p in pp:
            sys.stdout.write(",%f" % p)
    print ""

assert(len(sys.argv) > 1)
imgDir = sys.argv[1]
if len(sys.argv) > 2:
    tasks, imgs = process.readGT(sys.argv[2], False, False)
else:
    tasks = None
    imgs = [ f[:-4] for f in listdir(imgDir) \
            if isfile(join(imgDir, f)) and f[-4:] == '.jpg' ]

nTasks = 11
cnns = [None] * nTasks
for i in range(nTasks):
    cnns[i] = cnn.loadConvNet('task%d.pkl' % i, 1)

printHeader(cnns)

if tasks:
    sump = [0] * 11
    count = [0] * 11
for imgName in imgs:
    img = process.readImg(imgDir, imgName)
    pred = [None] * nTasks
    for i in range(nTasks):
        pred[i] = cnns[i].predict(img)[0]
    pred = process.makePred(pred)

    printPred(imgName,pred)

    if tasks:        
        for i in range(nTasks):
            if imgName in tasks[i]:
                for p, g in zip(pred[i], tasks[i][imgName]):
                    sump[i] += (p - g) ** 2
                    count[i] += 1
            else:
                for p in pred[i]:
                    sump[i] += p ** 2
                    count[i] += 1

if tasks:
    for i in range(11):
        sys.stderr.write("MSE: " + str(math.sqrt(sump[i] / count[i])) + "\n")
    sys.stderr.write("Average MSE: " + str(math.sqrt(sum(sump) / sum(count))) + "\n")

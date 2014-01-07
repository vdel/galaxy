import sys
sys.path.append('data')
sys.path.append('code')

import process
import convolutional_mlp as cnn

softObj = True
size = 40

taskID = None
if len(sys.argv) > 1:
    taskID = int(sys.argv[1])

for datasets in process.readGT('data/solutions_training.csv', 
                               'data/images_training_cropped_%d' % size, 
                               softObj, taskID):
    i = dataset[4]
    cnn.trainOn('task%d_%d.pkl' % (i, size), datasets, softObj)


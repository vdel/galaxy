import sys
sys.path.append('data')
sys.path.append('code')

import process
import convolutional_mlp as cnn

softObj = True
crop = 160
size = 60

taskID = None
if len(sys.argv) > 1:
    taskID = int(sys.argv[1])

meta = {
    'kernelShape': (5, 5), 
    'poolSize': (2, 2),
    'nConvLayers': 2,
    'nConvKernels': [20, 50],
    'nFullLayers': 1,
    'nFullOut': [500]
}

for dataset in process.readTrainVal('data/training_solutions.csv', 
                                    'data/images_training_cropped_%d_%d' % (crop, size), 
                              softObj, taskID):

    net, loss = cnn.train(dataset, dataset['nLabels'], dataset['shape'], softObj = softObj, **meta)
    net.save('task%d.pkl' % dataset['taskID'])
    net.save('task%d_%s_%.3f.pkl' % (dataset['taskID'], net.getMetaHash(), loss))

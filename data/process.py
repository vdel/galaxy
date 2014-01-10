import os
import csv
import Image
from scipy import misc
import random
import numpy as np
import cPickle
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True, softObj = False):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy   
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
    if softObj:
        return shared_x, shared_y
    else:
        return shared_x, T.cast(shared_y, 'int32')

def makeTree(pb):
    return [pb[0 : 3],   # Task 01
            pb[3 : 5],   # Task 02
            pb[5 : 7],   # Task 03
            pb[7 : 9],   # Task 04
            pb[9 : 13],  # Task 05
            pb[13 : 15], # Task 06
            pb[15 : 18], # Task 07
            pb[18 : 25], # Task 08
            pb[25 : 28], # Task 09
            pb[28 : 31], # Task 10
            pb[31 : 37]] # Task 11


def readGT(annotFile):
    tasks = [None] * 11
    mean_max = [0] * 11
    for i in range(11):
        tasks[i] = {}

    f = open(annotFile, 'rb')
    reader = csv.reader(f, delimiter=',', quotechar='|')
    headerRead = True
    allIDs = []
    for row in reader:
        if headerRead:
            headerRead = False
            continue

        allIDs.append(row[0])
        tree = makeTree(map(float, row[1:]))

        for i, task in enumerate(tree):
            s = reduce(lambda x, y: x + y, task)
            if s > 0:
                distrib = map(lambda x: x / s, task)
                mean_max[i] += max(distrib)
                tasks[i][row[0]] = distrib

    for i in range(11):
        mean_max[i] /= len(tasks[i].values())

    fname = "valSet.pkl"
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        valSet = cPickle.load(f)
        f.close()
    else:
        random.seed()
        splitth = 5000. / len(allIDs)
        valSet = filter(lambda _: random.random() < splitth, allIDs)
        f = open(fname, 'wb')
        cPickle.dump(valSet, f)
        f.close()

    for i, task in enumerate(tasks):
        print "Task #%02d: %d images, confidence = %f" % (i + 1, len(task.values()), mean_max[i])
        
    return tasks, valSet

def readImg(imgDir, imgName, ndata = None):
    img = Image.open(open(os.path.join(imgDir, imgName + ".jpg")))
    img = np.asarray(img, dtype='float64') / 256.

    if ndata == None:
        ndata = img.shape[0] * img.shape[1] * img.shape[2]

    # put image in 4D tensor of shape (1, 3, height, width)
    return img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, ndata)


def readImgTrain(imgDir, tasks, valSet, softObj = True, taskID = None):
    shape = misc.imread(os.path.join(imgDir, tasks[0].keys()[0] + ".jpg")).shape
    ndata = shape[0] * shape[1] * shape[2]
    print "Image size: %dx%d (%.1f Ko = %d dimensions)" % (shape[0], shape[1], ndata * 4 / 1000., ndata)

    for i, task in enumerate(tasks):
        if taskID != None and taskID != -1 and i != taskID:
            continue

        ntrain = 0
        nval = 0
        for key in task.keys():
            if key in valSet:
                nval += 1
            else:
                ntrain += 1
                            
        print "Preparing task #%02d: train: %d images, validation: %d images" % (i + 1, ntrain, nval)

        nlabels = len(task.values()[0])
        if softObj:
            train_set = (np.ndarray(shape = (ntrain, ndata), dtype = 'float32'), 
                         np.zeros(shape = (ntrain, nlabels), dtype = 'float32'))
            valid_set = (np.zeros(shape = (nval, ndata), dtype = 'float32'), 
                         np.zeros(shape = (nval, nlabels), dtype = 'float32'))
        else:
            train_set = (np.ndarray(shape = (ntrain, ndata), dtype = 'float32'),
                         np.zeros(shape = (ntrain), dtype = 'int64')) 
            valid_set = (np.zeros(shape = (nval, ndata), dtype = 'float32'), 
                         np.zeros(shape = (nval), dtype = 'int64'))

        train_count = 0
        valid_count = 0
        for imgName, distrib in task.items():
            isVal = imgName in valSet

            #print "Preparing task #%02d: train: %d/%d images, validation: %d/%d images" % (i + 1, train_count, ntrain, valid_count, nval)
            img = readImg(imgDir, imgName, ndata)

            if softObj:              
                if isVal:
                    valid_set[0][valid_count, :] = img
                    valid_set[1][valid_count, :] = distrib
                    valid_count += 1
                else:
                    train_set[0][train_count, :] = img
                    train_set[1][train_count, :] = distrib
                    train_count += 1
            else:
                distrib = distrib.index(max(distrib))
                if isVal:
                    valid_set[0][valid_count, :] = img
                    valid_set[1][valid_count] = distrib
                    valid_count += 1
                else:
                    train_set[0][train_count, :] = img
                    train_set[1][train_count] = distrib
                    train_count += 1

        dataset = {'train': shared_dataset(train_set, softObj = softObj),
                   'valid': shared_dataset(valid_set, softObj = softObj),
                   'nLabels': nlabels, 
                   'shape': shape, 
                   'taskID': i}
        yield dataset


def readTrainVal(annotFile, imgDir, softObj = False, taskID = None):
    tasks, valSet = readGT(annotFile)
    return readImgTrain(imgDir, tasks, valSet, softObj, taskID)

if __name__ == "__main__":
    readGT('solutions_training.csv', 'images_training_cropped')
    

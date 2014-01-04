import os
import csv
import Image
from scipy import misc
import numpy as np
import cPickle
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
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
    return shared_x, shared_y

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

def readGT(file, imgDir):
    tasks = [None] * 11
    for i in range(11):
        tasks[i] = {}
    
    f = open(file, 'rb')
    reader = csv.reader(f, delimiter=',', quotechar='|')
    headerRead = True
    for row in reader:
        if headerRead:
            headerRead = False
            continue

        tree = makeTree(map(float, row[1:]))

        for i, task in enumerate(tree):
            s = reduce(lambda x, y: x + y, task)
            if s > 0:
                tasks[i][row[0]] = map(lambda x: x / s, task)

    shape = misc.imread(os.path.join(imgDir, tasks[0].keys()[0] + ".jpg")).shape
    ndata = shape[0] * shape[1] * shape[2]
    print "Image size: %dx%d (%.1f Ko)" % (shape[0], shape[1], ndata * 4 / 1000.)

    for i, task in enumerate(tasks):
        print "Task #%02d: %d images" % (i + 1, len(task.values()))

    nsplits = 4
    splitth = 1. / nsplits
    for i, task in enumerate(tasks):
        val = np.random.rand(1, len(task.values())) < splitth
        val = val[0, :]
        nval = reduce(lambda acc, isVal: acc + 1 if isVal else acc, val, 0)
        ntrain = len(val) - nval
        print "Preparing task #%02d: train: %d images, validation: %d images" % (i + 1, ntrain, nval)

        nlabels = len(task.values()[0])
        train_set = (np.zeros([ntrain, ndata], 'float32'), 
                     np.zeros([ntrain, nlabels], 'float32'))
        valid_set = (np.zeros([nval, ndata], 'float32'), 
                     np.zeros([nval, nlabels], 'float32'))

        train_count = 0
        valid_count = 0
        for ((imgName, distrib), isVal) in zip(task.items(), list(val)):
            #print "Preparing task #%02d: train: %d/%d images, validation: %d/%d images" % (i + 1, train_count, ntrain, valid_count, nval)
            img = Image.open(open(os.path.join(imgDir, imgName + ".jpg")))
            img = np.asarray(img, dtype='float64') / 256.

            # put image in 4D tensor of shape (1, 3, height, width)
            img = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, ndata)

            if isVal:
                valid_set[0][valid_count, :] = img
                valid_set[1][valid_count, :] = distrib
                valid_count += 1
            else:
                train_set[0][train_count, :] = img
                train_set[1][train_count, :] = distrib
                train_count += 1

        yield shared_dataset(train_set),  \
              shared_dataset(valid_set)

        test_set = None

        #output = open('task%d.pkl' % i, 'wb')
        #cPickle.dump(train_set, output)
        #cPickle.dump(valid_set, output)
        #cPickle.dump(test_set, output)
        #output.close()

        #f_in = open('task%d.pkl' % i, 'rb')
        #f_out = gzip.open('task%d.pkl.gz' % i, 'wb')
        #f_out.writelines(f_in)
        #f_out.close()
        #f_in.close()

if __name__ == "__main__":
    readGT('solutions_training.csv', 'images_training_cropped')

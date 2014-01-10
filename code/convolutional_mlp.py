"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import copy
import numpy
import cPickle

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

def tuplify(l):
    r = []
    for v in l:
        if isinstance(v, list):
            v = tuple(v)
        r.append(v)
    return r

class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # Output shape
        self.oshape = ((image_shape[2] - filter_shape[2] + 1) / poolsize[0],
                       (image_shape[3] - filter_shape[3] + 1) / poolsize[1])
        assert(self.oshape[0] == int(self.oshape[0]))
        assert(self.oshape[1] == int(self.oshape[1]))

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class ConvNet(object):
    
    def __init__(self, batchSize, shape, nLabels, softObj = True,
                 kernelShape = (5, 5), poolSize = (2, 2),
                 nConvLayers = 2, nConvKernels = [20, 50],  
                 nFullLayers = 1, nFullOut = [500]):
        
        assert(len(nConvKernels) >= nConvLayers)
        assert(len(nFullOut) >= nFullLayers)

        self.meta = {'shape': shape,
                     'nLabels': nLabels,
                     'softObj': softObj,
                     'kernelShape': kernelShape,
                     'poolSize': poolSize,
                     'nConvLayers': nConvLayers,
                     'nConvKernels': nConvKernels,
                     'nFullLayers': nFullLayers,
                     'nFullOut': nFullOut}

        rng = numpy.random.RandomState(23455)
        self.x = T.matrix('x')   # the data is presented as rasterized images
        
        params = []
        ishape = shape[0 : 2]
        prev_output = self.x.reshape((batchSize, shape[2], ishape[0], ishape[1]))
        for l in range(nConvLayers):
            nlayers = shape[2] if l == 0 else nConvKernels[l - 1]
            layer = ConvPoolLayer(rng, input = prev_output,
                                  image_shape = (batchSize, nlayers, ishape[0], ishape[1]),
                                  filter_shape = (nConvKernels[l], nlayers, kernelShape[0], kernelShape[1]), poolsize = poolSize)
            params = layer.params + params
            ishape = layer.oshape
            prev_output = layer.output

        prev_out = nConvKernels[nConvLayers - 1] * ishape[0] * ishape[1]
        prev_output = prev_output.flatten(2)
        for l in range(nFullLayers):
            # construct a fully-connected sigmoidal layer
            layer = HiddenLayer(rng, input = prev_output,
                                n_in = prev_out, n_out = nFullOut[l], activation=T.tanh)
            params = layer.params + params
            prev_out = nFullOut[l]
            prev_output = layer.output        
                
        # classify the values of the fully-connected sigmoidal layer
        self.layer = LogisticRegression(input = prev_output, n_in = prev_out, n_out = nLabels, softObj = softObj)
        self.params = self.layer.params + params
    
        self.eval = theano.function([self.x], self.layer.p_y_given_x)


    def getParams(self):
        return map(lambda shared: shared.get_value(), self.params)

    def setParams(self, params):
        for shared, value in zip(self.params, params):
            shared.set_value(value)

    def getMeta(self):
        return copy.copy(self.meta)

    def getMetaHash(self):
        return hash(frozenset(tuplify(self.meta.keys() + self.meta.values())))

    def save(self, f):
        f = open(f, 'wb')
        cPickle.dump(self.getMeta(), f)
        cPickle.dump(self.getParams(), f)
        f.close()

    def predict(self, img):
        return self.eval(img)

def loadConvNet(f, batchSize):
    f = open(f, 'rb')
    meta = cPickle.load(f)
    params = cPickle.load(f)
    f.close()

    net = ConvNet(batchSize, meta['shape'], meta['nLabels'], meta['softObj'],
                  meta['kernelShape'], meta['poolSize'],
                  meta['nConvLayers'], meta['nConvKernels'],  
                  meta['nFullLayers'], meta['nFullOut'])  
    net.setParams(params)
    return net

def train(dataset, nLabels, shape, 
          learning_rate=0.1, n_epochs=200,          
          batchSize=500, softObj = True,
          kernelShape = (5, 5), poolSize = (2, 2),
          nConvLayers = 2, nConvKernels = [20, 50],  
          nFullLayers = 1, nFullOut = [500]):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    if not isinstance(dataset, dict):
        dataset = load_data(dataset)     

    train_set_x, train_set_y = dataset['train']
    valid_set_x, valid_set_y = dataset['valid']
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batchSize
    n_valid_batches /= batchSize

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    if softObj:
        y = T.matrix('y')
    else:
        y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    net = ConvNet(batchSize, shape, nLabels, softObj,
                 kernelShape, poolSize,
                 nConvLayers, nConvKernels,  
                 nFullLayers, nFullOut)  
   
    # the cost we minimize during training is the NLL of the model
    cost = net.layer.negative_log_likelihood(y)

    validate_model = theano.function([index], net.layer.errors(y),
                                     givens={
            net.x: valid_set_x[index * batchSize: (index + 1) * batchSize],
            y: valid_set_y[index * batchSize: (index + 1) * batchSize]})
 
    # create a list of gradients for all model parameters
    grads = T.grad(cost, net.params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(net.params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
                                  givens={
            net.x: train_set_x[index * batchSize: (index + 1) * batchSize],
            y: train_set_y[index * batchSize: (index + 1) * batchSize]})

    ###############
    # TRAIN MODEL #
    ###############
    if softObj:
        print '... training soft assignement'
    else:
        print '... training hard assignement'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_params = net.getParams()

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i' %
          (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    net.setParams(best_params)
    return net, best_validation_loss

if __name__ == '__main__':
    train('mnist.pkl.gz', 10, (28, 28, 1), softObj = False)


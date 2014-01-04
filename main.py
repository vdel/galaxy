import sys
sys.path.append('data')
sys.path.append('code')

import process
import convolutional_mlp as cnn

for i, datasets in enumerate(process.readGT('data/solutions_training.csv', 
                                            'data/images_training_cropped')):
    cnn.trainOn('task%d_params.pkl' % i, datasets)

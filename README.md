# BirdID_lasagne

Bird image classification using convolutional neural networks in Python

## How to use:

The folder with the images to classify has to be structured as such:

    .
    |-- path_to_folders_with_images
    |    |-- class1
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    └ ...
    |    |-- class2
    |    |    └ ...
    |    |-- class3
        ...
    |    └-- classN
  

Two kinds of python files are provided here: Configuration and Training

### Configuration: 

The parameters to be configured are:


Name | Values | Description
---- | ---- | -----
RATIO | [0,1] | The ratio of the dataset to use for training. The remainder will be used for validation
PER_CATEGORY | integer | The number of images per category to be used for classification. Can not be higher than the number of images in any given category.
CATEGORIES | integer | The number of categories--used to assign labels
DIR | String | The directory containing the images, can be relative to the working directory
TYPE | String | The extension for the images located in the folders, e.g. ".jpg"
DIM | integer | Size of network input images. e.g. "128" will mean that input images are 128x128 pixels. Images will be resized as needed
PREAUG_DIM | integer | The dimension of the images prior to augmentation through random crops. Set this value equal to DIM above to avoid random crops
EPOCHS | integer | Maximum number of epochs to train
BATCH_SIZE | integer | Batch size
SEED1 | int or RandomState | The seed used to pick PER_CATEGORY number of images from each directory. Set to None for a random pick.
SEED2 | int or RandomState | The seed used to generate stratified data splits based on RATIO. Set to None for a random split.
SAVE | boolean | Save the network state or not--can be set to false either way (see description for the training files)
l2_regularization_rate | [0,1] | L2 regularization constant
learning_rate | [0,1] | Learning rate
algorithm | String | The adaptive learning algorithm to use. Options are "rmpsprop", "adagrad", "adam".


## Results:
----------
These networks were used to classify photos of 9 species of birds. The dataset had a minimum of 98 images per category.

Images are resized to 140x140, and then augmented using random horizontal flips and crops to 128x128 with random offsets. The validation set goes through the exact same method for augmentation. 

The networks were trained using stochastic gradient descent(SGD), utilizing an adaptive subgradient method to change the learning rate over time. 

Rectified linear units were used as the activation function for both the convolutional and fully connected layers.

"Same" convolutions were used through zero-padding to keep the input and output dimensions the same.

### convnet_sx3_fc.py:

Layer Structure | Specifics
--------------- | ----------
Input           | 3x128x128
conv3-32        | Pad=1
pool2           | Stride=2
conv3-64        | Pad=1
pool2           | Stride=2
conv3-128       | Pad=1
pool2           | Stride=2
FC:512          | Dropout 50%
Softmax         | 9-way

Achieves 94-95% validation accuracy at about 150,000 gradient steps (about 300 epochs with a batch size of 1 using 60% of the data to train.)
(Loss: ~.3)

### convnet_sx3_ffc.py:

Layer Structure | Specifics
--------------- | ----------
Input           | 3x128x128
conv3-32        | Pad=1
pool2           | Stride=2
conv3-64        | Pad=1
pool2           | Stride=2
conv3-128       | Pad=1
pool2           | Stride=2
FC:512          | Dropout 50%
FC:512          | Dropout 50%
Softmax         | 9-way

Achieves 95-96% validation accuracy at about 200,000 gradient steps (about 300 epochs with a batch size of 1 using 60% of the data to train.)

Achieves slightly higher (up to 97%) accuracy when 80% of the data is used to train. However, using 20% to validate puts about 10 images per category in the validation set, which is not a representative sample.
(Loss: ~.2-.3)

### convnet_sx3_fffc.py:

Layer Structure | Specifics
--------------- | ----------
Input           | 3x128x128
conv3-32        | Pad=1
pool2           | Stride=2
conv3-64        | Pad=1
pool2           | Stride=2
conv3-128       | Pad=1
pool2           | Stride=2
FC:512          | Dropout 50%
FC:512          | Dropout 50%
FC:512          | Dropout 50%
Softmax         | 9-way

Achieves accuracy similar to convnet_sx3_ffc.py despite the additional fully connected layer, with the same number of epochs. Loss is around .2-.3 similarly, and accuracy can get as high as 97% when 80% of the dataset is used to train.

### convnet_sx5_fc.py:

Layer Structure | Specifics
--------------- | ----------
Input           | 3x128x128
conv5-16        | Pad=2
pool2           | Stride=2
conv5-32        | Pad=2
pool2           | Stride=2
conv5-64        | Pad=2
pool2           | Stride=2
FC:512          | Dropout 50%
Softmax         | 9-way

Achieves 92-93% validation accuracy at about 150,000 gradient steps (about 300 epochs with a batch size of 1 using 60% of the data to train.)
(Loss: ~.3) 

## Dependencies:
----------

- [Lasagne][1]: a lightweight library to build and train neural networks in [Theano][2]
- [Scikit-learn][3]
- [Simple Spearmint][2]

[1]: https://github.com/Lasagne/Lasagne
[2]: https://github.com/Theano/Theano
[3]: http://scikit-learn.org/stable/
[4]: https://github.com/craffel/simple_spearmint

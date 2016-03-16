# BirdID_lasagne
----------

Bird image classification using convolutional neural networks in Python

How to use:
----------

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
  

Within each .py file, you can find the constants at the top to specify the number of classes, folder to use, input size, and such.

To save network parameters as a .npy file after training, pass a single command line argument to use as the file's name. (these files can be several hundred MB in size depending on the network architecture)

Results:
----------
These networks were used to classify photos of 9 species of birds. The dataset had a minimum of 98 images per category.

Images are resized to 140x140, and then augmented using random horizontal flips and crops to 128x128 with random offsets. The validation set goes through the exact same method for augmentation. 

The networks were trained using stochastic gradient descent(SGD), utilizing the adaptive subgradient method (Adagrad) to change the learning rate over time. The initial learning rate with adagrad was set to 0.01.

An L2 regularization penalty was applied to the stoachastic loss function to allow for better generalization (l2 regularization rate = 0.0001)

Rectified linear units were used as the activation function for both the convolutional and fully connected layers.

"Same" convolutions were used through zero-padding to keep the input and output dimensions the same.

# convnet_sx3_fc.py:

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

# convnet_sx3_ffc.py:

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

# convnet_sx3_fffc.py:

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

# convnet_sx5_fc.py:

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

Dependencies:
----------

- [Lasagne][1]: a lightweight library to build and train neural networks in [Theano][2]
- [Scikit-learn][3]
- [Simple Spearmint][2]

[1]: https://github.com/Lasagne/Lasagne
[2]: https://github.com/Theano/Theano
[3]: http://scikit-learn.org/stable/
[4]: https://github.com/craffel/simple_spearmint

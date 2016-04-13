# BirdID_lasagne

Bird image classification using convolutional neural networks in Python

![Sample Bird Images](http://i.imgur.com/R2rdTBe.png)

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


Additionally, all configuration files must have a network architecture specified within their build_model() methods, and this method must return a tuple of the input and output layers of this network for the training file to use.
For available layers and such, see documentation for [Lasagne][1].

### Training: 

Files named train_net*.py are used for training networks based on configurations. The recommended one to use is train_net_args.py. The training scripts accept a few command line arguments:

flag | alternative | description
---- | ---- | ----
-c | --config | Name of the configuration file. e.g. sx3_b32_random (do not include the extension)
-s | --save | Name that will be given to the .npy containing network parameters. If no name is specified, the network parameters are not saved after training.
-r | --resume | Name of the npy file to use to load a network to resume training. Make sure that a matching configuration file is used (and a low learning rate might be preferred)

## Results:
These networks were used to classify photos of 9 species of birds. The dataset had a minimum of 98 images per category.

Images are resized to 140x140, and then augmented using random horizontal flips and crops to 128x128 with random offsets. The validation set goes through the exact same method for augmentation. 

The networks were trained using stochastic gradient descent(SGD), utilizing an adaptive subgradient method to change the learning rate over time. 

Rectified linear units were used as the activation function for both the convolutional and fully connected layers.

"Same" convolutions were used through zero-padding to keep the input and output dimensions the same.

The optimal initial learning rate and adaptive algorithm were determined using [simple_spearmint][4].
The script used for hyperparameter optimization is included, see optimize.py


### sx3_ffc_b32.py:
This architecture was chosen for optimization, because (1) it ran in a reasonable amount of time on both the CPU and GPU (2) achieved over 90% accuracy easily wih un-optimized hyperparameters.

After many trials of optimization, the chosen learning rate update algorithm was adam and the chosen initial learning rate was 0.0007.

#### Network architecture:

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

#### Performance: (see run_100_times.sh to see how data was obtained)
After running with stratified random data splits for ~100 runs, mean validation accuracy was found to be 92.9%. 

![Graph of Data](http://i.imgur.com/GeW4UUM.png)


## Training using GPU instances on Amazon EC2

A substantial amount of training was done on Amazon EC2 g2.2xlarge instances. This provided a 20x speedup compared to training on CPUs.
Instance image used: _gpu_theano_
Used setup_aws_gpu.sh to set up the environment (gets git, updates Theano, installs Lasagne, scikit-learn and (simple)spearmint.) This script also mounts an EBS in its default location when attached, to copy logs and states over after training is complete.
See run_and_save.sh for an example to use on these instances.


## Dependencies:
----------

- [Lasagne][1]: a lightweight library to build and train neural networks in [Theano][2]
- [Scikit-learn][3]
- [Simple Spearmint][2]

[1]: https://github.com/Lasagne/Lasagne
[2]: https://github.com/Theano/Theano
[3]: http://scikit-learn.org/stable/
[4]: https://github.com/craffel/simple_spearmint

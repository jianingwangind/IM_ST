# Image Classification and Style Transfer

Pytorch implementation of the joint task of image classification on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) of different styles and style transfer, where [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) is the source domain and [CARLA](http://carla.org/), [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) and Van-Gogh images (from internet) are target domains respectively.

## Datasets
* Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)

* Download the [CARLA Dataset](https://drive.google.com/open?id=1vYjldREhGBRbyoPE3fIWjJ8DlewYkDte)

* Download the [Van-Gogh images](https://drive.google.com/open?id=1yGi44x3xilyNvdysAZ2THZUgINMLbhxu)

## Training model for image classification and Test
* The used model architecture is defined in ./classification/models.py

* To train classifiers with different datasets, specify the dataroot of the corresponding training set in ./classification/cifar_train.py
```
python ./classification/cifar_train.py

```

* To test the trained model for classification task, specify the dataroot of the correspongding test set in ./classification/cifar_test.py
```
python ./classification/cifar_test.py

```

## Training model for Style Transfer
* [CycleGAN](https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix/) is used to perform this task. Clone this repo and follow the instructions to install necessary tools and modules.

In the original code, the model requires the input source and target domain images to have the same size. Since the images of these two domains differ greatly in image size, so replace the ./cyclegan/data/unaligned_dataset.py in [CycleGAN](https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix/) with ./modifications/unaligned_dataset.py in this repo. This enables the target domain images to be resized before cropped to a small image size. The resized target size is controled by the argument loadSize, please choose a reasonable value for different target domains. In ./modifications/unaligned_dataset.py, A represents source domain and B represents target domain. Please give the corresponding paths to the datasets of the two domains.

Also replace ./cyclegan/scripts/train_cyclegan.sh in [CycleGAN](https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix/) with ./modifications/train_cyclegan.sh in this repo.

```
./cyclegan/scripts/train_cyclegan.sh

```

The relevant training arguments can be found in ./cyclegan/options/train_options.py and ./cyclegan/options/train_options.py

## Generate transfered source domain images
Replace ./cyclegan/scripts/test_cyclegan.sh in [CycleGAN](https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix/) with ./modifications/test_cyclegan.sh in this repo.

```
./cyclegan/scripts/test_cyclegan.sh

```


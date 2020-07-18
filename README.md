# Constrained Attention Filter (CAF)
(ECCV 2020) Tensorflow implementation of **A Generic Visualization Approach for Convolutional Neural Networks**

Will upload code after finalizing the camera ready version.

### Teaser
![](https://github.com/ahmdtaha/constrained_attention_filter/blob/master/gif/l2_caf.gif)

### TL;DR
The core components of L2-CAF are

1- TF filter

2- Optimization loop

3- Finalize filter before saving

## Requirements

* Python 3+ [Tested on 3.7]
* Tensorflow 1.X [Tested on 1.14]

## ImageNet Pretrained Models
I used the following
* [DenseNet](https://github.com/pudae/tensorflow-densenet)
* [InceptionV1](https://github.com/tensorflow/models/tree/master/research/slim)
* [ResNet](https://github.com/tensorflow/models/tree/master/research/slim)



## Release History
* 1.0.0
    * First commit Vanilla L2-CAF on DenseNet 12 July 2020
    * Add support for InceptionV1 - 15 July 2020
    * Add support for ResNet50V2 - 18 July 2020
    
### TODO LIST
* Add Fast L2-CAF on DenseNet
* ~~Add InceptionNet and ResNet support~~
* Document to use the code
* Document extra technical tricks not mentioned in the paper 

### Contributing
**It would be great if someone re-implement this in pytorch. Let me know and I will add a link to your Pytorch implementation here**


### MISC Notes
* We did not write localization evaluation code. We used the matlab code released by [CAM](https://github.com/zhoubolei/CAM) in Tables 1  and 3.
We used the python code released by [ADL](https://github.com/junsukchoe/ADL) in Table 2. 
Feel free to evaluate L2-CAF localization with other evaluation codes.

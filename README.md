# Constrained Attention Filter (CAF)
(ECCV 2020) Tensorflow implementation of **A Generic Visualization Approach for Convolutional Neural Networks**

[Paper](https://arxiv.org/abs/2007.09748) | [1 Min Video](https://youtu.be/W4xaKQlPEl0) | [10 Mins Video](https://youtu.be/Wpw3ewSvnFE)

### Qualitative Evaluation -- L2-CAF Slow Motion Convergence

|                   | One Object | Two Objects |
|-------------------|------------|-------------|
|     Last Conv     |![](https://github.com/ahmdtaha/constrained_attention_filter/blob/master/gif/ILSVRC2012_val_00000003_cls_230_dense_block4_conv_block24_0.gif)|![](https://github.com/ahmdtaha/constrained_attention_filter/blob/master/gif/ILSVRC2012_val_00000021_cls_334_dense_block4_conv_block24_0.gif)|
|Intermediate Conv |![](https://github.com/ahmdtaha/constrained_attention_filter/blob/master/gif/ILSVRC2012_val_00000003_cls_230_dense_block4_conv_block10_0.gif)|![](https://github.com/ahmdtaha/constrained_attention_filter/blob/master/gif/ILSVRC2012_val_00000021_cls_334_dense_block4_conv_block10_0.gif)|

### TL;DR
L2-CAF has three core components:

1- [TF filter](https://github.com/ahmdtaha/constrained_attention_filter/blob/035f0880baae6a12540dd0b4cc0830cef243c1af/nets/attention_filter.py#L19) This is the function that inserts L2-CAF inside a network (E.g, inside a [DenseNet](https://github.com/ahmdtaha/constrained_attention_filter/blob/035f0880baae6a12540dd0b4cc0830cef243c1af/nets/densenet161.py#L90)). L2-CAF is by default disabled; it is passive during classification.
To active/de-activate L2-CAF (turn on and off the filter), I use the bool `atten_var_gate`. False deactivate L2-CAF, while True activates the filter.

2- [Optimization loop](https://github.com/ahmdtaha/constrained_attention_filter/blob/1d45e121fa56b131e94dbb72c22c169589bb679f/visualize_attention.py#L160)  In this loop, we computes the class-oblivious and class-specific loss and leverage gradient descent to minimize it. When the loss stabilize (loss - prev_loss< 10e-5), break out of the loop.

3- [Finalize filter before saving](https://github.com/ahmdtaha/constrained_attention_filter/blob/1d45e121fa56b131e94dbb72c22c169589bb679f/visualize_attention.py#L18) After convergence, the output filter is normalized (L2-Norm|Softmaxed|Gauss-ed) before generating the heatmap.

## Requirements

* Python 3+ [Tested on 3.7]
* Tensorflow 1.X [Tested on 1.14]

## ImageNet Pretrained Models
I used the following
* [DenseNet](https://github.com/pudae/tensorflow-densenet)
* [InceptionV1](https://github.com/tensorflow/models/tree/master/research/slim)
* [ResNet](https://github.com/tensorflow/models/tree/master/research/slim)

## Usage example

Update [`base_config._load_user_setup`](https://github.com/ahmdtaha/constrained_attention_filter/blob/f95afd6c547a24122b8f182427fa4191ce5cb86c/config/base_config.py#L74) with your configurations.
Mainly, set the location of pre-trained model (e.g, densenet). The released code optimizes the constrained attention filter on samples images from the "input_imgs" directory. However, if you plan to run the code on a whole dataset (e.g, ImageNet), you shoud set the `local_datasets_dir` in _load_user_setup  

The unit L2-Norm constrained attention filter has two operating modes. 
* `visualize_attention.py` is the script for the vanilla "slow" (4 seconds) mode. I recommend running this first before experimenting with the fast L2-CAF version. The code of this mode is much easier to understand. The script's main function sets all the hyper-parameters needed. I will ellaborate more on each hyper-parameter soon.

* `visualize_attention_fast.py` is the script for the fast (0.3 seconds) mode. The script only supports denseNet. I will add support to Inception and ResNet soon.
 This script only works for visualizing attention is the last conv layer. I only use it for quantitative evaluation experiments, for instance, when I evaluate L2-CAF using ImageNet validation split.

    
### TODO LIST
~~* Add Fast L2-CAF on DenseNet~~
* ~~Add InceptionNet and ResNet support~~
* Document to use the code
* Document the intermediate layer visualization
* Document extra technical tricks not mentioned in the paper 

### Contributing
**It would be great if someone re-implement this in pytorch. Let me know and I will add a link to your Pytorch implementation here**


### MISC Notes
* We did not write localization evaluation code. We used the matlab code released by [CAM](https://github.com/zhoubolei/CAM) in Tables 1  and 3.
We used the python code released by [ADL](https://github.com/junsukchoe/ADL) in Table 2. 
Feel free to evaluate L2-CAF localization with other evaluation codes.
* The softmax and Gaussian filters are released upon a reviewer request. The current Gaussian filter implementation is hard-coded to support only 7x7 attention filter.
 It is straight forward to extend it for any odd filter-size (e.g., 13x13). However, for even filter-size I think more changes are required. The last conv layer in standard architectures is 7x7. So the current configuration should cover most typical case-scenario.
 
* I used modules of this code (especially the nets package) in multiple projects, so there is a lot of code that is not related to L2-CAF. I will iteratively clean the code. The TL;DR section, at the top of the readme file, highlights the core functions related to L2-CAF.

## Release History
* 1.0.0
    * First commit Vanilla L2-CAF on DenseNet, InceptionV1, and ResNet50V2 on 12, 15,18 July 2020
    * Add Fast L2-CAF on DenseNet 21 July 2020
    * Add Fast L2-CAF on Inception 22 July 2020
    * Add Fast L2-CAF on ResNet 23 July 2020

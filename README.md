# Medical SSL
Medical Imaging Using Self-Supervised Learning

## Abstract
Visual representation learning in deep learning has been performed previously using visual data with annotations or metadata, in a fashion known as supervised learning.
Although, supervised learning in vision has resulted in breakthrough performeances it comes at a very expensive cost - the amount of money and time needed to annotate datasets. Also, using annotations likely results in the introduction of biases during pre-training. This forces the algorithm to learn feature representations that are prone to spurious correlation.

One way to mitigate this pnenomenon is self-supervised learning, a paradigm that enables algorithms to learn intrinsic representations within the data using signals from the data itself, and enabling representation learning from huge magnitudes of data without annotations or metadata.

In this work, we explore some of the recent contributions in self-supervised learning for vision in the space of medical imaging. 
Our contributions:

1. Apply self-supervised learning algorithms such as BarlowTwins, BYOL, MoCo, NNCLR, SimCLR, and SimSiam in the domain of medical imaging for pathology detection/classification.

2. Use various model architectures as the backbone for these algorithms. We experiment with DenseNet-121, GoogleNet, MnasNet, MobileNet V2, ResNet-50, ShuffleNet and SqueezeNet. The original algorithms use a ResNet-50 as the backbone architecture.

3. We train all the algorithms using the data augmentation technique proposed in SimCLR, and use the embedding size proposed in NNCLR in BarlowTwins, BYOL, and SimSiam. Given the smaller number of samples of medical data, we use smaller batch sizes than the proposed large batch sizes from some of the original works.

## Requirements
Install the following with `pip` \
`torch` \
`torchvision` \
`pytorch-lighning` \
`pytorch-bolts` \
`lightly` \
`timm` \
`sklern`

## SSL Models
[Self-Supervised Learning via Redundancy Reduction (Barlow Twins)](https://arxiv.org/abs/2103.03230) \
[Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) \
[Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)](https://arxiv.org/abs/1911.05722) \
[Nearest-Neighbor Contrastive Learning of Visual Representations (NNCLR)](https://arxiv.org/abs/2104.14548) \
[A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709) \
[Exploring Simple Siamese Representation Learning (SimSiam)](https://arxiv.org/abs/2011.10566)

## Backbone Architectures
We use the following [Torchvision pre-trained models](https://github.com/pytorch/vision) as the **backbone** architecture for each SSL model \
[DenseNet-121](https://arxiv.org/abs/1608.06993) \
[GoogleNet](https://arxiv.org/abs/1409.4842v1) \
[Mnasnet-1_0](https://arxiv.org/abs/1807.11626v3) \
[MobileNet-V2](https://arxiv.org/abs/1801.04381) \
[ResNet-50](https://arxiv.org/abs/1512.03385) \
[ShuffleNet-V2_x1_0](https://arxiv.org/abs/1807.11164) \
[SqueezeNet-1_1](https://arxiv.org/abs/1602.07360)

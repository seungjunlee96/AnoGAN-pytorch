# AnoGAN-pytorch
Pytorch implementation of "Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery".<br>
Official Paper: https://arxiv.org/pdf/1703.05921.pdf <br>


# What is AnoGAN?
AnoGAN is a [deep convolutional generative adversarial network](https://arxiv.org/pdf/1703.05921.pdf) to learn a manifold of normal anatomical variability, accompanying a novel anomaly scoring scheme based on the mapping from image space to a latent space.<br>
![AnoGAN](AnoGAN_paper/imgs/AnoGAN.jpg)
<br>
This paper aims to find a point **z** in the latent space that corresponds to an image *G*(**z**) that is visually most similar to query image **x** and that is located on the manifold ***X***.<br>

## Brief procedure of AnoGAN 
1. Train **[DCGAN](https://arxiv.org/pdf/1703.05921.pdf)** with solely on image data of **healthy cases** with the aim to model the variety of healthy appearance.
2. Map new images to the *latent space* and find the most similar image *G*(**z**) via applying backpropagation iteratively.
3. Compute Anomaly score *A*(**x**)

## Loss functions of AnoGAN
- Anomaly score
- Residual Loss
- Disrimination Loss


# references
- [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/pdf/1703.05921.pdf)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- https://github.com/GunhoChoi/AnoGAN-Pytorch
- https://github.com/seokinj/anoGAN

# Webcam-Image-Quality-Optimizer
A model for improving the quality of images.

Based on the 'ESRGAN: Enhanced Super-Resolution Generative Adversial Networks' paper by Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong2, Chen Change Loy, Yu Qiao, Xiaoou Tang and uses the pretrained ESRGAN model from Tensorflow-Hub

- <a href="https://arxiv.org/pdf/1809.00219.pdf">ESRGAN: Enhanced Super-Resolution Generative Adversial Networks</a>
- <a href="https://github.com/xinntao/ESRGAN">ESRGAN Github Repository (Requires Pytorch insteda of Tensorflow)</a>
- <a href="https://tfhub.dev/captain-pool/esrgan-tf2/1">Tensorflow Model</a>

Was originally considering feeding the model a stream of images from the webcam, however as I do not have a GPU and a relatively slow CPU, the model took more than a minute per image.

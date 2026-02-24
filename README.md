# Low-Rank-Modulated Functa

This repo contains the official Pytorch implementation of the MICCAI submission 1186 *Low-Rank-Modulated Functa: Exploring the Latent Space of Implicit Neural Representations for Ultrasound Video Analysis*
 

## Data
-The Echonet Dynamic dataset can be downloaded [here](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound).

A mini-example how the data needs to be stored can be found in the folder *data*. 


### Training of the Meta-Model
Training configurations are stored in the folder *configs/experiments*
- For the training of the our LRM-Functa approach, run `python3 train.py --config ./configs/experiments/2d_imgs/echo_lrmf.yaml`.


The trained models will be stored in a folder *logs*.


### Inference and saving of the modulation vectors

- To store the modulations and reconstruct the videos, run
`python3 rescontruct.py --config ./configs/reconstruct/echo_lrmf_reconstruct.yaml`

In the yaml file, you will need to adapt the path to the right model stored in the *logs* folder. The output will be stored in a folder *reconstructions*.




## Comparing Methods

### MedFuncta
This Github repository was based on MedFuncta available [here](https://github.com/pfriedri/medfuncta).

### VidFuncta
We followed the description in the paper [VidFuncta](https://arxiv.org/abs/2507.21863) available here.

### Coin++
We followed the description in the paper [Coin++](https://arxiv.org/abs/2201.12904) .


### Convolutional Autoencoder
We followed the description in the paper [Latent Motion Profiling](https://arxiv.org/abs/2302.03130) 

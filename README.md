# Low-Rank-Modulated Functa

This repo contains the official Pytorch implementation of the MICCAI submission 1186 *Low-Rank-Modulated Functa: Exploring the Latent Space of Implicit Neural Representations for Ultrasound Video Analysis*
 

## Data
-The Echonet Dynamic dataset can be downloaded [here](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound).

A mini-example how the data needs to be stored can be found in the folder *data*. 


### Training of the Meta-Model
Training configurations are stored in the folder *configs/experiments*
- For the training of the our LRM-Functa approach, run `python3 train.py --config ./configs/experiments/cardiac.yaml` or `python3 train.py --config ./configs/experiments/lung.yaml`.


The trained models will be stored in a folder *logs*.


### Inference and saving of the modulation vectors

- To store the modulations and reconstruct the videos, run
`python3 reconstruct.py --config ./configs/reconstruct/cardiac_reconstruct.yaml` or `python3 reconstruct.py --config ./configs/reconstruct/lung_reconstruct.yaml`

In the yaml file, you will need to adapt the path to the right model stored in the *logs* folder. The output will be stored in a folder *reconstructions*.
 To compute SSIM and PSNR on the reconstructions, run `python3 compute_scores_all.py ` . The scores will be stored in a csv file.


### End-systole and End-diastolic frame selection

To find the ED and ES frames, run
`python3 downstream_tasks/eval_edes_all.py `
The results will be stored in a csv file.



### Downstream classification and regression


For the downstream regression task for ejection fraction prediction on the reconstructed videos, run
`python3 downstream_tasks/ejectionfraction_prediction.py `. You will need to specifiy the path to the reconstructed videos.




## Comparing Methods

### MedFuncta
This Github repository was based on MedFuncta available [here](https://github.com/pfriedri/medfuncta).

### VidFuncta
We followed the description in the paper [VidFuncta](https://arxiv.org/abs/2507.21863) available here. The code is available [here](https://github.com/JuliaWolleb/VidFuncta_public).

### Coin++
We followed the description in the paper [Coin++](https://arxiv.org/abs/2201.12904), and the code available in  [this repo](https://github.com/EmilienDupont/coinpp).   


### Convolutional Autoencoder
We followed the description in the paper [Latent Motion Profiling](https://papers.miccai.org/miccai-2025/0478-Paper4211.html), with the code base and implementation details [here](https://github.com/YingyuYyy/CardiacPhase).



# Matérn Kernels for Tunable Implicit Surface Reconstruction

**[Paper (arXiv)](https://arxiv.org/abs/2409.15466)**

[Maximilian Weiherer](https://mweiherer.github.io), [Bernhard Egger](https://eggerbernhard.ch)\
Friedrich-Alexander-Universität Erlangen-Nürnberg

Official implementation of the paper "Matérn Kernels for Tunable Implicit Surface Reconstruction".

This repository essentially implements [Neural Kernel Fields (NKFs)](https://research.nvidia.com/labs/toronto-ai/nkf/) proposed by [Francis Williams](https://fwilliams.info) et al. but, as opposed to NKF, uses the family of Matérn kernels instead of the first-order arc-cosine kernel (a.k.a. Neural Splines kernel).
Please see our paper for more details and why it may be a good idea to use Matérn kernels over the arc-cosine kernel.

Abstract:
*We propose to use the family of Matérn kernels for tunable implicit surface reconstruction, building upon the recent success of kernel methods for 3D reconstruction of oriented point clouds.
As we show, both, from a theoretical and practical perspective, Matérn kernels have some appealing properties which make them particularly well suited for surface reconstruction---outperforming state-of-the-art methods based on the arc-cosine kernel while being significantly easier to implement, faster to compute, and scaleable.
Being stationary, we demonstrate that the Matérn kernels' spectrum can be tuned in the same fashion as Fourier feature mappings help coordinate-based MLPs to overcome spectral bias. 
Moreover, we theoretically analyze Matérn kernel's connection to SIREN networks as well as its relation to previously employed arc-cosine kernels. 
Finally, based on recently introduced Neural Kernel Fields, we present data-dependent Matérn kernels and conclude that especially the Laplace kernel (being part of the Matérn family) is extremely competitive, performing almost on par with state-of-the-art methods in the noise-free case while having a more than five times shorter training time.*

## Setup
We're using Python 3.9, PyTorch 2.0.1, and CUDA 11.7.
To install all dependencies within a conda environment, simply run: 
```
conda env create -f environment.yaml 
conda activate learnable-matern
```
This may take a while.

For training, you'd also need a [wandb](https://wandb.ai/site) account. To log in to your account, simply type `wandb login` and follow the instructions.

## Training
We only support training on [ShapeNet](https://shapenet.org) as of now.
If you want to train a model from scratch, you first need to download the pre-processed data available [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip), taken from [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).
After downloading and extracting the data, put the `ShapeNet` folder under `../data/shapenet`. 
You may also choose a custom location; however, in this case, you'll have to update the `dataset_root` variable in the respective config file.

To start training NKF with a Matérn kernel of smoothness $\nu=1/2$ (the Laplace kernel) and $h=1$, run:
```
python train.py configs/1k_no-noise_matern12_h1.yaml
```
Similarly, if you want to train NKF using the arc-cosine (Neural Splines) kernel, run:
```
python train.py configs/1k_no-noise_neural-spline.yaml
```
### Good to Know
If you...
- have a small GPU with limited VRAM, try (i) lowering the batch size, or (ii) use gradient accumulation by setting `accumulate_grad` to true in the respective config file.
- want to train NKF with Matérn kernels for $\nu=3/2$, $\nu=5/2$, or $\nu\rightarrow\infty$ (the Gaussian kernel), change the `order` variable in the config file accordingly.
- want to increase the number of observations (from 1,000 per default), change `num_observations` in the respective config file.

## Inference
We provide pre-trained checkpoints for NKFs trained with Matérn 1/2 (and $h=1$) and Neural Splines [here](https://drive.google.com/drive/folders/1NJ_E5wEiBzE19EvLq3mgsneXNOtur3iK).
After downloading, make sure to place them in the `./checkpoints` folder.
Then, run:
```
python test.py configs/1k_no-noise_matern12_h1.yaml --ckpt checkpoints/shapenet_1k_no-noise_matern12_h1.pth
```
or
```
python test.py configs/1k_no-noise_neural-spline.yaml --ckpt checkpoints/shapenet_1k_no-noise_neural-spline.pth
```
Add `--print_metrics` if you want to print test metrics during inference. 
If you want to save the reconstructed meshes along with input point clouds to disk, add `--save_reconstructions`.

Of course, you can also test on your own trained model.
Simply type
```
python test.py <your-config-file>.yaml --ckpt <your-checkpoint>.pth
```

## Acknowledgments and Citation 
I'd like to thank Francis Williams for his tireless support during re-implementation of the NKF framework.

If you use NKF in combination with Matérn kernels or the arc-cosine kernel, please cite:
```
@misc{weiherer2024matern,
    title={Matérn Kernels for Tunable Implicit Surface Reconstruction},
    author={Weiherer, Maximilian and Egger, Bernhard},
    archivePrefix={arXiv},
    eprint={2409.15466},
    year={2024}
}
```
and
```bibtex
@inproceedings{williams2022nkf,
  title={Neural Fields as Learnable Kernels for 3D Reconstruction},
  author={Williams, Francis and Gojcic, Zan and Khamis, Sameh and Zorin, Denis and Bruna, Joan and Fidler, Sanja and Litany, Or},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18500--18510},
  year={2022}
}
```
Also, in case you have any questions, feel free to contact the authors.

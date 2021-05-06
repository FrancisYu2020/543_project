Repository for UIUC CS543 Computer Vision course project.
# Surface normal reconstruction pipeline #

This is a pipeline for getting 3D reconstruction imags from 2D images using the method in [this paper](https://openreview.net/forum?id=FGqiDsBUKL0). Most code are borrowed from the original implementation [GitHub repo](https://github.com/XingangPan/GAN2Shape). The pretrained [StyleGAN2](https://github.com/NVlabs/stylegan2) models are already downloaded in GAN2Shape/checkpoints/stylegan2/*.pt.

To do the 3D reconstruction using your own dataset, there are two steps to go. The first is GAN inversion since the paper uses encoded GAN latent vector as input for the training. However, the latent vectors are not kept so we need to invert the image back to the latent vector. This is done by GAN2Shape/projector.py which is modified from [here](https://github.com/rosinality/stylegan2-pytorch/blob/master/projector.py). The latent vectors will be put under the same directory of image data. Since the latent vector file is not consistent with the input of the GAN2Shape, we implemented process_latent.py to convert the data directory consistent with the input requirement of GAN2Shape. The second step is to set the configurations in GAN2Shape/configs and the bash scripts in GAN2Shape/scripts.

## Run the pipeline script ##

First create a folder named "data" and "results" under the root directory. Create folder "checkpoints" under directory "GAN2Shape" and put the corresponding StyleGAN pretrained weights under the directory. Put your own image dataset in a folder and put the folder under "data" folder, select the matched config files in GAN2Shape/configs or make your own configs in "GAN2Shape/configs/surface_normal.yml" and simply run
```sh
sh quick_run.sh
```

The results are located under specified directory in "results" folder under the root directory. Then run
```sh
python get_normal_output.py -i [results folder] -o [folder to save selected surface normal]
```

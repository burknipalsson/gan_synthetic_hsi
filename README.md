## Python sources for the method proposed in
# Synthesis of Synthetic Hyperspectral Images with Controllable Spectral Variability using a Generative Adversarial Network
The code uses the Pytorch and Pytorch Lightning frameworks. The requirements needed to run the code is in the file requirements.txt. The method requires the hyperspectral images to be in Matlab mat files having the following named variables:

| Variable | Content |
| --- | ----------- |
| Y | Array having dimensions B x P containing the spectra |
| GT | Array having dimensions R x B containing the reference endmembers |
|cols | The number of columns in the hyperspectral image (HSI) |
|rows | The number of rows in the HSI |

Here, R is the number of endmembers, B the number of bands, and P the number of pixels. Edit the wgan_purity.yaml file under the configs directory to change hyperparameters. Run the train_cwgan.py to train the model. Edit and run the file create_synthetic_hsi.py to generate synthetic HSIs. If you use the sources provided, make sure you cite the paper:

Palsson, B.; Ulfarsson, M.O.; Sveinsson, J.R. Synthesis of Synthetic Hyperspectral Images with Controllable Spectral Variability Using a Generative Adversarial Network. Remote Sens. 2023, 15, 3919. https://doi.org/10.3390/rs15163919

## Bibtext

@Article{rs15163919,
AUTHOR = {Palsson, Burkni and Ulfarsson, Magnus O. and Sveinsson, Johannes R.},
TITLE = {Synthesis of Synthetic Hyperspectral Images with Controllable Spectral Variability Using a Generative Adversarial Network},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {16},
ARTICLE-NUMBER = {3919},
URL = {https://www.mdpi.com/2072-4292/15/16/3919},
ISSN = {2072-4292},
ABSTRACT = {In hyperspectral unmixing (HU), spectral variability in hyperspectral images (HSIs) is a major challenge which has received a lot of attention over the last few years. Here, we propose a method utilizing a generative adversarial network (GAN) for creating synthetic HSIs having a controllable degree of realistic spectral variability from existing HSIs with established ground truth abundance maps. Such synthetic images can be a valuable tool when developing HU methods that can deal with spectral variability. We use a variational autoencoder (VAE) to investigate how the variability in the synthesized images differs from the original images and perform blind unmixing experiments on the generated images to illustrate the effect of increasing the variability.},
DOI = {10.3390/rs15163919}
}


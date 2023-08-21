import numpy as np
from PIL import Image
from pytorch_lightning.core import datamodule
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from train_cwgan import CWGAN_GP_TV, Generator, HSIDataModule
from scipy.stats import truncnorm
from models.types_ import *
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from scipy.io import savemat,loadmat
from sklearn.decomposition import PCA

def decode_one_hot(code:Tensor)->np.array:
    decoded = torch.argmax(code,dim=1).numpy()
    return decoded.astype(int)

class Sampler(object):
    """
    Class for sampling from a trained GAN and generating synthetic HSIs

    Args:
        path (str): Path to the trained GAN
        latent_dim (int): Dimension of the latent space
        num_endmembers (int): Number of endmembers in the HSI
        datamodule (HSIDataModule): DataModule for the HSI

    Methods:
        load_model: Loads the trained GAN
        sample_noise: Samples noise from a truncated normal distribution
        generate_endmember_matrices: Generates endmember matrices from the GAN
        generate_endmembers_from_labels: Generates endmembers from labels (class, purity)
        generate_hsi: Generates an HSI from the GAN

    """
    def __init__(
        self,
        path: str,
        latent_dim: int,
        num_endmembers: int,
        datamodule: HSIDataModule,
    ):
        super(Sampler, self).__init__()
        self.num_endmembers = num_endmembers
        self.latent_dim = latent_dim
        self.datamodule = datamodule
        self.gan = self.load_model(path)
        self.gan.eval()
        self.gen = self.gan.gen

    def load_model(self, path: str) -> CWGAN_GP_TV:
        return torch.load(path)

    def sample_noise(self, truncation: float, labels: Tensor) -> Tensor:
        n_samples = labels.size()[0]
        truncated_noise = torch.as_tensor(
            truncnorm.rvs(-truncation, truncation, size=(n_samples, self.latent_dim))
        )
        labeled_noise = torch.cat([truncated_noise, labels], dim=-1)
        return labeled_noise.float()

    def generate_endmember_matrices(
        self, num_samples, truncation: float, purity: float
    ) -> Tensor:
        labels = []
        endmembers = []
        if purity is not None:
            for i in range(self.num_endmembers):
                # labels=torch.tile(F.one_hot(torch.LongTensor([i]),self.num_endmembers),[num_samples,1])*purity
                l = torch.tile(
                    F.one_hot(torch.LongTensor([i]), self.num_endmembers),
                    [num_samples, 1],
                ) * ((1 - purity) * torch.rand([num_samples, 1]) + purity)
                noise = self.sample_noise(truncation, l)
                labels.append(l.cpu().detach().numpy())
                endmembers.append(self.gen(noise).cpu().detach().numpy())
        else:
            for i in range(self.num_endmembers):
                l = torch.tile(
                    F.one_hot(torch.LongTensor([i]), self.num_endmembers),
                    [num_samples, 1],
                )
                noise = self.sample_noise(truncation, l)
                labels.append(l.cpu().detach().numpy())
                endmembers.append(self.gen(noise).cpu().detach().numpy())
        return np.asarray(endmembers), np.asarray(labels)

    def generate_endmembers_from_labels(self, labels, truncation, purity: float):
        if purity is not None:
            purity_noise = (1 - purity) * torch.rand([labels.size()[0], 1]) + purity
            labels = labels * purity_noise
        noise = self.sample_noise(truncation, labels)
        endmembers = self.gen(noise)
        return endmembers.cpu().detach().numpy()

    def generate_hsi(self, truncation: float, purity):
        abunds = self.datamodule.abundances
        n_pixels = np.max(abunds.shape)
        endmembers, labels = self.generate_endmember_matrices(
            num_samples=n_pixels, truncation=truncation, purity=purity
        )
        HSI = np.zeros([n_pixels, endmembers[0].shape[1]])
        for i in range(n_pixels):
            HSI[i, :] = np.matmul(abunds[i, :], endmembers[:, i, :])
        return HSI# /np.max(HSI)

def save_hsi_to_file(name, Y, dm: HSIDataModule):
    savemat(
        name,
        {
            'Y': Y.T,
            'GT': dm.gt.cpu().detach().numpy(),
            'S_GT': dm.abundances,
            'cols': dm.abundances.shape[1],
            'lines': dm.abundances.shape[0],
        },
    )

def main():
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    num_endmembers = 4
    latent_dim = 10
    threshold = 0.3
    orig_dataset='./Datasets/Urban4.mat'
    

    dm_orig = HSIDataModule(orig_dataset, threshold, 64)
    dm_orig.prepare_data()
    dm_orig.setup(stage='fit')
    sampler = Sampler('./generator.pt', 10, 4, dm_orig)
    #Generate synthetic HSI with truncation 0.4 and purity 0.9
    synthetic_hsi = sampler.generate_hsi(0.4,0.9)
    save_hsi_to_file(name='./hsis/synthetic_urban.mat',Y=synthetic_hsi,dm=dm_orig)
    

if __name__ == "__main__":
    main()

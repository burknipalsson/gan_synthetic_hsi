import numpy as np
import yaml
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import io
import os
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch.backends.cudnn as cudnn
from scipy import io as sio
from scipy.stats import truncnorm
from torch.utils.data import DataLoader, Dataset, random_split
from models.types_ import *
from typing import List, Callable, Union, Any, TypeVar, Tuple
from collections import OrderedDict
import matplotlib.pyplot as plt

def decode_one_hot(code:Tensor)->np.array:
    decoded = torch.argmax(code,dim=1).numpy()
    return decoded.astype(int)

def make_plots(samples:List):
    fig = plt.figure(figsize=(9,9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col')
    fig.suptitle('Sample spectra')
    ax1.plot(samples[0].cpu().detach().numpy())
    ax2.plot(samples[1].cpu().detach().numpy())
    ax3.plot(samples[2].cpu().detach().numpy())
    ax4.plot(samples[3].cpu().detach().numpy())

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    buf.close()
    return im

class HSIDataset(Dataset):
    def __init__(self, spectra:Tensor, labels:Tensor):
        self.spectra = spectra
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [self.spectra[idx],self.labels[idx]]
         
def combine_tensors(x:Tensor, y:Tensor):
        return torch.cat([x.float(),y.float()],dim=1).float()

class HSIDataModule(LightningDataModule):
    def __init__(self, data_path: str, threshold: float, batch_size: int):
        super(HSIDataModule,self).__init__()
        self.path = data_path
        #Threshold controls the purity of the pixels used for training the GAN
        self.threshold = threshold
        self.batch_size = batch_size

        self.labels = None
        self.spectra = None
        self.pure_spectra = None
        self.pure_labels = None
        self.hsi_full = None
        self.hsi_train = None
        self.hsi_test = None
        self.hsi_pure = None
        self.gt = None
        self.num_pixels = 0
        self.abundances = None
        self.hsi_shape = None

    def prepare_data(self):
        data = sio.loadmat(self.path)
        spectra = data['Y'].astype(np.float32)
        abundances = data['S_GT'].astype(np.float32)
        self.hsi_shape = (int(data['lines'][0]), int(data['cols'][0]), np.min(spectra.shape))
        abundances = np.transpose(abundances, [1, 0, 2])
        self.gt = data['GT'].astype(np.float32)
        self.abundances = np.reshape(abundances, (abundances.shape[0] * abundances.shape[1], abundances.shape[-1]))
        spectra = np.transpose(spectra)
        spectra = spectra + np.abs(np.min(spectra))
        spectra = spectra / np.max(spectra)
        self.num_pixels = np.max(spectra.shape)
        # permutation = np.random.permutation(np.max(spectra.shape))
        # spectra = spectra[permutation,:]
        # abundances = abundances[permutation,:]
        rows, cols = np.where(self.abundances > self.threshold)
        purity = torch.FloatTensor(np.around(np.max(self.abundances[rows,:], axis=1),decimals=2))
        
        self.spectra = torch.FloatTensor(spectra[rows, :])
        one_hot = F.one_hot(torch.LongTensor(cols), num_classes=cols.max() + 1).float()
        self.labels = torch.unsqueeze(purity,dim=1)*one_hot
        self.gt = torch.Tensor(self.gt)
        self.hsi_full = HSIDataset(self.spectra, self.labels)
        rows, cols = np.where(self.abundances > 0.99)
        self.pure_spectra = torch.FloatTensor(spectra[rows, :])
        one_hot = F.one_hot(torch.LongTensor(cols), num_classes=cols.max() + 1)
        purity = torch.FloatTensor(np.around(np.max(self.abundances[rows,:], axis=1),decimals=1))
        self.pure_labels = torch.unsqueeze(purity,dim=1)*one_hot
        self.hsi_pure = HSIDataset(self.pure_spectra, self.pure_labels)

    def setup(self, **kwargs):
        self.hsi_train=self.hsi_full

    def train_dataloader(self):
        return DataLoader(self.hsi_train, batch_size=self.batch_size, num_workers=4)

    def teardown(self, **kwargs):
        del self.hsi_test
        del self.hsi_full
        del self.hsi_train
        del self.hsi_pure

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        noise_dim: the latent dimension of noise
        hidden_dims: the inner dimensions, a list
        num_bands: The size of output layer
    '''
    def __init__(self, noise_dim, hidden_dims, num_bands):
        super(Generator, self).__init__()
        self.latent_dim = noise_dim
        self.hidden_dims = hidden_dims
        self.num_bands = num_bands
        self.gen = self.make_generator_net()

    def make_generator_block(self, input_dim, output_dim, is_final=False):
        if is_final==False:
            block = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=output_dim),
                nn.Dropout(0.1),
                nn.LeakyReLU(0.2)
            )
        else:
            block = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=output_dim),
                nn.LeakyReLU(0.05)
            )

        return block

    def make_generator_conv_block(self, input_dim, output_dim, is_final=False):
        if is_final==False:
            block = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=output_dim, stride=1,kernel_size=3),
                nn.LeakyReLU(0.1)
            )
        else:
            block = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=output_dim, stride=1,kernel_size=2),
                nn.Flatten()
            )
        return block

    def make_generator_net(self):
        dims = [self.latent_dim] + self.hidden_dims + [self.num_bands]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(
                self.make_generator_block(dims[i], dims[i + 1])
            )
        layers.append(self.make_generator_block(dims[-2],dims[-1],True))
        return nn.Sequential(*layers)

    def forward(self, noise,labels):
        labelled_noise = combine_tensors(noise,labels)
        generated = self.gen(labelled_noise)
        return generated

class Critic(nn.Module):
    '''
    Critic Class
    Values:
        input_dim: the number of bands in the spectrum,
        hidden_dims: the inner dimensions, a list
    '''

    def __init__(self, input_dim, hidden_dims,output_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.crit = self.make_critic()

    def make_crit_block(self, input_dim, output_dim, final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=output_dim),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=output_dim)
            )

    def make_critic(self):
        dims = [self.input_dim] + self.hidden_dims
        layers = []
        for i in range(len(dims) - 2):
            layers.append(
                self.make_crit_block(dims[i], dims[i + 1])
            )
        layers.append(self.make_crit_block(dims[-2], self.output_dim, True))
        return nn.Sequential(*layers)

    def forward(self, spectra,labels):

        labelled = combine_tensors(spectra,labels)
        validity = self.crit(labelled)
        return validity.view(len(validity), -1)

class CWGAN_GP_TV(LightningModule):
    """Class WGAN_GP contains the Generator and Critic and implements
    the cost function and training
    """

    def __init__(self, params):
        super(CWGAN_GP_TV, self).__init__()
        #initalize variables
        self.automatic_optimization = False
        self.latent_dim = params['latent_dim']
        self.num_bands = params['num_bands']
        self.hidden_dims_crit = params['hidden_dims_crit']
        self.hidden_dims_gen = params['hidden_dims_gen']
        self.truncation = params['truncation']
        self.lambda_gp = params['lambda_gp']
        self.n_critic = params['n_critic']
        self.tv = params['tv']
        self.num_classes = params['num_classes']
        self.params = params
        self.gen = Generator(self.latent_dim+self.num_classes, self.hidden_dims_gen, self.num_bands)
        self.crit = Critic(self.num_bands+self.num_classes, self.hidden_dims_crit,self.latent_dim)

        self.validation_z = self.get_truncated_noise(self.num_classes,self.latent_dim,self.truncation)
        self.validation_labels = torch.eye(self.num_classes).float().cuda()
        #self.example_input_array = [torch.zeros(10, self.latent_dim),{'labels':torch.zeros(10,self.num_classes)}]

    def forward(self, spectra, **kwargs):
        labels = torch.zeros(input.size()[0],4)
        if 'labels' in kwargs.keys():
            labels = kwargs['labels']
        return self.gen(spectra,labels)

    def total_variation(self,x):
        return torch.sum(torch.abs(x[:,:-1]-x[:,1:]))

    def get_truncated_noise(self, n_samples, z_dim, truncation):
        
        truncated_noise = torch.as_tensor(truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim)).astype(np.float32))
      
        return truncated_noise

    def get_gradient(self, real, fake, labels, epsilon):
        '''
        Return the gradient of the critic's scores with respect to mixes of real and fake images.
        Parameters:
            crit: the critic model
            real: a batch of real spectra
            fake: a batch of fake spectra
            epsilon: a vector of the uniformly random proportions of real/fake per mixed spectra
        Returns:
            gradient: the gradient of the critic's scores, with respect to the mixed spectrum
        '''
        # Mix the images together
        mixed_spectra = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = self.crit(mixed_spectra,labels)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=mixed_spectra,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    def gradient_penalty(self, gradient):
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)

        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)

        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1) ** 2)

        return penalty

    def get_gen_loss(self, crit_fake_pred):
       
        gen_loss = -torch.mean(crit_fake_pred)

        return gen_loss

    def get_crit_loss(self, crit_fake_pred, crit_real_pred, gp):
   
        crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + self.lambda_gp * gp

        return crit_loss
    def combine_tensors(self,x:Tensor, y:Tensor):
        return torch.cat([x.float(),y.float()],dim=1).float()

    def training_step(self, batch, batch_idx):
        spectra, labels = batch
        g_opt, d_opt = self.optimizers()
        #Sample noise
        z = self.get_truncated_noise(spectra.shape[0], self.latent_dim,self.truncation)
        z = z.type_as(spectra)

        #Train Generator
      

        # generate images
        generated_spectra = self.gen(z,labels)

        # log sampled images
        # sample_imgs = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('generated_images', grid, 0)
        g_loss = self.get_gen_loss(self.crit(generated_spectra,labels))+self.tv*self.total_variation(generated_spectra)

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
    

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples

        fake_spectra = self.gen(z,labels)
        # Real images
        real_validity = self.crit(spectra,labels)
        # Fake images
        fake_validity = self.crit(fake_spectra,labels)

        # Gradient penalty
        epsilon = torch.rand(len(real_validity), 1, device=self.device, requires_grad=True)
        gradient = self.get_gradient(spectra, fake_spectra, labels, epsilon)
        gradient_penalty = self.gradient_penalty(gradient)

        # Adversarial loss
        d_loss = self.get_crit_loss(fake_validity, real_validity, gradient_penalty)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log_dict({"g_loss":g_loss,"d_loss":d_loss},prog_bar=True)

    def configure_optimizers(self):
        lr = self.params['lr']
        b1 = self.params['b1']
        b2 = self.params['b2']

        g_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        d_opt = torch.optim.Adam(self.crit.parameters(), lr=lr, betas=(b1, b2))
        return g_opt, d_opt

    def on_train_epoch_end(self):
        if self.current_epoch % 5 ==0:
            z = self.validation_z.to(self.device)
            # log sampled images
            sample_imgs = self.gen(z,self.validation_labels)
            images = make_plots(sample_imgs)
            self.logger.experiment.add_image('generated_images', images, self.current_epoch)

def main():
    #set the config file to use
    config_file = './configs/wgan_purity.yaml'

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        version=config['logging_params']['version'],
        default_hp_metric=False,
        log_graph=True
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = CWGAN_GP_TV(config['model_params'])
    data_module = HSIDataModule(**config['datamodule_params'])

    runner = Trainer(accelerator="gpu",default_root_dir=f"{tb_logger.save_dir}",
                    min_epochs=1,
                    logger=tb_logger,
                    log_every_n_steps=250,
                    **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    
    if os.path.exists('./logs'):
        shutil.rmtree('./logs')
    runner.fit(model, data_module)
    torch.save(model.gen, './generator.pt')

if __name__ == "__main__":
    main()

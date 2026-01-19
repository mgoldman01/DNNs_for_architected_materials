
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SamplingLayer(nn.Module):
    def forward(self, mean, log_var):
        epsilon = torch.randn_like(log_var) # samples are of len(log_var)
        return mean + torch.exp(0.5 * log_var) * epsilon


class TGVAE(nn.Module):
    def __init__(self, latent_dim, matprops):
    
        super(TGVAE, self).__init__()

        # Encoder - convolutional layers perform feature extraction to produce a feature vector
        self.encoder = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=5, padding=2),   
                nn.ReLU(),
                nn.Conv3d(64, 32, kernel_size=5, padding=0), 
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 32, kernel_size=3, padding=0), 
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, kernel_size=4, padding=0),
                nn.ReLU(),
                nn.Conv3d(64, 128, kernel_size=4, padding=0),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(128, 256, kernel_size=3, padding=1),
                nn.Conv3d(256, 512, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.AvgPool3d(2),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
        )
        
        # Producing the mean and log(variance) vectors as inputs to the latent space via reparameterization
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)
        
        # Predicting the material property - volume fraction - through producing the mean and log(variance) vectors
        self.matprop_mean =  nn.Sequential(nn.Linear(256, latent_dim), 
                                     nn.ReLU(),
                                     nn.Linear(latent_dim, len(matprops))
                                    )
                                     
        self.matprop_log_var = nn.Sequential(nn.Linear(256, latent_dim),
                                       nn.ReLU(),
                                       nn.Linear(latent_dim, len(matprops))
                                      )
                                     
        
        self.sampling = SamplingLayer()
        
        # Decoder - convolution transpose and upsampling layers reconstruct the input based on its latent vector
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim+len(matprops), 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Unflatten(1, (512, 1,1,1)),

                nn.Upsample(scale_factor=2, mode='trilinear'), 
                nn.ConvTranspose3d(512, 256, kernel_size=3, padding=0), 
                nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode='trilinear'),   
                nn.ConvTranspose3d(128, 64, kernel_size=4, padding=0), 
                nn.ConvTranspose3d(64, 32, kernel_size=4, padding=0),   
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='trilinear'),  

                nn.ConvTranspose3d(32, 32, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='trilinear'),  
                nn.ConvTranspose3d(32, 16, kernel_size=5, padding=0),
                nn.ConvTranspose3d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
    
    def forward(self, x):
        mean_log_var = self.encoder(x)
        z_mean = self.z_mean(mean_log_var)
        z_log_var = self.z_log_var(mean_log_var)

        # produce latent vector through reparameterization trick                               
        z = self.sampling(z_mean, z_log_var)
        
        matprop_mean = self.matprop_mean(mean_log_var)
        matprop_log_var = self.matprop_log_var(mean_log_var)

        # "predict" material properties using reparameterization
        m = self.sampling(matprop_mean, matprop_log_var)
        
        zm = torch.cat((z,m), dim=1)
        
        # reconstruct from the latent vector and the material property vector (volume fraction)
        reconstructed = self.decoder(zm)
        
        
        return reconstructed, z_mean, z_log_var, m, z, matprop_mean, matprop_log_var

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

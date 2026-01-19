import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Linear_Feature_Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear_Feature_Attention, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Attention weights implemented by sigmoid function
        attn_weights = torch.sigmoid(self.fc(x))
        attended_features = x * attn_weights

        return attended_features

class MatPropPredictor(nn.Module):
    def __init__(self, in_dim, num_matprops):
        super(MatPropPredictor, self).__init__()
        
        self.attention_layer = Linear_Feature_Attention(in_dim, in_dim)
        
        self.mlp = nn.Sequential(
                            nn.Linear(in_dim, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, 256),
                            nn.ReLU(),
                            nn.Linear(256, num_matprops)
                        )
        
    def forward(self, x):
        matprops_group = self.mlp(self.attention_layer(x))

        return matprops_group
        
class SamplingLayer(nn.Module):
    def forward(self, mean, log_var):
        epsilon = torch.randn_like(log_var) # samples are of len(log_var)
        return mean + torch.exp(0.5 * log_var) * epsilon
    

class TOVAE(nn.Module):
    def __init__(self, latent_dim, matprops, matprops_segmented):
        outdims = [len(segment) for segment in matprops_segmented]
    
        super(TOVAE, self).__init__()
        # self.latent_dim = latent_dim

        # Encoder
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
        
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)
        
        
        self.vf_mean =  nn.Sequential(nn.Linear(256, latent_dim), 
                                     nn.ReLU(),
                                     nn.Linear(latent_dim, 1)
                                    )
                                     
        self.vf_log_var = nn.Sequential(nn.Linear(256, latent_dim),
                                       nn.ReLU(),
                                       nn.Linear(latent_dim, 1)
                                      )
        
        self.mpp0 = MatPropPredictor(latent_dim, outdims[0])
        self.mpp1 = MatPropPredictor(latent_dim, outdims[1])
        self.mpp2 = MatPropPredictor(latent_dim, outdims[2])
        self.mpp3 = MatPropPredictor(latent_dim, outdims[3])
        self.mpp4 = MatPropPredictor(latent_dim, outdims[4])
        self.mpp5 = MatPropPredictor(latent_dim, outdims[5])
        self.mpp6 = MatPropPredictor(latent_dim, outdims[6])
        self.mpp7 = MatPropPredictor(latent_dim, outdims[7])
        

        self.sampling = SamplingLayer()
        
        # Decoder
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim+1, 256),
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
                                       
        z = self.sampling(z_mean, z_log_var)
        
        vf_mean = self.vf_mean(mean_log_var)
        vf_log_var = self.vf_log_var(mean_log_var)
        vf = self.sampling(vf_mean, vf_log_var)
        
                
        # mpv = self.mpv_MLP(self.mpv_attention(z))
        # mpv = self.mpv_MLP(z)
        
        mpv0 = self.mpp0(z)
        mpv1 = self.mpp1(z)
        mpv2 = self.mpp2(z)
        mpv3 = self.mpp3(z)
        mpv4 = self.mpp4(z)
        mpv5 = self.mpp5(z)
        mpv6 = self.mpp6(z)
        mpv7 = self.mpp7(z)
        
        fullmpv = torch.cat([mpv0, mpv1, mpv2, mpv3, mpv4, mpv5, mpv6, mpv7], axis=1)
        
        zvf = torch.cat((z,vf), dim=1)
        reconstructed = self.decoder(zvf)
        
        # reconstructed = self.decoder(z)
        
        
        return reconstructed, z, z_mean, z_log_var, fullmpv, vf

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
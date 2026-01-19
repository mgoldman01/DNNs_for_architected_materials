import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual  # Add the residual connection
        x = self.relu(x)
        return x


class Linear_Feature_Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear_Feature_Attention, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Attention weights implemented by sigmoid function
        attn_weights = torch.sigmoid(self.fc(x))
        attended_features = x * attn_weights

        return attended_features
    

class MatProp_Pred_Module(nn.Module):
    def __init__(self, output_dimension=1):
        super(MatProp_Pred_Module, self).__init__()

        self.out_dim = output_dimension

        self.lin_attn = Linear_Feature_Attention(1024, 1024)

        self.mpp_mod =  nn.Sequential(
                        nn.Linear(1024, 1024),nn.ReLU(),
                        nn.Linear(1024, 512),nn.ReLU(),
                        nn.Linear(512, 256),nn.ReLU(),
                        nn.Linear(256, 128),nn.ReLU(),
                        nn.Linear(128, self.out_dim))
        
    def forward(self, feature_vector):
        matprops = self.mpp_mod(self.lin_attn(feature_vector))

        return matprops


class MatProp_CNN3D_varmod(nn.Module): # "varmod" = variable number of modules
    def __init__(self, param_segments): #, params#
        outdims = [len(segment) for segment in param_segments]
        
        super(MatProp_CNN3D_varmod, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, padding=2),   
            nn.BatchNorm3d(16),   
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=5, padding=0), 
            nn.BatchNorm3d(32),
            nn.ReLU(),
            ResidualBlock(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, padding=0), 
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=4, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ResidualBlock(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=4, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.Conv3d(512, 1024, kernel_size=3, padding=0),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.AvgPool3d(2)
        )

        self.flatten = nn.Flatten()

        self.mpps = nn.ModuleList()
        
        for i in range(len(outdims)):
            mpp = MatProp_Pred_Module(outdims[i])

            self.mpps.append(mpp)

    def forward(self, x):
            
        x = self.convolutions(x)
        x = self.flatten(x)
        
        mpv_parts = []

        for mpp in self.mpps:
            sub_mpv = mpp(x)
            mpv_parts.append(sub_mpv)

        output = torch.cat(mpv_parts, axis=1)
        return output      
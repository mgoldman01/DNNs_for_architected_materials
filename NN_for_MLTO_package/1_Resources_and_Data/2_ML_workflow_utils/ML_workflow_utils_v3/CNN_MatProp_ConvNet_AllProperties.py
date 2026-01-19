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
        
class MatProp_CNN3D_all_matprops(nn.Module):
    def __init__(self, param_segments): #, params#
        outdims = [len(segment) for segment in param_segments]
        
        super(MatProp_CNN3D_all_matprops, self).__init__()

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
        
        self.attention0 = Linear_Feature_Attention(1024, 1024)
        self.attention1 = Linear_Feature_Attention(1024, 1024)
        self.attention2 = Linear_Feature_Attention(1024, 1024)
        self.attention3 = Linear_Feature_Attention(1024, 1024)
        self.attention4 = Linear_Feature_Attention(1024, 1024)
        self.attention5 = Linear_Feature_Attention(1024, 1024)
        self.attention6 = Linear_Feature_Attention(1024, 1024)
        self.attention7 = Linear_Feature_Attention(1024, 1024)
        
        

        self.mlp0 = nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[0])
        )
            
        self.mlp1= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[1])
        )
            
        self.mlp2= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[2])
        )
        
        self.mlp3= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[3])
        )
        
        self.mlp4= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[4])
        )
        
        self.mlp5= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[5])
        )
        
        self.mlp6= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[6])
        )
        
        self.mlp7= nn.Sequential(
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, outdims[7])
        )


    def forward(self, x):
            
        x = self.convolutions(x)
        x = self.flatten(x)
        
        params0 = self.mlp0(self.attention0(x))
        params1 = self.mlp1(self.attention1(x))
        params2 = self.mlp2(self.attention2(x))
        params3 = self.mlp3(self.attention3(x))
        params4 = self.mlp4(self.attention4(x))
        params5 = self.mlp5(self.attention5(x))
        params6 = self.mlp6(self.attention6(x))
        params7 = self.mlp7(self.attention7(x))
        
        
        
        output = torch.cat([params0, params1, params2, params3, params4, params5, params6, params7], axis=1)
        return output
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RVAEDataset(data.Dataset):


    def __init__(self, split_dataframe, array_directory, matprops):

        super().__init__()
        
        self.df = split_dataframe
        self.data_dir = array_directory
        self.matprops = matprops


    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return len(self.df)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        array_filename = self.df['full PN'].iloc[idx] + '.npz'
        array_filepath = os.path.join(self.data_dir, array_filename)
        array = np.load(array_filepath)
        array = array['arr_0']
        
        array = np.expand_dims(array, 0).astype(np.float32)
        mpvec = np.asarray(self.df[self.matprops].iloc[idx])
        
        return array, mpvec
    
    
# Define your training and validation functions
def tgvae_train(net, dataloader, optimizer, beta, alpha=1.0):
    net.train()  # Set the model to training mode
    running_loss = 0.0
    pbar = tqdm(dataloader)  # Use tqdm for progress bars
    
    for voxelmesh, matpropvec in pbar:
        voxelmesh = voxelmesh.to(device)  
        matpropvec = matpropvec.to(device).to(dtype=torch.float32)
        optimizer.zero_grad()
        
        reconstructed, z_mean, z_log_var, mpv_pred, *_ = net(voxelmesh)
        # reconstructed, z_mean, z_log_var, m, z, matprop_mean, matprop_log_var - output from TGVAE
        loss = TGVAE_Loss(reconstructed, voxelmesh, z_mean, z_log_var, matpropvec, mpv_pred, beta=beta, alpha=alpha)
        loss[0].to(dtype=torch.float32).backward()
        optimizer.step()
        running_loss += loss[0].item()
        pbar.set_description(f'Train Loss: {running_loss / (pbar.n + 1):.4f}')
    return running_loss / len(dataloader), loss[1], loss[2], loss[3]


def tgvae_validate(net, dataloader, beta, alpha=1.0):
    net.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    pbar = tqdm(dataloader)  # Use tqdm for progress bars
    with torch.no_grad():
        for voxelmesh, matpropvec in pbar:
            voxelmesh = voxelmesh.to(device)  # Move inputs to GPU
            matpropvec = matpropvec.to(device)
            reconstructed, z_mean, z_log_var, mpv_pred, *_ = net(voxelmesh)
            loss = TGVAE_Loss(reconstructed, voxelmesh, z_mean, z_log_var, matpropvec, mpv_pred, beta=beta, alpha=alpha)
            running_loss += loss[0].item()
            pbar.set_description(f'Val Loss: {running_loss / (pbar.n + 1):.4f}')
    return running_loss / len(dataloader), loss[1], loss[2], loss[3]


alpha = 1
beta = 0.01


def TGVAE_Loss(reconstructed, input_voxelmesh_array, z_mean, z_log_var, matprop_vector, mpv_pred, beta=0.01, alpha=1):
    """
    Loss function for RVAE

    Args:
        reconstructed (_type_): _description_
        input_voxelmesh_array (_type_): _description_
        z_mean (_type_): _description_
        z_log_var (_type_): _description_
        matprop_vector (_type_): _description_
        mpv (_type_): _description_
        beta (float, optional): _description_. Defaults to 0.01.
        alpha (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    
    # reconstruction_loss = BCEpenalized()(reconstructed, input_voxelmesh_array)
    reconstruction_loss = nn.BCELoss()(reconstructed, input_voxelmesh_array)
    
    # regression_loss = Regression_Loss(r, r_mean, r_log_var, param_vector)
    regression_loss = nn.L1Loss()(mpv_pred, matprop_vector)
    
    kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1),dim=0)
    
    # loss_value =  reconstruction_loss + (1-alpha)*kl_loss + beta*regression_loss
    loss_value =  reconstruction_loss + beta*kl_loss + alpha*regression_loss
    
    
    return loss_value, reconstruction_loss, kl_loss, regression_loss

    
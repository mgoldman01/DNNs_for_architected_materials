import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define your training and validation functions
def tovae_train(net, dataloader, optimizer, beta, alpha):
    net.train()  # Set the model to training mode
    running_loss = 0.0
    pbar = tqdm(dataloader)  # Use tqdm for progress bars
    
    for voxel, paramvec in pbar:
        voxel = voxel.to(device)  # Move inputs to GPU
        paramvec = paramvec.to(device).to(dtype=torch.float32)
        optimizer.zero_grad()
        # reconstructed, z, z_mean, z_log_var, mpv, vf
        reconstructed, z, z_mean, z_log_var, mpv, vf = net(voxel)
        loss = TOVAE_loss(reconstructed, voxel, z_mean, z_log_var, paramvec, mpv, vf, beta=beta, alpha=alpha)
        loss[0].to(dtype=torch.float32).backward()
        optimizer.step()
        running_loss += loss[0].item()
        pbar.set_description(f'Train Loss: {running_loss / (pbar.n + 1):.4f}')
    return running_loss / len(dataloader), loss[1], loss[2], loss[3], loss[4]


def tovae_validate(net, dataloader, beta, alpha):
    net.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    pbar = tqdm(dataloader)  # Use tqdm for progress bars
    with torch.no_grad():
        for voxel, paramvec in pbar:
            voxel = voxel.to(device)  # Move inputs to GPU
            paramvec = paramvec.to(device)
            reconstructed, z, z_mean, z_log_var, mpv, vf = net(voxel)
            loss = TOVAE_loss(reconstructed, voxel, z_mean, z_log_var, paramvec, mpv, vf, beta=beta, alpha=alpha)
            running_loss += loss[0].item()
            pbar.set_description(f'Val Loss: {running_loss / (pbar.n + 1):.4f}')
    return running_loss / len(dataloader), loss[1], loss[2], loss[3], loss[4]

# alpha = 5
# beta = 0.01
def TOVAE_loss(reconstructed, input_voxel_array, z_mean, z_log_var, param_vector, mpv, vf, beta=0.01, alpha=1):
    
    reconstruction_loss = nn.BCELoss()(reconstructed, input_voxel_array)
    
    regression_loss = nn.L1Loss()(mpv, param_vector)
    
    vf_loss = nn.L1Loss()(vf, param_vector[:, 1].unsqueeze(1))
    
    kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1),dim=0)
    
    loss_value =  reconstruction_loss + beta*kl_loss + alpha*regression_loss + vf_loss
    
    
    return loss_value, reconstruction_loss, kl_loss, regression_loss, vf_loss
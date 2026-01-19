import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class VoxelDataset(Dataset):
    def __init__(self, split_dataframe, array_directory, params):
        self.df = split_dataframe
        self.data_dir = array_directory
        self.params = params


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        array_filename = self.df['full PN'].iloc[idx] + '.npz'
        
        translate_by = self.df['translate_by'].iloc[idx]

        array_filepath = os.path.join(self.data_dir, array_filename)
        # array = np.load(array_filepath)
        array = np.load(array_filepath)['arr_0']        
        # augment the data by translating the array along an axis
        array = np.roll(array, int(array.shape[0]*translate_by), axis=0)
        array = np.expand_dims(array, 0).astype(np.float32)#).to(dtype=torch.float32)
        param_vec = np.asarray(self.df[self.params].iloc[idx])
                   
        return array, param_vec
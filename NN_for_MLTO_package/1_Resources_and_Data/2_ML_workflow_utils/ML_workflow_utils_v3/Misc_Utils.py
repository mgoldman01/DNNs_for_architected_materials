import pandas as pd
import pickle


def unscale(df, colname, value):
    max = df[colname].max()
    min = df[colname].min()
    unscaled = value*(max - min) + min
    return unscaled

def scale(df, colname, value):
    max = df[colname].max()
    min = df[colname].min()
    scaled = (value - min) / (max - min)
    return scaled




def save_dict_to_pickle(file_path, data_dict):
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)

def load_dict_from_pickle(file_path):
    with open(file_path, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    return loaded_dict
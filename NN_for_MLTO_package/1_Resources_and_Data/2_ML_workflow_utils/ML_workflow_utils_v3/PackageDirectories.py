# rootpath = '/home/mgolub4/DLproj/MLTO_2024/'

import os
currentpath = os.getcwd()
os.chdir('../../..')
path = os.getcwd()
#print(path)
os.chdir(currentpath)

class PackageDirectories():

    def __init__(self, rootpath=f'{path}/'): #

        
        # Directory in which the package is located
        self.rootpath = rootpath
        
        # Directories of the structure of the ML package:
        # Entire ML package 
        self.pkgpath = os.path.join(f'{self.rootpath}', 'ML_package_3_SEP24')
        
        # 0_Functional_Notebooks - code notebooks
        self.notebookpath = os.path.join(f'{self.pkgpath}', '0_Functional_Notebooks')
        
        # Paths to folders for each code notebook - used for saving outputs
        self.nb_1_1_path = os.path.join(f'{self.notebookpath}', '1_1_Train_Property_Prediction_CNN')
        self.nb_1_2_path = os.path.join(f'{self.notebookpath}', '1_2_Predict_Material_Properties_with_CNN')
        self.nb_2_1_path = os.path.join(f'{self.notebookpath}', '2_1_Train_TopoGen_VAE')
        self.nb_2_2_path = os.path.join(f'{self.notebookpath}', '2_2_Generate_Synthetic_Topologies')
        self.nb_3_1_path = os.path.join(f'{self.notebookpath}', '3_1_Train_TO_VAE')
        self.nb_3_2_path = os.path.join(f'{self.notebookpath}', '3_2_TO_with_TO_VAE')


        # 1_Resources_and_Data - pre-built models, custom functions and classes, prepared data
        self.resourcepath  = os.path.join(f'{self.pkgpath}', '1_Resources_and_Data')


        
        # 1_pre-built_models - Paths to pre-trained models

            # Path to pre-trained convolutional neural network for regression
        self.convnetpath      =  os.path.join(f'{self.resourcepath}', '1_pre-trained_models/1_Property_Prediction_CNN')

            # Path to pre-trained Topology-Generating Variational Autoencoder (TopoGen-VAE)
        self.topogen_vae_path =  os.path.join(f'{self.resourcepath}', '1_pre-trained_models/2_Topo_Gen_VAE')

            # Path to pre-trained Topology-Optimizing Variational Autoencoder (TO-VAE)
        self.topopt_vae_path  =  os.path.join(f'{self.resourcepath}', '1_pre-trained_models/3_TO_VAE')
        
            # 2_ML_workflow_data_utilities - Path to data utilities - custom classes and functions
        self.data_utils_path  =  os.path.join(f'{self.resourcepath}', '2_ML_workflow_data_utilities')

            # 3_data_prepped - Paths to included data files
        self.source_data_path =  os.path.join(f'{self.resourcepath}', '3_dataset')

                # Voxel_Topos_by_PN - voxel topologies by part number, in compressed numpy format (.npz)
        self.voxeltopo_path   =  os.path.join(f'{self.source_data_path}', 'voxel_topos_by_partno')



        
        
        
        

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tt_split
import random
from scipy.io import loadmat
import os
import glob
import re
from math import floor
import sys
import json

# from ML_workflow_utils_v3.PackageDirectories import PackageDirectories as PD


# This code automatically sets the rootpath as the directory the entire package is contained in, which is then called to initialize the PackageDirectories class below
# The current working directory (cwd) will be called within the notebook, so the package's directory will already be set in the 
# import os
# currentpath = os.getcwd()
# print(currentpath)
# os.chdir('../../../')
# path = os.getcwd()
# print(path)
# os.chdir(currentpath)
# rootpath = os.getcwd() #'/mnt/c/Users/matth/OneDrive - Johns Hopkins/MLTO project/_ML package SEP 24 FINAL/' #'/home/mgolub4/DLproj/' # Change this to the parent directory of the entire package
# print(rootpath)
# # Module contains strings of all directories of this package
# directory = PD(rootpath=rootpath) 

# data_source_directory = directory.source_data_path
# meshdir = directory.voxeltopo_path

class Dataset_Preprocessor():
    """
    Performs train/test/validation splits of the input data in accordance with the indexing by topology family, cell type, and volume fraction

    """


    def __init__(self,
                csv_fn = 'data_full.csv', #partnum_dict_filename = 'part_number_lookup.json',
                 data_source_directory = None,
                 meshdir = None,
                 volfrac_range=(0.01, 0.75)):
        """
        
        Initialize the Dataset_Preprocessor -- needed is the name of the database, the database directory path, the voxel topology database path, and the volume fraction range

        Args:
            csv_fn (str, optional): **Mandatory** - Filename of the CSV of the material proeprty database. Defaults to 'data_full.csv'.
            data_source_directory (_type_, optional): **Mandatory** -  Directory in the package that contains the database. Specify as path or as the ".source_data_path" attribute of an instance of the PackageDirectories class. Defaults to None.
            meshdir (_type_, optional): **Not mandatory since I don't think it's called anywhere in here, but I'm too nervous to delete it.** Directory in the package that contains the voxel topologies. Specify as path or as the ".voxel_topo_path" attribute of an instance of the PackageDirectories class. Defaults to None.
            volfrac_range (tuple, optional): Range of volume fractions to include in the dataset splits. Defaults to (0.01, 0.75).
        """
        
        self.csv_fn     = csv_fn
        # print(self.csv_fn)
        self.csvdir     = data_source_directory
        # print(self.csvdir)
        self.volfrac_range = volfrac_range

        self.csvpath    = os.path.join(self.csvdir, self.csv_fn)

        # self.datadir    = created_directory
        self.topfams      = sorted(['tubeplt', 'tpms', 'synth', 'topopt', 'interp', 'lattice']) # topfams = topology families
        self.partitions   = ['tr','val', 'te'] # tr - train/training; val - validation; te - test/testing

        self.matpropcsv   = self.LoadSpreadsheet(self.csvpath, volfrac_range = self.volfrac_range)
        self.lookup_dict, self.topfams_partnums, self.topfams_strings, self.celltypes_partnums, self.celltypes_strings = self.GenerateLookupDict(self.matpropcsv)

        #self.partnum_dict = self.LoadPartnumDict(data_source_directory, partnum_dict_filename)


    def LoadSpreadsheet(self, csvpath, truncate_for_test=False, truncate_pct=0.4, volfrac_range=(0.01, 0.75)):
        """
        

        Args:
            csvpath (str): path to csv of data
            truncate_for_test (bool, optional): flag to truncate the data for testing. Defaults to False.
            truncate_pct (float, optional): truncate by percentage. Defaults to 0.4.
            volfrac_range (tuple, optional): volume fraction range. Defaults to (0.01, 0.75).

        Returns:
            pd.DataFrame: dataframe of material properties
        """
        # print(truncate_for_test)
        vfr = volfrac_range

        pncols = [('topology_family PN', 2), ('cell_type PN', 4), ('volfrac_index PN', 3)]

        if truncate_for_test:
            matpropcsv = pd.read_csv(csvpath)
            matpropcsv = matpropcsv[:floor(len(matpropcsv)*truncate_pct)]
        else:
            matpropcsv = pd.read_csv(csvpath)

        matpropcsv = matpropcsv[matpropcsv.volFrac.between(vfr[0],vfr[1])]

        #Sorting values to align with part numberign convention
        matpropcsv = matpropcsv.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'],axis=1)

        for col in pncols:
            matpropcsv[col[0]] = matpropcsv[col[0]].astype(str).str.zfill(col[1])
        

        # print(matpropcsv.topology_family.unique())
        # print(len(list(matpropcsv.cell_type.unique())))
        return matpropcsv
    
    # def LoadPartnumDict(self, source_dir, json_filename):
    #     json_filepath = os.path.join(source_dir, json_filename)

    #     with open(json_filepath, 'r') as f:
    #         partnum_dict = json.load(f)
    #     f.close()

    #     return partnum_dict

    def GenerateLookupDict(self, csv: pd.DataFrame, cols = ['topology_family', 'cell_type']):

        lookup_dict = {}

        for col in cols:
            category_col = csv[col]

            partnum_col = csv[f'{col} PN']

            unique_pairs = {partnum: topo for (partnum, topo) in (sorted(set(zip(partnum_col, category_col))))}

            lookup_dict[col] = unique_pairs

        topfams_partnums = list(lookup_dict['topology_family'].keys())
        topfams_strings = list(lookup_dict['topology_family'].values())

        celltypes_partnums = list(lookup_dict['cell_type'].keys())
        celltypes_strings = list(lookup_dict['cell_type'].values())

        return lookup_dict, topfams_partnums, topfams_strings, celltypes_partnums, celltypes_strings
    
    def Lookup_Partnum(self, input_type = 'full PN', value = '01-0001-001', target_lookup = ['topology_family',]):

        # target_PN_cols = [elem + ' PN' for elem in target_lookup]
        # target_PN_cols.extend(target_lookup)

        partnum_info = self.matpropcsv.loc[self.matpropcsv[input_type] == value].iloc[0]

        # return partnum_info[target_PN_cols]
        return partnum_info[target_lookup]



        
    def TrainTestSplit(self, random_state=42, train_size=0.8, rem_split=0.5, 
                       omit_tf=False, tfomit=['lattice'], subset_tf=False, 
                       topfam_sampling_set=['lattice', 'topopt', 'tpms'], topfams_for_test_set=['tpms',], 
                       test_set_topo_counts=[2,2,2], testset_seed=10,
                       translate=True, translate_by=(0.25, 0.5),
                       topfam_translate_omit=None):
        
        csv = self.matpropcsv


        self.testsubset = self.PickLeaveOut(test_set_topfams=topfam_sampling_set, subset_entire_topfam = subset_tf, topfams_for_test_set=topfams_for_test_set, test_set_topo_counts=test_set_topo_counts, set_seed=True, 
                            seed=testset_seed)
        
        # if subset_topfam = True: -->> in this spot, change from csv[csv['cell_type']==celltype] to csv[csv['topology_family']==topfam]... save the lines of code

        self.idxTe = pd.DataFrame()
        self.ttsplit_df = csv
        for celltype in self.testsubset:
            concatdf = csv[csv['cell_type PN']==celltype[1]]
            self.idxTe = pd.concat([self.idxTe, concatdf])
            # self.ttsplit_df = self.ttsplit_df.drop(self.ttsplit_df[self.ttsplit_df.cell_type==celltype])
            self.ttsplit_df = self.ttsplit_df[self.ttsplit_df['cell_type PN'] != celltype[1]]

        # if 'plttube' in topfam_sampling_set:
        #     self.plttubedf = self.ttsplit_df[self.ttsplit_df.topology_family == 'plttube']
        #     self.ttsplit_df = self.ttsplit_df[self.ttsplit_df.topology_family != 'plttube']
        
        #     self.idxTr, self.idxVal = tt_split(self.ttsplit_df, stratify = self.ttsplit_df['cell_type'], random_state=random_state, train_size=train_size)
        #     plttb_Tr, plttb_Val = tt_split(self.plttubedf, stratify = None, random_state=random_state, train_size=train_size)

        #     self.idxTr = pd.concat([self.idxTr, plttb_Tr])
        #     self.idxVal = pd.concat([self.idxVal, plttb_Val])


        #     self.idxTr  = self.idxTr.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)
        #     self.idxVal = self.idxVal.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)
        #     self.idxTe  = self.idxTe.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)

        # else:

        #     self.idxTr, self.idxVal = tt_split(self.ttsplit_df, stratify = self.ttsplit_df['cell_type'], random_state=random_state, train_size=train_size)


        #     self.idxTr  = self.idxTr.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)
        #     self.idxVal = self.idxVal.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)
        #     self.idxTe  = self.idxTe.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)

        self.idxTr, self.idxVal = tt_split(self.ttsplit_df, stratify = None, shuffle=True,  random_state=random_state, train_size=train_size)


        self.idxTr  = self.idxTr.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)
        self.idxVal = self.idxVal.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)
        self.idxTe  = self.idxTe.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)


        
        if translate == True:
            translate_by = (0.0,) + translate_by
            for split_name in ['idxTr', 'idxVal', 'idxTe']:
            # for split in self.splitdfs:
                if topfam_translate_omit is not None:

                    df_to_not_augment = getattr(self, split_name)[getattr(self, split_name)['topology_family'].isin(topfam_translate_omit)]

                    df_to_augment = getattr(self, split_name)[~getattr(self, split_name)['topology_family'].isin(topfam_translate_omit)]


                    orig_len = len(df_to_augment)

                    augmented_df = pd.concat([getattr(self, split_name)]*len(translate_by), ignore_index=True).sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)


                    augmented_df = augmented_df.assign(translate_by=pd.Series(translate_by * orig_len))

                    augmented_df = pd.concat([augmented_df, df_to_not_augment])

                    augmented_df.sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)

                    augmented_df['translate_by'] = augmented_df['translate_by'].fillna(0.0)

                    setattr(self, split_name, augmented_df)


                else:
                    orig_len = len(getattr(self, split_name))

                    augmented_df = pd.concat([getattr(self, split_name)]*len(translate_by), ignore_index=True).sort_values(by=['topology_family','cell_type','volFrac']).reset_index().drop(['index'], axis=1)


                    augmented_df = augmented_df.assign(translate_by=pd.Series(translate_by * orig_len))

                    setattr(self, split_name, augmented_df)

        else:
            translate_by = (0.0,)
            for split_name in ['idxTr', 'idxVal', 'idxTe']:
                
                split_df = getattr(self, split_name)
                orig_len = len(getattr(self, split_name))

                split_df = split_df.assign(translate_by=pd.Series(translate_by*orig_len))

                setattr(self, split_name, split_df)


    
    def PickLeaveOut(self,  set_seed=True, seed = 10, test_set_topo_counts = [2,2,2], test_set_topfams=['lattice', 'topopt', 'tpms'],
                     subset_entire_topfam=False, topfams_for_test_set = ['lattice',]):


        if set_seed:
            random.seed(seed)
        else:
            pass

        self.testsubsetlen = dict(zip(test_set_topfams, test_set_topo_counts))

        testsubset = []


        if subset_entire_topfam:
            for topfam in topfams_for_test_set:
                testsubset.append(topfam)
        else:
            for fam in test_set_topfams:
                topfam_celltypes = list(zip(self.matpropcsv[self.matpropcsv.topology_family==fam]['cell_type'].unique(), self.matpropcsv[self.matpropcsv.topology_family==fam]['cell_type PN'].unique()))
                picks = random.sample(list(topfam_celltypes), self.testsubsetlen[fam])
                # print(picks)



                if len(picks) >1:
                    for pick in picks:
                        testsubset.append(pick)
                else:
                    testsubset = picks

        return testsubset

    


    def SelectMatPropsToDF(self, df: pd.DataFrame, matprops=['CH_11 scaled',]):

        return df[[matprops]]
        

"""

!!!!!! 27MAR24: I think for now I'm going to skip data augmentation through translation -- too many irregular topologies in the new group, and 
originally we had only about 5-6k data points so I wanted to augment the data to get more robust models, but now with about 50k data points, I
am going to cautiously assume that's not a problem anymore  
"""


"""

What does the interface with the torch.data.Dataset need to do?

It needs to load the numpy files and, in order, pull the matpropeters and 

"""